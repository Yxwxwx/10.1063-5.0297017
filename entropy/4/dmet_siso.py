from pyscf import gto, scf, ao2mo, mcscf, fci, dmrgscf, lib

# from pyscf.csf_fci import csf_solver  # TODO:
from pyscf.mcscf import avas
from pyscf.tools import molden, fcidump
from pyscf.lib import logger, cho_solve
from pyscf.data import nist
import numpy as np
import scipy
from functools import reduce
import os
import tempfile
from liblan.utils.rdiis import tag_rdiis_
import h5py
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = ""


def read_cas_info(fname):
    with open(fname, "r") as f1:
        ncasorb, ncaselec = [int(x) for x in f1.readline().split(" ")]
        casorbind = [int(x) for x in f1.readline().split(" ")]
        statelis = [int(x) for x in f1.readline().split(" ")]
    return ncasorb, ncaselec, casorbind, statelis


def sacasscf_dump_chk(solver, sav):
    # saving properties for SISO
    with h5py.File(sav, "w") as fh5:
        fh5["ncore"] = np.asarray(solver.ncore)
        fh5["ncas"] = np.asarray(solver.ncas)
        fh5["nelecas"] = np.asarray(solver.nelecas)
        fh5["mo_coeff"] = solver.mo_coeff
        fh5["rdm1"] = solver.make_rdm1()

        for i in range(len(solver.ci)):
            fh5["ci" + str(i)] = solver.ci[i]
        fh5["e_states"] = solver.e_states


def sacasscf_load_chk(solver, sav):
    with h5py.File(sav, "r") as fh5:
        solver.ncore = fh5["ncore"][()]
        solver.ncas = fh5["ncas"][()]
        solver.nelecas = fh5["nelecas"][:]

        solver.mo_coeff = fh5["mo_coeff"][:]
        rdm1 = fh5["rdm1"][:]
        solver.make_rdm1 = lambda *args: rdm1

        # solver.e_states = fh5["e_states"][:]
        # solver.ci = [fh5["ci" + str(i)][:] for i in range(len(solver.e_states))]
    return solver


def rohf_solve_imp(hcore, eris, nelec, spin, max_mem, dm0=None, verbose=logger.INFO):
    mol = gto.M()
    mol.verbose = verbose
    mol.incore_anyway = True
    mol.nelectron = nelec
    mol.spin = spin

    nao = hcore.shape[0]
    print("nao = ", nao)
    print("nelec = ", mol.nelec)

    mf = scf.rohf.ROHF(mol).x2c()
    mf.max_memory = max_mem
    mf.mo_energy = np.zeros((nao))
    mf.max_cycle = 1000
    mf.diis_space = 8
    mf.init_guess = "1e"
    mf.level_shift = 0.2

    mf.get_hcore = lambda *args: hcore
    mf.get_ovlp = lambda *args: np.eye(nao)
    mf._eri = ao2mo.restore(8, eris, nao)

    if dm0 is None:
        mf.kernel()
    else:
        if dm0.shape != hcore.shape:
            bath_size = hcore.shape[0] - dm0.shape[0]
            dm0 = scipy.linalg.block_diag(
                dm0, np.eye(bath_size) * (nelec - np.trace(dm0)) / bath_size
            )
        mf.kernel(dm0)
    return mf


def sacasscf_solve_imp(
    solver_base, mol, ncasorb, ncaselec, casorbind, statelis, avas=False
):
    solver = mcscf.CASSCF(solver_base, ncasorb, ncaselec)

    if avas:
        mo = casorbind
    else:
        mo = solver.sort_mo(casorbind)
    mcfs = []
    logger.info(solver, "Attempting SA-DMRGSCF with")
    for i in range(len(statelis)):
        if statelis[i] != 0:
            new_mcf = dmrgscf.DMRGCI(mol, maxM=1500, tol=1e-10)
            new_mcf.spin = i
            new_mcf.nroots = statelis[i]
            new_mcf.threads = 16
            new_mcf.memory = int(mol.max_memory / 1000)
            new_mcf.block_extra_keyword = [
                "real_density_matrix",
                "davidson_soft_max_iter 1600",
                "noreorder",
                "cutoff 1E-24",
            ]
            new_mcf.runtimeDir = lib.param.TMPDIR + "/%d" % i
            new_mcf.scratchDirectory = lib.param.TMPDIR + "/%d" % i
            mcfs.append(new_mcf)
            logger.info(
                solver, "%s states with spin multiplicity %s", statelis[i], i + 1
            )

    statetot = sum(statelis)
    mcscf.state_average_mix_(solver, mcfs, np.ones(statetot) / statetot)
    solver.run(mo, verbose=mol.verbose)
    return solver


def lowdin(s):
    """new basis is |mu> c^{lowdin}_{mu i}"""
    e, v = scipy.linalg.eigh(s)
    idx = e > 1e-15
    return np.dot(v[:, idx] / np.sqrt(e[idx]), v[:, idx].conj().T)


def caolo(s):
    return lowdin(s)


def cloao(s):
    return lowdin(s) @ s


def get_dmet_as_props(mf, imp_inds, lo_meth="lowdin", thres=1e-13):
    """
    Returns C(AO->AS), entropy loss, and orbital composition
    """
    s = mf.get_ovlp()
    caolo_mat, cloao_mat = caolo(s), cloao(s)
    env_inds = [x for x in range(s.shape[0]) if x not in imp_inds]

    if mf.mol.spin == 0:
        pass
    else:
        dma, dmb = mf.make_rdm1()

        ldma = reduce(np.dot, (cloao_mat, dma, cloao_mat.conj().T))
        ldmb = reduce(np.dot, (cloao_mat, dmb, cloao_mat.conj().T))

        ldma_env = ldma[env_inds, :][:, env_inds]
        ldmb_env = ldmb[env_inds, :][:, env_inds]

        nat_occa, nat_coeffa = np.linalg.eigh(ldma_env)
        nat_occb, nat_coeffb = np.linalg.eigh(ldmb_env)

        ldmr_env = ldma_env + ldmb_env
        nat_occr, nat_coeffr = np.linalg.eigh(ldmr_env)

        caoas = np.hstack([caolo_mat[:, imp_inds], caolo_mat[:, env_inds] @ nat_coeffr])

        nuu = np.sum([nat_occr < thres])
        ne = np.sum([(nat_occr >= thres) & (nat_occr <= 2 - thres)])
        nuo = np.sum([nat_occr > 2 - thres])

        asorbs = (nuu, ne, nuo)
        print("caoas.shape = ", caoas.shape)
        print("asorbs = ", asorbs)

        nat_occa = nat_occa[nat_occa > thres]
        nat_occb = nat_occb[nat_occb > thres]
        nat_occr = nat_occr[nat_occr > thres]

        ent = -np.sum(nat_occa * np.log(nat_occa)) - np.sum(nat_occb * np.log(nat_occb))
        entr = -2 * np.sum(nat_occr / 2 * np.log(nat_occr / 2))

        return caoas, entr - ent, asorbs


def get_dmet_imp_ldm(mf, imp_inds, lo_meth="lowdin"):
    """
    Returns better initial guess than '1e' for impurity
    """
    s = mf.get_ovlp()
    caolo_mat, cloao_mat = caolo(s), cloao(s)

    dma, dmb = mf.make_rdm1()

    ldm = reduce(np.dot, (cloao_mat, (dma + dmb), cloao_mat.conj().T))[imp_inds, :][
        :, imp_inds
    ]
    return ldm


def get_as_1e_ints(mf, caoas, asorbs):
    # hcore from mf
    hcore = mf.get_hcore()

    # HF J/K from env UO
    uos = hcore.shape[0] - asorbs[-1]
    dm_uo = caoas[:, uos:] @ caoas[:, uos:].conj().T * 2
    vj, vk = mf.get_jk(dm=dm_uo)

    fock = hcore + vj - 0.5 * vk

    nimp = hcore.shape[0] - np.sum(asorbs)
    act_inds = [
        *list(range(nimp)),
        *list(range(nimp + asorbs[0], nimp + asorbs[0] + asorbs[1])),
    ]

    asints1e = reduce(np.dot, (caoas[:, act_inds].conj().T, fock, caoas[:, act_inds]))
    return asints1e


def get_as_2e_ints(mf, caoas, asorbs):
    nimp = caoas.shape[0] - np.sum(asorbs)
    act_inds = [
        *list(range(nimp)),
        *list(range(nimp + asorbs[0], nimp + asorbs[0] + asorbs[1])),
    ]

    if mf._eri is not None:
        asints2e = ao2mo.full(mf._eri, caoas[:, act_inds])
    else:
        asints2e = ao2mo.full(mf.mol, caoas[:, act_inds])

    return asints2e


def get_bp_hso2e(mol, dm0):
    hso2e = mol.intor("int2e_p1vxp1", 3).reshape(3, mol.nao, mol.nao, mol.nao, mol.nao)
    vj = np.einsum("yijkl,lk->yij", hso2e, dm0, optimize=True)
    vk = np.einsum("yijkl,jk->yil", hso2e, dm0, optimize=True)
    vk += np.einsum("yijkl,li->ykj", hso2e, dm0, optimize=True)
    return vj, vk


def get_bp_hso2e_amfi(mol, dm0):
    """atomic-mean-field approximation"""
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    vj = np.zeros((3, nao, nao))
    vk = np.zeros((3, nao, nao))
    import copy

    atom = copy.copy(mol)
    aoslice = mol.aoslice_by_atom(ao_loc)
    for ia in range(mol.natm):
        b0, b1, p0, p1 = aoslice[ia]
        atom._bas = mol._bas[b0:b1]
        vj1, vk1 = get_bp_hso2e(atom, dm0[p0:p1, p0:p1])
        vj[:, p0:p1, p0:p1] = vj1
        vk[:, p0:p1, p0:p1] = vk1
    return vj, vk


mol = gto.M(
    atom="""
                Co    3.946594   13.913238   30.684257
                P    3.512134   16.084388   28.030428
                S    3.288337   17.313925   26.593099
                N    3.366331   14.435231   27.565538
                N    3.616930   13.494506   28.544517
                N    2.390738   16.330189   29.327326
                N   2.371977   15.382532   30.351455
                N    4.986512   16.188715   28.690866
                N    5.265253   15.509610   29.883762
                C    3.807224   12.252707   28.193902
                H    3.764341   11.980779   27.304913
                C    4.072564   11.365834   29.211803
                N    4.300381   11.867388   30.500917
                C    4.659528   10.834421   31.288709
                H    4.900746   10.884186   32.184236
                N    4.605924    9.690018   30.523337
                H    4.753335    8.893610   30.811051
                C    4.276259   10.022195   29.222080
                H    4.206574    9.442795   28.497498
                C    3.409215   14.166859   26.176161
                H    3.263143   14.980862   25.688850
                H    2.724423   13.532362   25.950410
                H    4.268219   13.806067   25.947296
                C    1.468750   15.421632   31.251653
                H    0.797360   16.063238   31.222072
                N    2.512688   13.450607   32.199496
                C    1.510293   14.477886   32.283568
                C    0.763857   14.216623   33.407027
                H    0.041543   14.719599   33.709999
                N    1.271755   13.066888   34.016398
                H    0.966212   12.693477   34.728214
                C    2.347855   12.636604   33.272198
                H    2.881215   11.902579   33.473352
                C    1.313298   17.310902   29.269097
                H    1.460709   17.904522   28.528635
                H    1.295877   17.813879   30.085536
                H    0.474395   16.857691   29.154509
                C    6.273008   15.848185   30.627648
                H    6.868012   16.521781   30.387575
                N    5.432766   14.205071   32.134415
                C    6.401658   15.128378   31.815875
                C    7.243240   15.309663   32.884531
                H    7.974935   15.881953   32.925321
                N    6.794307   14.481441   33.882816
                H    7.138713   14.399685   34.665935
                C    5.684704   13.802512   33.400487
                H    5.191547   13.171571   33.871918
                C    5.971486   17.286020   28.385713
                H    5.746349   17.694799   27.547789
                H    6.855952   16.916342   28.332467
                H    5.942003   17.945398   29.082893
        """,
    basis={"default": "def2tzvp", "C": "6-31G*", "H": "6-31G*"},
    symmetry=0,
    spin=3,
    charge=+2,
    verbose=4,
    max_memory=200000,
)

mf = scf.rohf.ROHF(mol).x2c()
mf.max_cycle = 100
title = "CoSMM10"
imp_inds = mol.search_ao_label(["Co *"])
mf.chkfile = title + "_rohf.chk"
mf.init_guess = "atom"
if os.path.isfile(title + "_rohf.chk"):
    mf.init_guess = "chk"
mf = tag_rdiis_(mf=mf, imp_inds=imp_inds)
mf.kernel()

print("nao = ", mol.nao_nr())
print("nelec = ", mol.nelec)


print("imp_inds.shape", imp_inds.shape)

as_fname = title + "_as_chk.h5"
if not os.path.isfile(as_fname):
    # AS general info
    caoas, ent, asorbs = get_dmet_as_props(mf, imp_inds, thres=1e-13)
    # AS hcore + j/k
    as1e = get_as_1e_ints(mf, caoas, asorbs)
    # AS ERIs
    as2e = get_as_2e_ints(mf, caoas, asorbs)

    print(f"Entanglement S: {ent:.3f}")

    with h5py.File(as_fname, "w") as fh5:
        fh5["caoas"] = caoas
        fh5["ent"] = ent
        fh5["asorbs"] = asorbs
        fh5["1e"] = as1e
        fh5["2e"] = as2e
else:
    fh5 = h5py.File(as_fname, "r")
    caoas = fh5["caoas"][:]
    ent = fh5["ent"][()]
    asorbs = fh5["asorbs"][:]
    as1e = fh5["1e"][:]
    as2e = fh5["2e"][:]
    fh5.close()

nimp = caoas.shape[0] - np.sum(asorbs)
print("nimp = ", nimp)
act_inds = [
    *list(range(nimp)),
    *list(range(nimp + asorbs[0], nimp + asorbs[0] + asorbs[1])),
]

print("act_inds = ", act_inds)

nelec = int(mf.mol.nelectron - asorbs[-1] * 2)
ldm = get_dmet_imp_ldm(mf, imp_inds)
solver_base = rohf_solve_imp(as1e, as2e, nelec, mf.mol.spin, mf.max_memory, dm0=ldm)

with open(title + "_imp_rohf_orbs.molden", "w") as f1:
    molden.header(mf.mol, f1)
    molden.orbital_coeff(
        mf.mol,
        f1,
        caoas[:, act_inds] @ solver_base.mo_coeff,
        ene=solver_base.mo_energy,
        occ=solver_base.mo_occ,
    )

# avas
# ao_labels = ["Co 3d"]
# ncasorb, ncaselec, casorbind = avas.avas(solver_base, ao_labels, canonicalize=False)

# ncasorb = 5
# ncaselec = 7
# casorbind = [40, 41, 55, 56, 57]  # F-style

caschk_fname = title + "_imp_cas_chk.h5"


if not os.path.isfile(caschk_fname):
    casinfo_fname = title + "_cas_info"
    if not os.path.isfile(casinfo_fname):
        print("Failed to read saved CAS briefings.")
        exit()
    ncasorb, ncaselec, casorbind, statelis = read_cas_info(casinfo_fname)

    solver = sacasscf_solve_imp(
        solver_base, mf.mol, ncasorb, ncaselec, casorbind, statelis=statelis
    )
    sacasscf_dump_chk(solver, caschk_fname)

else:
    solver = sacasscf_load_chk(lib.StreamObject(), caschk_fname)


with open(title + "_imp_casscf_orbs.molden", "w") as f1:
    molden.header(mf.mol, f1)
    molden.orbital_coeff(
        mf.mol,
        f1,
        caoas[:, act_inds] @ solver.mo_coeff,
        ene=solver_base.mo_energy,
        occ=solver_base.mo_occ,
    )

huge_casorbind = [
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
]
if len(huge_casorbind) != 22:
    exit()

mc = mcscf.CASSCF(solver_base, ncas=22, nelecas=31)
mc.mo_coeff = mc.sort_mo(huge_casorbind, solver.mo_coeff)

bond_dims = [250] * 4 + [500] * 4 + [2000]
noises = [1e-4] * 4 + [1e-5] * 4 + [0]
thrds = [1e-10] * 8
ncas = 22
n_elec = 31
spin = 3
ecore = mc.get_h1eff()[1]
h1e = mc.get_h1eff()[0]
g2e = mc.get_h2eff()
orb_sym = [0] * ncas
# ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(
#     mf, ncore=0, ncas=None, g2e_symm=8
# )
driver = DMRGDriver(
    scratch="./tmp",
    symm_type=SymmetryTypes.SZ,
    n_threads=32,
    stack_mem=int(200 * 1024**3),
)
driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)

mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0)
ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
energy = driver.dmrg(
    mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises, thrds=thrds, iprint=1
)
print("DMRG energy = %20.15f" % energy)

ordm1 = driver.get_orbital_entropies(ket, orb_type=1)
ordm2 = driver.get_orbital_entropies(ket, orb_type=2)
minfo = 0.5 * (ordm1[:, None] + ordm1[None, :] - ordm2) * (1 - np.identity(len(ordm1)))

with h5py.File("minfo_data.h5", "w") as f:
    f.create_dataset("minfo", data=minfo)
    f.create_dataset("ordm1", data=ordm1)
    f.create_dataset("ordm2", data=ordm2)

    # 可选：写入注释
    f.attrs["description"] = (
        "Mutual information matrix computed from 1-RDM and 2-RDM orbital entropies."
    )
    f.attrs["formula"] = (
        "minfo = 0.5 * (ordm1[:, None] + ordm1[None, :] - ordm2) * (1 - I)"
    )

print("已保存为 minfo_data.h5")

import matplotlib.pyplot as plt

if not os.path.exists("minfo_data.h5"):
    print("minfo_data.h5 not found. Please run the DMRG calculation first.")
    exit()

with h5py.File("minfo_data.h5", "r") as f:
    minfo = f["minfo"][:]
    # ordm1 = f["ordm1"][:]   # 可选
    # ordm2 = f["ordm2"][:]   # 可选

fig, ax = plt.subplots(figsize=(6, 6))  # 图像尺寸可调整
cax = ax.imshow(minfo, cmap="ocean_r")  # 使用你原来的颜色风格

# 添加 colorbar
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Mutual Information", fontsize=10)

# 设置坐标轴刻度为 1~N（从1开始）
n = minfo.shape[0]
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(np.arange(1, n + 1))
ax.set_yticklabels(np.arange(1, n + 1))
ax.tick_params(length=0, labelsize=8)

# 添加细网格线
ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3)
ax.tick_params(which="minor", bottom=False, left=False)

# 保持正方形比例
ax.set_aspect("equal")

# 保存图片
plt.tight_layout()
plt.savefig("minfo_matrix_oceanr.png", dpi=600)
plt.close()
