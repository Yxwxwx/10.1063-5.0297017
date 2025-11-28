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
  Cr  -1.35424450239280      0.00748246778796     -0.00087595645706 
  Cr   1.35423764131845      0.00747803055009     -0.00087750355078 
  O   -0.00000294430974     -0.13734660180713      1.46342333077295 
  O   -0.00000064383679      1.35420350893235     -0.59959802230353 
  O   -0.00000516089286     -1.18217624605433     -0.86562559483160 
  H   -0.00000065463206      0.52075486357441      2.16950923336362 
  H    0.00000446473449      1.64878725909317     -1.51878718478551 
  H   -0.00000926896581     -2.12530440890393     -0.66035454411104 
  N   -2.68346617865321     -1.45853826988486      0.70650005032200 
  C   -2.01822727274717     -2.52465540625361      1.50592497177303 
  C   -3.25911308583331     -2.04955652529081     -0.54278797058362 
  C   -3.74827281074952     -0.80152247456954      1.54881850901734 
  H   -1.56748557922213     -2.08271923805020      2.39129571316802 
  H   -1.24243826043958     -2.99862043316970      0.90648770606852 
  H   -2.74840997387176     -3.28356694917433      1.79607780436372 
  H   -2.46291984415168     -2.62337513441757     -1.02093781896581 
  H   -4.07044743926176     -2.74311239041734     -0.30244269679464 
  H   -4.65448299937734     -0.70680487372819      0.95372217138132 
  H   -3.99500774926338     -1.44490669699879      2.39456976728496 
  N   -2.69755450329625      1.34136962911627      0.91026422821942 
  N   -2.69230526823111      0.11718842081111     -1.61665366474216 
  C   -3.74849616889413     -0.94865197019848     -1.46353644818261 
  C   -3.27063403754831      0.54952804549553      2.04434124343513 
  N    2.68346715675698     -1.45854069314242      0.70650003406182 
  N    2.69754925024949      1.34136611134027      0.91026654207433 
  N    2.69230403104748      0.11718547928447     -1.61665262689138 
  C   -2.04437616396754      2.57225987860668      1.43663608392829 
  C   -3.76194284730143      1.73300596201565     -0.08408294061084 
  C   -2.03228717940052     -0.03618283988186     -2.94309134997238 
  C   -3.27725483448617      1.49033558111027     -1.50018850579138 
  H   -4.65645471974087     -0.48734410738601     -1.07991760798513 
  H   -3.99664215151405     -1.36168809799084     -2.44226987052617 
  H   -2.47549715669751      0.42734931293811      2.78234874695929
  H   -4.08755174458298      1.09782217898898      2.52290059499979 
  C    2.01823189480386     -2.52466013170271      1.50592432110894 
  C    3.25911692194754     -2.04955805616288     -0.54278693277931 
  C    3.74827350165531     -0.80152393465555      1.54881874867098 
  C    2.04437438874437      2.57225673960399      1.43664047370217 
  C    3.27063260709400      0.54952529364165      2.04434238389264 
  C    3.76193758070494      1.73300459398023     -0.08408076294943 
  C    2.03228955044192     -0.03618719210796     -2.94309210428304 
  C    3.27724995734855      1.49033424715459     -1.50018655506513 
  C    3.74849826690828     -0.94865216224951     -1.46353430216360 
  H   -1.59575047514265      3.12234814848417      0.61291937461478 
  H   -1.26826445350661      2.29664085726858      2.14911186944439 
  H   -2.78162146845096      3.19639860592609      1.94666596437982 
  H   -4.66409353100933      1.16248658840109      0.12799867351868 
  H   -4.01827672254151      2.78498397644774      0.04889460213876 
  H   -1.57918667365396     -1.02245710553356     -3.00817218975222 
  H   -1.25884477264019      0.72228568282228     -3.05436504943444 
  H   -2.76640836866444      0.09214024180044     -3.74175962042542 
  H   -2.48702512015610      2.19699107950093     -1.76109217570029 
  H   -4.09281437247312      1.62358792037224     -2.21722220410103 
  H    1.56748953447666     -2.08272559171093      2.39129530400812 
  H    1.24244410784823     -2.99862712397942      0.90648689200259 
  H    2.74841706997603     -3.28356963121360      1.79607623870542 
  H    2.46292569126273     -2.62337849835601     -1.02093792725634 
  H    4.07045302720973     -2.74311163538248     -0.30244048876426 
  H    4.65448346115229     -0.70680401028077      0.95372228032966 
  H    3.99501075900995     -1.44490851377397      2.39456915731191 
  H    1.59574566134859      3.12234477429960      0.61292534612400 
  H    1.26826627874473      2.29663922181641      2.14912084106002 
  H    2.78162298309563      3.19639495340303      1.94666639317082 
  H    2.47549786087627      0.42734489313183      2.78235173581885 
  H    4.08755032211489      1.09782125435258      2.52289983946985 
  H    4.66408898713428      1.16248602569396      0.12799994420443 
  H    4.01827034865908      2.78498276510688      0.04889794787615 
  H    1.57919077794174     -1.02246221135116     -3.00817477609090 
  H    1.25884680990156      0.72228058908288     -3.05436842804018 
  H    2.76641276635084      0.09213636615134     -3.74175868249209 
  H    2.48701845128170      2.19698799301717     -1.76108932637069 
  H    4.09280944357511      1.62358920085783     -2.21721985740965 
  H    4.65645441062758     -0.48734137231740     -1.07991321549768 
  H    3.99664791915730     -1.36168656486501     -2.44226768508526 
        """,
    basis={
        "default": "def2tzvp",
        "C": "6-31G*",
        "H": "6-31G*",
        "N": "6-31G*",
    },
    symmetry=0,
    spin=6,
    charge=+3,
    verbose=4,
    max_memory=200000,
)

mf = scf.rohf.ROHF(mol).x2c()
mf.max_cycle = 100
title = "Cr2O4"
imp_inds = mol.search_ao_label(["Cr *", "O *"])
mf.chkfile = title + "_rohf.chk"
mf.init_guess = "atom"
mf.level_shift = 0.2
if os.path.isfile(title + "_rohf.chk"):
    mf.init_guess = "chk"
    mf.level_shift = 0.0
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
    73,
    78,
    79,
    100,
    101,
    102,
    107,
    108,
    109,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
]
if len(huge_casorbind) != 19:
    exit()

mc = mcscf.CASSCF(solver_base, ncas=19, nelecas=24)
mc.mo_coeff = mc.sort_mo(huge_casorbind, solver.mo_coeff)

bond_dims = [250] * 4 + [500] * 4 + [2000]
noises = [1e-4] * 4 + [1e-5] * 4 + [0]
thrds = [1e-10] * 8
ncas = 19
n_elec = 24
spin = 6
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
