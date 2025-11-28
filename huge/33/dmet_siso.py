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
        Co       4.51732000       6.67569000       9.46591000 
        O        0.10673000       5.71754000       4.08359000 
        O       -0.20263000       3.87954000       5.33764000 
        N        2.20103000       4.81931000       9.64841000 
        N        3.27947000       5.18898000      10.40808000 
        N        2.96166000       6.34059000       8.12730000 
        N        4.78118000       8.13083000       7.91817000 
        N        3.93960000       7.95118000       6.84297000 
        C        2.13409000       6.56281000       5.89663000 
        H        2.15281000       7.00310000       5.05479000 
        C        2.03077000       3.66458000      11.49540000 
        H        1.73700000       3.06541000      12.17190000 
        C        5.50442000       9.19936000       7.60933000 
        H        6.18258000       9.56592000       8.16466000 
        C        1.43158000       3.90965000      10.29386000 
        H        0.62724000       3.51713000       9.97328000 
        C        3.16869000       4.48078000      11.52190000 
        H        3.78222000       4.52024000      12.24510000 
        C        0.33451000       5.05711000       5.05858000 
        C       -1.20830000       3.41017000       4.41364000 
        H       -1.53949000       2.53582000       4.70645000 
        H       -1.95114000       4.04880000       4.38587000 
        H       -0.81534000       3.32710000       3.51879000 
        C        1.21376000       4.88369000       7.39477000 
        H        0.61320000       4.17029000       7.57651000 
        C        5.13582000       9.71857000       6.35604000 
        H        5.50592000      10.47662000       5.91809000 
        C        2.97247000       6.92938000       6.92905000 
        C        2.09430000       5.35617000       8.35146000 
        C        1.25428000       5.50570000       6.15663000 
        C        4.13930000       8.90860000       5.89915000 
        H        3.67507000       8.99583000       5.07499000 
        O        8.92791000       5.71754000      14.84822000 
        O        9.23727000       3.87954000      13.59418000 
        N        6.83361000       4.81931000       9.28341000 
        N        5.75518000       5.18898000       8.52374000
        N        6.07299000       6.34059000      10.80451000 
        N        4.25346000       8.13083000      11.01365000 
        N        5.09504000       7.95118000      12.08885000 
        C        6.90055000       6.56281000      13.03519000 
        H        6.88184000       7.00310000      13.87702000 
        C        7.00388000       3.66458000       7.43642000 
        H        7.29765000       3.06542000       6.75992000 
        C        3.53022000       9.19936000      11.32249000 
        H        2.85206000       9.56592000      10.76716000 
        C        7.60306000       3.90965000       8.63796000 
        H        8.40741000       3.51713000       8.95854000 
        C        5.86595000       4.48078000       7.40991000 
        H        5.25243000       4.52024000       6.68672000 
        C        8.70014000       5.05710000      13.87324000 
        C       10.24294000       3.41017000      14.51818000 
        H       10.57414000       2.53582000      14.22537000 
        H       10.98578000       4.04880000      14.54595000 
        H        9.84999000       3.32710000      15.41302000 
        C        7.82088000       4.88369000      11.53705000 
        H        8.42144000       4.17029000      11.35530000 
        C        3.89883000       9.71857000      12.57578000 
        H        3.52872000      10.47662000      13.01373000 
        C        6.06217000       6.92938000      12.00277000 
        C        6.94034000       5.35617000      10.58036000 
        C        7.78036000       5.50570000      12.77519000 
        C        4.89535000       8.90860000      13.03266000 
        H        5.35957000       8.99583000      13.85683000 
            """,
    basis={"default": "def2tzvp", "C": "6-31G*", "H": "6-31G*", "O": "6-31G*"},
    symmetry=0,
    spin=3,
    charge=+2,
    verbose=4,
    max_memory=200000,
)

mf = scf.rohf.ROHF(mol).x2c()
mf.max_cycle = 100
title = "Co_bpp_COOMe_2"
imp_inds = mol.search_ao_label(["Co *"])
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
    36,
    37,
    38,
    39,
    40,
    41,
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
]
if len(huge_casorbind) != 22:
    exit()

mc = mcscf.CASSCF(solver_base, ncas=22, nelecas=31)
mc.mo_coeff = mc.sort_mo(huge_casorbind, solver.mo_coeff)

fcidump.from_integrals(
    "FCIDUMP",
    mc.get_h1eff()[0],
    mc.get_h2eff(),
    mc.ncas,
    mc.nelecas,
    nuc=mc.get_h1eff()[1],
)

uos = caoas.shape[0] - asorbs[-1]
dm_uo = caoas[:, uos:] @ caoas[:, uos:].conj().T * 2
mo_cas = caoas[:, act_inds] @ mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
print("mo_cas.shape = ", mo_cas.shape)
sodm1_act = reduce(
    np.dot, [caoas[:, act_inds], solver.make_rdm1(), caoas[:, act_inds].conj().T]
)
sodm1 = sodm1_act + dm_uo

# soc integral
# breit-pauli amfi
# amfi = True
# hso1e = mol.intor_asymmetric("int1e_pnucxp", 3)
# vj, vk = get_bp_hso2e_amfi(mol, sodm1) if amfi else get_bp_hso2e(mol, sodm1)
# hso2e = vj - vk * 1.5
# hsoao = 1.0j * (nist.ALPHA**2 / 2) * (hso1e + hso2e)
# print("hsoao.shape = ", hsoao.shape)

# hso = np.asarray([reduce(np.dot, (mo_cas.T, x.T, mo_cas)) for x in hsoao])
# print("hso.shape = ", hso.shape)
# # Save each component of hso (shape: 3 x nao x nao) to separate binary files as complex128
# hso[0].astype(np.complex128).tofile("hso_x.bin")
# hso[1].astype(np.complex128).tofile("hso_y.bin")
# hso[2].astype(np.complex128).tofile("hso_z.bin")

# x2camf
from mycamf import x2camf
import os

xmol, ctr = mf.with_x2c.get_xmol(mol)
xdmao = ctr @ sodm1 @ ctr.conj().T
print("sodm1.shape = ", sodm1.shape)
print("xdmao.shape = ", xdmao.shape)
x = mf.with_x2c.get_xmat()
r = mf.with_x2c._get_rmat(x=x)
hso1e = xmol.intor_asymmetric("int1e_pnucxp", 3)
hso1e = np.einsum(
    "pq,qr,irs,sk,kl->ipl", r.conj().T, x.conj().T, hso1e, x, r, optimize=True
)

hso2e = x2camf.amfi(mf.with_x2c, printLevel=1)

hso2e = hso2e / (nist.ALPHA**2 / 4)
sphsp = xmol.sph2spinor_coeff()
p_repr = np.einsum("ixp,pq,jyq->ijxy", sphsp, hso2e, sphsp.conj(), optimize=True)
hso2e = (p_repr[0, np.array([1, 1, 0])] * np.array([-1j, 1, -1j])[:, None, None]).real

s22 = xmol.intor_symmetric("int1e_ovlp")
s21 = gto.intor_cross("int1e_ovlp", xmol, mol)
c = cho_solve(s22, s21)
hso1e = c.conj().T @ hso1e @ c
hso2e = c.conj().T @ hso2e @ c
print("hso1e.shape = ", hso1e.shape)
print("hso2e.shape = ", hso2e.shape)
hsoao = (1j * nist.ALPHA**2 / 2) * (hso1e + hso2e)

hso = np.asarray([reduce(np.dot, (mo_cas.T.conj(), i.T, mo_cas)) for i in hsoao])
print("hso.shape = ", hso.shape)
# Save each component of hso (shape: 3 x nao x nao) to separate binary files as complex128
hso_labels = ["x", "y", "z"]
for i, label in enumerate(hso_labels):
    with open(f"hso_{label}.bin", "wb") as f:
        # 先写入shape信息（int32） ,再写入数据（complex128）
        shape = np.array(hso[i].shape, dtype=np.int32)
        f.write(shape.tobytes())
        f.write(hso[i].astype(np.complex128).tobytes())
        # print(hso[i])
