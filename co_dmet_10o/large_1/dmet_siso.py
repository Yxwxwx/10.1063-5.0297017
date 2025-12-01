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

dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = " "

def set_custom_tmpdir(workdir):
    # 动态获取Slurm分配的作业ID（当在Slurm环境下运行时）
    slurm_job_id = os.getenv('SLURM_JOB_ID', 'localrun')
    
    # 构建唯一临时目录路径
    custom_tmp = os.path.abspath( 
        os.path.join(workdir, f"tmp_{slurm_job_id}") 
    )
    
    # 确保目录存在
    os.makedirs(custom_tmp, exist_ok=True)
    
    # 双重配置方案：修改PySCF参数 + 重定向环境变量
    lib.param.TMPDIR = custom_tmp
    os.environ['TMPDIR'] = custom_tmp  # 覆盖系统环境变量
    tempfile.tempdir = custom_tmp      # 影响Python标准库的临时文件
    
    print(f"[DEBUG] 临时目录已设置为：{custom_tmp}")

current_workdir = os.path.dirname(os.path.abspath(__file__))
set_custom_tmpdir(current_workdir)
def read_cas_info(fname):
    with open(fname, "r") as f1:
        ncasorb, ncaselec = [int(x) for x in f1.readline().split(" ")]
        casorbind = [int(x) for x in f1.readline().split(" ")]
        statelis = [int(x) for x in f1.readline().split(" ")]
    return ncasorb, ncaselec, casorbind, statelis


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
            Co   1.506499   10.764206    0.414024
            S    0.900702   12.969251    0.863740
            S   -0.630821    9.850987    0.263627
            S    2.753892   10.415545   -1.502746
            S    3.027943    9.878195    1.923416
            C    2.377311   13.975173    0.913216
            C    2.276518   15.365718    0.836418
            H    1.444112   15.762234    0.713835
            C    3.399137   16.154649    0.937832
            H    3.312247   17.078944    0.888601
            C    4.645140   15.606362    1.112844
            H    5.397607   16.149181    1.179058
            C    4.758096   14.244532    1.186689
            H    5.597454   13.867157    1.321825
            C    3.649380   13.407744    1.065829
            H    3.753648   12.483449    1.085522
            C   -0.483108    8.032887    0.187074
            C   -1.603989    7.293178    0.071383
            H   -2.422493    7.725245   -0.019692
            C   -1.574447    5.919042    0.086153
            H   -2.366884    5.440487    0.000000
            C   -0.364938    5.235392    0.231381
            H   -0.335395    4.306995    0.285534
            C    0.776797    5.988774    0.292919
            H    1.598776    5.558075    0.347072
            C    0.738565    7.384787    0.273226
            H    1.527526    7.875648    0.319995
            C    2.008897   11.246043   -2.857063
            C    0.792437   11.947467   -2.723896
            H    0.359725   11.969345   -1.900278
            C    0.238079   12.605139   -3.809910
            H   -0.559572   13.071388   -3.704557
            C    0.851522   12.577793   -5.050260
            H    0.465730   13.018063   -5.772218
            C    2.036702   11.896878   -5.208288
            H    2.450298   11.872266   -6.040521
            C    2.615389   11.247410   -4.127935
            H    3.423466   10.801671   -4.246087
            C    2.286945    9.211500    3.375947
            C    3.044626    9.114422    4.556237
            H    3.906575    9.464451    4.578390
            C    2.530237    8.503239    5.693449
            H    3.049839    8.432139    6.461437
            C    1.239051    8.001440    5.679173
            H    0.886278    7.585780    6.431899
            C    0.479633    8.125864    4.519314
            H   -0.391005    7.797712    4.514391
            C    0.962741    8.709702    3.404255
            H    0.424023    8.782168    2.648574
        """,
    basis={"default": "def2tzvp", "C": "6-31G*", "H": "6-31G*"},
    symmetry=0,
    spin=3,
    charge=-2,
    verbose=4,
    max_memory=200000,
)

mf = scf.rohf.ROHF(mol).x2c()
mf.max_cycle = 100
title = "CoSPh4"
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

# AS general info
caoas, ent, asorbs = get_dmet_as_props(mf, imp_inds, thres=1e-13)
# AS hcore + j/k
as1e = get_as_1e_ints(mf, caoas, asorbs)
# AS ERIs
as2e = get_as_2e_ints(mf, caoas, asorbs)

print(f"Entanglement S: {ent:.3f}")

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

casinfo_fname = title + "_cas_info"


if not os.path.isfile(casinfo_fname):
    print("Failed to read saved CAS briefings.")
    exit()


ncasorb, ncaselec, casorbind, statelis = read_cas_info(casinfo_fname)

solver = sacasscf_solve_imp(
    solver_base, mf.mol, ncasorb, ncaselec, casorbind, statelis=statelis
)

mc = solver.copy()
fcidump.from_integrals(
    "FCIDUMP",
    solver.get_h1eff()[0],
    solver.get_h2eff(),
    solver.ncas,
    solver.nelecas,
    nuc=solver.get_h1eff()[1],
)

# dm = solver.make_rdm1()
# print("dm.shape = ", dm.shape)

# casinfo_fname_large = title + "_cas_info_large"

# if not os.path.isfile(casinfo_fname_large):
#     print("Failed to read saved Large CAS briefings.")
#     exit()
# ncasorb_large, ncaselec_large, casorbind_large, statelis_large = read_cas_info(
#     casinfo_fname
# )
# solver.ncas = ncasorb_large
# solver.nelecas = ncaselec_large
# solver.mo_coeff = solver.sort_mo(casorbind_large, solver.mo_coeff)
# fcidump.from_integrals(
#     "FCIDUMP_1",
#     solver.get_h1eff()[0],
#     solver.get_h2eff(),
#     solver.ncas,
#     solver.nelecas,
#     nuc=mc.get_h1eff()[1],
# )

# with open("cas.molden", "w") as f1:
#     molden.header(mol, f1)
#     molden.orbital_coeff(
#         mol,
#         f1,
#         caoas[:, act_inds] @ solver.mo_coeff,
#         ene=solver_base.mo_energy,
#         occ=solver_base.mo_occ,
#     )


uos = caoas.shape[0] - asorbs[-1]
dm_uo = caoas[:, uos:] @ caoas[:, uos:].conj().T * 2
mo_cas = caoas[:, act_inds] @ mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
print("mo_cas.shape = ", mo_cas.shape)
sodm1_act = reduce(
    np.dot, [caoas[:, act_inds], mc.make_rdm1(), caoas[:, act_inds].conj().T]
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
        # 先写入shape信息（int32），再写入数据（complex128）
        shape = np.array(hso[i].shape, dtype=np.int32)
        f.write(shape.tobytes())
        f.write(hso[i].astype(np.complex128).tobytes())
        # print(hso[i])
