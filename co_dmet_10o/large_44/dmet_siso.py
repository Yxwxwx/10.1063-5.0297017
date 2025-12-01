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

if __name__ == "__main__":
    mol = gto.M(
        atom="""
Co       0.08614000      12.11827000       3.79913000 
N       -1.03087000      12.08263000       1.98298000 
N       -1.95892000      12.14777000       4.39138000 
N       -2.70024000      12.08608000       5.50995000 
H       -2.37428000      12.16884000       6.32356000 
N        1.18073000      10.64850000       2.70623000 
N        0.50813000      10.51364000       5.18129000 
N        0.27159000      10.17225000       6.45848000 
H       -0.20185000      10.65923000       7.01687000 
N        1.38678000      13.37998000       2.68823000 
N        0.85251000      13.55623000       5.21213000 
N        0.85714000      13.78190000       6.54474000 
H        0.31753000      13.38534000       7.11490000 
O       -0.46275000      12.07689000       0.71844000 
O        1.52561000      10.77360000       1.37361000 
O        1.64253000      13.22940000       1.35543000 
B        1.03912000      12.02516000       0.72183000 
C       -3.19233000      11.76462000       0.83411000 
H       -3.05941000      12.50218000       0.20318000 
H       -4.12766000      11.74738000       1.12641000 
H       -2.97296000      10.91594000       0.39567000 
C       -2.30918000      11.95428000       2.01220000 
C       -2.82609000      11.96769000       3.37566000 
C       -4.12290000      11.80485000       3.86935000 
H       -4.91909000      11.66883000       3.36853000 
C       -4.00314000      11.88148000       5.23637000 
H       -4.70720000      11.80485000       5.86908000 
C        2.45210000       8.55574000       2.59323000
H        3.26754000       8.96763000       2.23856000 
H        1.93667000       8.16493000       1.85715000 
H        2.69438000       7.85266000       3.23129000 
C        1.63440000       9.58641000       3.27763000 
C        1.24980000       9.50787000       4.68742000 
C        1.48880000       8.53658000       5.65699000 
H        1.98618000       7.73197000       5.56252000 
C        0.84263000       8.99828000       6.78696000 
H        0.80778000       8.56915000       7.63354000 
C        3.17695000      15.04822000       2.63066000 
H        2.79814000      15.55397000       1.88388000 
H        3.86951000      14.44092000       2.29737000 
H        3.57097000      15.66700000       3.27941000 
C        2.10425000      14.26084000       3.28832000 
C        1.82096000      14.34897000       4.72485000 
C        2.43325000      15.06737000       5.75145000 
H        3.14855000      15.68808000       5.67481000 
C        1.78385000      14.68422000       6.88855000 
H        1.95883000      15.00032000       7.76722000 
C        1.53838000      11.95811000      -0.79668000 
C        1.26255000      10.84123000      -1.58802000 
H        0.75942000      10.12665000      -1.21730000 
C        1.70236000      10.74544000      -2.90157000 
H        1.48992000       9.97914000      -3.42200000 
C        2.45632000      11.77803000      -3.45051000 
H        2.78318000      11.70906000      -4.33987000 
C        2.72753000      12.89874000      -2.70195000 
H        3.22141000      13.61332000      -3.08693000 
C        2.28708000      12.99262000      -1.39019000 
H        2.49441000      13.77041000      -0.88402000
            """,
    basis={"default": "def2tzvp", "C": "6-31G*", "H": "6-31G*", "O": "6-31G*", "B": "6-31G*"},
    symmetry=0,
    spin=3,
    charge=+1,
    verbose=4,
    max_memory=200000,
    )
    
    mf = scf.rohf.ROHF(mol).x2c()
    mf.max_cycle = 100
    title = "CoN6BO3"
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
