from pyscf import gto, scf, ao2mo, mcscf, dmrgscf, lib
from pyscf.mcscf import avas
from pyscf.tools import molden, fcidump
from pyscf.lib import logger, cho_solve
from pyscf.data import nist
import numpy as np
from functools import reduce
import os
import copy
from time import time


def get_bp_hso2e(mol, dm0):
    # SOC_2e integrals are anti-symmetric towards exchange (ij|kl) -> (ji|kl)
    vj, vk, vk2 = scf.jk.get_jk(
        mol,
        [dm0, dm0, dm0],
        ["ijkl,kl->ij", "ijkl,jk->il", "ijkl,li->kj"],
        intor="int2e_p1vxp1",
        comp=3,
    )
    # hso2e = vj - 1.5 * vk - 1.5 * vk2
    return vj, vk + vk2
    # hso2e = mol.intor("int2e_p1vxp1", 3).reshape(3, mol.nao, mol.nao, mol.nao, mol.nao)
    # vj = np.einsum("yijkl,lk->yij", hso2e, dm0, optimize=True)
    # vk = np.einsum("yijkl,jk->yil", hso2e, dm0, optimize=True)
    # vk += np.einsum("yijkl,li->ykj", hso2e, dm0, optimize=True)
    # return vj, vk


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


dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = " "

start = time()

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
    max_memory=500000,
)

mf = scf.rohf.ROHF(mol).x2c()
mf.max_cycle = 100
title = "CoSPh4"
mf.chkfile = title + "_rohf.chk"
mf.init_guess = "atom"
mf.level_shift = 0.2
if os.path.isfile(title + "_rohf.chk"):
    mf.init_guess = "chk"
    mf.level_shift = 0.0
mf.kernel()

print("nao = ", mol.nao_nr())
print("nelec = ", mol.nelec)


with open(title + "_rohf.molden", "w") as f1:
    molden.header(mol, f1)
    molden.orbital_coeff(mol, f1, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)


ao_labels = ["Co 3d"]
ncas, nelecas, mo = avas.avas(mf, ao_labels)
assert ncas == 5
# assert nelecas == 7
solver = mcscf.CASSCF(mf, ncas=ncas, nelecas=nelecas)
# mo = solver.sort_mo([130, 131, 132, 133, 134, 135, 139, 140, 142, 143])
statelis = [0, 40, 0, 10]
mcfs = []
logger.info(solver, "Attempting SA-DMRGSCF with")
for i in range(len(statelis)):
    if statelis[i] != 0:
        new_mcf = dmrgscf.DMRGCI(mol, maxM=1500, tol=1e-10)
        new_mcf.spin = i
        new_mcf.nroots = statelis[i]
        new_mcf.threads = 32
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
        logger.info(solver, "%s states with spin multiplicity %s", statelis[i], i + 1)
statetot = sum(statelis)
mcscf.state_average_mix_(solver, mcfs, np.ones(statetot) / statetot)
solver.kernel(mo)

mc = copy.copy(solver)

fcidump.from_integrals(
    "FCIDUMP",
    solver.get_h1eff()[0],
    solver.get_h2eff(),
    solver.ncas,
    solver.nelecas,
    nuc=solver.get_h1eff()[1],
)

mo_cas = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
sodm1 = mc.make_rdm1()

# soc integral
# breit-pauli amfi
soc_start = time()
amfi = False
hso1e = mol.intor_asymmetric("int1e_pnucxp", 3)
vj, vk = get_bp_hso2e_amfi(mol, sodm1) if amfi else get_bp_hso2e(mol, sodm1)
hso2e = vj - vk * 1.5
hsoao = 1.0j * (nist.ALPHA**2 / 2) * (hso1e + hso2e)
print("hsoao.shape = ", hsoao.shape)

hso = np.asarray([reduce(np.dot, (mo_cas.T, x.T, mo_cas)) for x in hsoao])
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

soc_end = time()
print(f"soc time = {(soc_end - soc_start):.3f} s")

# np.save("sodm1.npy", sodm1)
# # x2camf
# from mycamf import x2camf

# xmol, ctr = mf.with_x2c.get_xmol(mol)
# xdmao = ctr @ sodm1 @ ctr.conj().T
# print("sodm1.shape = ", sodm1.shape)
# print("xdmao.shape = ", xdmao.shape)
# x = mf.with_x2c.get_xmat()
# r = mf.with_x2c._get_rmat(x=x)
# hso1e = xmol.intor_asymmetric("int1e_pnucxp", 3)
# hso1e = np.einsum(
#     "pq,qr,irs,sk,kl->ipl", r.conj().T, x.conj().T, hso1e, x, r, optimize=True
# )

# hso2e = x2camf.amfi(mf.with_x2c, printLevel=1)

# hso2e = hso2e / (nist.ALPHA**2 / 4)
# sphsp = xmol.sph2spinor_coeff()
# p_repr = np.einsum("ixp,pq,jyq->ijxy", sphsp, hso2e, sphsp.conj(), optimize=True)
# hso2e = (p_repr[0, np.array([1, 1, 0])] * np.array([-1j, 1, -1j])[:, None, None]).real

# s22 = xmol.intor_symmetric("int1e_ovlp")
# s21 = gto.intor_cross("int1e_ovlp", xmol, mol)
# c = cho_solve(s22, s21)
# hso1e = c.conj().T @ hso1e @ c
# hso2e = c.conj().T @ hso2e @ c
# print("hso1e.shape = ", hso1e.shape)
# print("hso2e.shape = ", hso2e.shape)
# hsoao = (1j * nist.ALPHA**2 / 2) * (hso1e + hso2e)

# hso = np.asarray([reduce(np.dot, (mo_cas.T.conj(), i.T, mo_cas)) for i in hsoao])
# print("hso.shape = ", hso.shape)
# # Save each component of hso (shape: 3 x nao x nao) to separate binary files as complex128
# hso_labels = ["x", "y", "z"]
# for i, label in enumerate(hso_labels):
#     with open(f"hso_{label}.bin", "wb") as f:
#         # 先写入shape信息（int32），再写入数据（complex128）
#         shape = np.array(hso[i].shape, dtype=np.int32)
#         f.write(shape.tobytes())
#         f.write(hso[i].astype(np.complex128).tobytes())
#         # print(hso[i])

print(f"total time elapsed: {time() - start:.2f}s")
