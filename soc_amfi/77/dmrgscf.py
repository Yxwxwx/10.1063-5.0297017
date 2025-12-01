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
Co      11.04404000       1.04900000       3.34378000 
S        8.77823000       3.00936000       4.33821000 
S       11.43622000       3.02151000       0.92730000 
S       10.91108000      -2.02878000       2.57231000 
S       13.08635000       0.19519000       5.60771000 
O       11.83167000      -0.45768000       5.96583000 
O       12.19019000       3.07283000      -0.30709000 
O        7.93263000       3.08738000       5.51462000 
O       10.07860000       2.53858000       0.82177000 
N       10.12989000      -0.70978000       2.99572000 
O       13.74529000       0.85669000       6.71938000 
O       10.27427000      -2.76510000       1.49667000 
O       10.00204000       3.77701000       4.38210000 
O       12.27951000      -1.59956000       2.30125000 
N       12.76369000       1.14747000       4.38064000 
N        9.20359000       1.51260000       3.96893000 
N       12.13222000       2.14916000       2.06988000 
C        6.10275000      -0.60027000       4.18585000 
H        5.20999000      -0.54352000       4.44081000 
C       13.46188000       2.40605000       2.47683000 
C        8.76888000      -0.75790000       3.38109000 
C        7.81280000       3.62067000       2.98712000 
H        6.98919000       3.13006000       2.93372000 
H        8.30293000       3.51372000       2.16919000 
H        7.62236000       4.55121000       3.12576000 
C       14.42792000       3.12175000       1.75566000 
H       14.21536000       3.46576000       0.91816000 
C       13.81006000       1.85310000       3.73518000 
C       11.00381000      -3.08882000       3.98356000 
H       11.42504000      -2.62011000       4.70785000 
H       11.51861000      -3.86861000       3.76408000 
H       10.11859000      -3.35226000       4.24328000 
C       11.35066000       4.67750000       1.54807000 
H       10.85506000       4.68550000       2.37038000 
H       10.91096000       5.23861000       0.90353000 
H       12.23807000       5.00681000       1.70646000 
C       15.09147000       2.06251000       4.23761000 
H       15.32372000       1.70571000       5.06450000 
C       15.70143000       3.32269000       2.28021000 
H       16.32999000       3.80946000       1.79791000 
C        6.91566000       0.52770000       4.27382000 
H        6.56140000       1.32684000       4.59079000 
C       14.17866000      -1.05460000       5.00085000 
H       13.76692000      -1.50748000       4.26157000 
H       14.36363000      -1.68492000       5.70099000 
H       14.99923000      -0.64903000       4.70968000 
C        8.26027000       0.46871000       3.88809000 
C        7.93481000      -1.87020000       3.30665000 
H        8.26769000      -2.67286000       2.97395000 
C        6.60455000      -1.79779000       3.72549000 
H        6.06192000      -2.55296000       3.69458000 
C       16.03321000       2.80059000       3.51717000 
H       16.88295000       2.94142000       3.86834000
        """,
    basis={
        "default": "def2tzvp",
        "C": "6-31G*",
        "H": "6-31G*",
        "S": "6-31G*",
        "O": "6-31G*",
    },
    symmetry=0,
    spin=3,
    charge=-2,
    verbose=4,
    max_memory=500000,
)

mf = scf.rohf.ROHF(mol).x2c()
mf.max_cycle = 100
title = "CoNSPh2"
imp_inds = mol.search_ao_label(["Co *"])
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
solver.max_cycle_micro = 100
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
amfi = True
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
