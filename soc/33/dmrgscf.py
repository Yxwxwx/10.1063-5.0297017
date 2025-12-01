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
    max_memory=400000,
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
