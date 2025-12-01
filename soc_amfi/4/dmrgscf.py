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
    max_memory=500000,
)

mf = scf.rohf.ROHF(mol).x2c()
mf.max_cycle = 100
title = "CoSMM10"
imp_inds = mol.search_ao_label(["Co *"])
mf.chkfile = title + "_rohf.chk"
mf.init_guess = "atom"
if os.path.isfile(title + "_rohf.chk"):
    mf.init_guess = "chk"
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
