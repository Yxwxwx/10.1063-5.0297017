from pyscf import gto, scf, ao2mo, mcscf, dmrgscf, lib
from pyscf.mcscf import avas
from pyscf.tools import molden, fcidump
from pyscf.lib import logger, cho_solve
from pyscf.data import nist
import numpy as np
import scipy
from functools import reduce
import os
import tempfile
import copy
from time import time

start = time()

if not os.path.exists("/nvme/sxy/dmrgscf"):
    os.makedirs("/nvme/sxy/dmrgscf")
lib.param.TMPDIR = "/nvme/sxy/dmrgscf"

dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = " "
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
    basis={
        "default": "def2tzvp",
        "C": "6-31G*",
        "H": "6-31G*",
        "O": "6-31G*",
        "B": "6-31G*",
    },
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
    mf.max_cycle = 0
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
        new_mcf.threads = 16
        new_mcf.memory = int(mol.max_memory / 1000)
        new_mcf.block_extra_keyword = [
            "real_density_matrix",
            "davidson_soft_max_iter 1600",
            "noreorder",
            "cutoff 1E-24",
        ]
        new_mcf.runtimeDir = lib.param.TMPDIR + title + "/%d" % i
        new_mcf.scratchDirectory = lib.param.TMPDIR + title + "/%d" % i
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

np.save("sodm1.npy", sodm1)
# x2camf
from mycamf import x2camf

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

print(f"total time elapsed: {time() - start:.2f}s")
