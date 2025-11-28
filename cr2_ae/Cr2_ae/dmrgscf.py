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


dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
dmrgscf.settings.MPIPREFIX = " "


def set_custom_tmpdir(workdir):
    # 动态获取Slurm分配的作业ID（当在Slurm环境下运行时）
    slurm_job_id = os.getenv("SLURM_JOB_ID", "localrun")

    # 构建唯一临时目录路径
    custom_tmp = os.path.abspath(os.path.join(workdir, f"tmp_{slurm_job_id}"))

    # 确保目录存在
    os.makedirs(custom_tmp, exist_ok=True)

    # 双重配置方案：修改PySCF参数 + 重定向环境变量
    lib.param.TMPDIR = custom_tmp
    os.environ["TMPDIR"] = custom_tmp  # 覆盖系统环境变量
    tempfile.tempdir = custom_tmp  # 影响Python标准库的临时文件

    print(f"[DEBUG] 临时目录已设置为：{custom_tmp}")


current_workdir = os.path.dirname(os.path.abspath(__file__))
set_custom_tmpdir(current_workdir)

start = time()

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
# imp_inds = mol.search_ao_label(["Cr *", "O *"])
mf.chkfile = title + "_rohf.chk"
mf.init_guess = "atom"
mf.level_shift = 0.2
if os.path.isfile(title + "_rohf.chk"):
    mf.init_guess = "chk"
    mf.level_shift = 0.0
mf.kernel()

print("nao = ", mol.nao_nr())
print("nelec = ", mol.nelec)


with open("Cr2O4_rohf.molden", "w") as f1:
    molden.header(mol, f1)
    molden.orbital_coeff(mol, f1, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)


# ao_labels = ["Cr 3d"]
# ncas, nelecas, mo = avas.avas(mf, ao_labels)
solver = mcscf.CASSCF(mf, 10, 6)
mo = solver.sort_mo([130, 131, 132, 133, 134, 135, 139, 140, 142, 143])
statelis = [10, 0, 10, 0, 10, 0, 10]
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
