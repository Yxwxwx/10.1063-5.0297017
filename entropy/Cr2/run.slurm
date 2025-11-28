#!/bin/bash
#SBATCH -J CoN6BO3_huge          
#SBATCH -p cpu             
#SBATCH -N 1               
#SBATCH --cpus-per-task=32 
#SBATCH --mem=500G         

WORKDIR="$(pwd)"
cd $WORKDIR || exit  
module purge
module load anaconda3-2024.10-1 pybind11/2.13.6 gcc-13.3.0
conda activate pyscf_311

# ========================= 关键变量导出 =========================
export PYTHONPATH="/home/Yxwxwx/code/dmet/liblan_preview:$PYTHONPATH"
export PYTHONPATH="/home/Yxwxwx/code/dmet/:$PYTHONPATH"
ulimit -s unlimited
export OMP_STACKSIZE=2G

# ========================= 执行任务 =========================
python -u dmet_siso.py > dmet_siso.out

# module purge
# module load cmake-3.31.0 gcc-13.3.0 oneapi-2025.0 boost/1.86.0

# #Kylin
# export KYLIN=$HOME/Applications/kylin
# export PATH=$KYLIN/bin:$PATH


# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export OMP_PLACES=cores
# export OMP_PROC_BIND=close
# ulimit -s unlimited
# export OMP_STACKSIZE=2G

# kylin dmrg.inp > dmrg.out