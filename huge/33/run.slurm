#!/bin/bash
#SBATCH -J CoN6BO3_huge          
#SBATCH -p cpu             
#SBATCH -N 1               
#SBATCH --cpus-per-task=16 
#SBATCH --mem=500G         
#SBATCH --output=CoN6BO3_huge.dat
#SBATCH --error=CoN6BO3_huge.err

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
