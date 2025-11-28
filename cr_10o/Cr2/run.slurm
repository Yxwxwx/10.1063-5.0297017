#!/bin/bash
#SBATCH -J Cr2O4          
#SBATCH -p cpu             
#SBATCH -N 1               
#SBATCH --cpus-per-task=32 
#SBATCH --mem=500G         
#SBATCH --output=Cr2O4.dat
#SBATCH --error=Cr2O4.err

# ========================= 新增临时目录设置 =========================
WORKDIR="$(pwd)"
TMPDIR="$WORKDIR/tmp_${SLURM_JOB_ID}"  # 动态路径命名，防止多任务冲突

# 创建专属临时目录
mkdir -p "$TMPDIR" || { echo "无法创建临时目录"; exit 1; }

# ========================= 环境配置 =========================
cd $WORKDIR || exit  
module purge
module load anaconda3-2024.10-1 pybind11/2.13.6 gcc-13.3.0
conda activate my_pyscf
# export PYSCF_EXT_PATH=/home/Yxwxwx/code/dmrgscf:$PYSCF_EXT_PATH

# ========================= 关键变量导出 =========================
export PYTHONPATH="/home/Yxwxwx/code/dmet/liblan_preview:$PYTHONPATH"
export PYTHONPATH="/home/Yxwxwx/code/dmet/:$PYTHONPATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
ulimit -s unlimited
export OMP_STACKSIZE=2G


# 指定全局临时目录（双重保险）
export TMPDIR="$TMPDIR"  # 确保系统级临时目录指向此处

# ========================= 执行任务 =========================
python dmet_siso.py > dmet_siso.out

# ========================= 任务完成后清理（可选） =========================
# 如需保留临时文件可注释以下两行
echo "清理临时目录: $TMPDIR"
rm -rf "$TMPDIR"