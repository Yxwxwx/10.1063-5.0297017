#!/bin/bash
#SBATCH -J CoNSPh2_KYLIN          
#SBATCH -p cpu             
#SBATCH -N 1               
#SBATCH --cpus-per-task=16 
#SBATCH --mem=500G         
#SBATCH --output=CoNSPh2_KYLIN.dat
#SBATCH --error=CoNSPh2_KYLIN.err


module purge
module load cmake-3.31.0 gcc-13.3.0 oneapi-2025.0 boost/1.86.0

#Kylin
export KYLIN=$HOME/Applications/kylin
export PATH=$KYLIN/bin:$PATH


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export OMP_PROC_BIND=close
ulimit -s unlimited
export OMP_STACKSIZE=2G

kylin dmrg.inp > dmrg.out