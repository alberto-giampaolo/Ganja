#!/bin/bash -l
#SBATCH --job-name=job_name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pasquale.musella@cern.ch
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH -A d78

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CRAY_CUDA_MPS=1

source $HOME/env.sh

cd $HOME/jupyter/GanjaPy

srun $@

## cms_zee_conditional_wgan.ipynb --Parameters.plot_every=100 --Parameters.epochs=1000 --Parameters.monitor_dir=logs_reweight_10000
