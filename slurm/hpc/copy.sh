#!/bin/bash
#======================================================
# Job name and output files
#======================================================
#SBATCH --job-name=copy-dit-xl2           # Job name
#SBATCH --output=hpc_logs/%j_%x.out
#SBATCH --error=hpc_logs/%j_%x.err
#======================================================
# Resource requests
#======================================================
#SBATCH --ntasks=1                 # Number of tasks (MPI ranks)
#SBATCH --cpus-per-task=1          # CPUs per task (OpenMP threads)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --time=72:00:00           # Time limit (HH:MM:SS)
#SBATCH --mem=4G                   # Memory per node
#======================================================
# Partition and QOS (optional - uses defaults if omitted)
#======================================================
#SBATCH --partition=compute       # Partition name
#SBATCH --qos=normal               # Quality of Service

set -e 

source /usr/local/apps/conda/miniconda3/26.3.2/etc/profile.d/conda.sh
conda activate /usr/local/usrapps/m1/btang5/envs/DiT

python -u slurm/hpc/copy.py results/000-DiT-XL-2/checkpoints  /gpfs_common/share01/m1/btang5/workspace/fast_dit/vae_ema_latents_hf/results/000-DiT-XL-2/checkpoints