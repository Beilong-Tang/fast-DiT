#!/bin/bash
#SBATCH --job-name=fast-DiT-train-dit-xl2
#SBATCH --output=hpc_logs/%j_%x.out
#SBATCH --error=hpc_logs/%j_%x.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
set -e 

# Initialization
module load cuda
module load conda
source /usr/local/apps/conda/miniconda3/26.3.2/etc/profile.d/conda.sh
conda activate /usr/local/usrapps/m1/btang5/envs/DiT


# hugging face cache
export HF_HUB_CACHE=/share/m1/btang5/hf-cache        # just the model cache
export HF_HOME=/share/m1/btang5/hf                   # cache + tokens + everything (cache lands in $HF_HOME/hub)
export HF_HUB_OFFLINE=1                              # IMPORTANT

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 30 > gpu_usage.log &

vae='ema'
save_dir="/share/m1/btang5/workspace/fast_dit/vae_${vae}_latents_hf"
epochs=80
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --model DiT-XL/2 \
    --feature-path "$save_dir" --resume \
    --epochs "$epochs" 
