#!/bin/bash
#SBATCH --job-name=sample_eval_dit_xl2
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

# hugging face cache
export HF_HUB_CACHE=/share/m1/btang5/hf-cache        # just the model cache
export HF_HOME=/share/m1/btang5/hf                   # cache + tokens + everything (cache lands in $HF_HOME/hub)
export HF_HUB_OFFLINE=1                              # IMPORTANT

mkdir -p gpu_usage
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 30 > "gpu_usage/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" &
NVSMI_PID=$!
trap 'kill $NVSMI_PID 2>/dev/null' EXIT



##################
# % Parameters % #
##################
NUM_GPU=4 # same as the gre
SAMPLE_DIR=samples/train_dit_xl2
CKPT_PATH="/usr/local/usrapps/m1/btang5/fast-DiT/results/000-DiT-XL-2/checkpoints/0400000.pt" # the ckpt to infer

CFG_SCALE=1.0 # as paper
VAE=mse # as paper
NUM_FID_SAMPLES=50000 # default: 50000
SAMPLE_PER_PROC_BATCH_SIZE=32 # default: 32

# script
conda activate /usr/local/usrapps/m1/btang5/envs/DiT
torchrun --nnodes=1 --nproc_per_node=$NUM_GPU sample_ddp.py --model DiT-XL/2 --num-fid-samples $NUM_FID_SAMPLES \
    --ckpt pretrained_models/DiT-XL-2-256x256.pt --cfg-scale $CFG_SCALE --vae $VAE \
    --sample-dir $SAMPLE_DIR --per-proc-batch-size $SAMPLE_PER_PROC_BATCH_SIZE\
    --ckpt $CKPT_PATH

ckpt_stem=$(basename $CKPT_PATH .pt)
echo "ckpt_stem $ckpt_stem"
conda activate /usr/local/usrapps/m1/btang5/envs/sbgm
ref_batch=/usr/local/usrapps/m1/btang5/software/guided-diffusion-fid-eval/precomputed_stats/VIRTUAL_imagenet256_labeled.npz
sample_batch=${SAMPLE_DIR}/DiT-XL-2-${ckpt_stem}-size-256-vae-${VAE}-cfg-${CFG_SCALE}-seed-0.npz
guided_evaluator $ref_batch $sample_batch