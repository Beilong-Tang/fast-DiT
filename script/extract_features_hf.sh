#!/bin/bash
# This extracts feature using hf data

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 30 > gpu_usage.log &

data_path="/share/m1/btang5/data/2026-09-04/imagenet-256-hf/data/train-*.parquet"
vae='ema'
save_dir="/share/m1/btang5/workspace/fast_dit/vae_${vae}_latents_hf"

torchrun --nnodes=1 --nproc_per_node=3 \
    extract_features.py \
    --model DiT-XL/2 \
    --data-path "$data_path" \
    --features-path "$save_dir" \
    --vae "$vae" \
    --global-batch-size=192
