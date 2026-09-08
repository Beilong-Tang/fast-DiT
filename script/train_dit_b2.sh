#!/bin/bash

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 30 > gpu_usage.log &



vae='ema'
save_dir="/share/m1/btang5/workspace/fast_dit/vae_${vae}_latents_hf"
epochs=80
accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 train.py --model DiT-B/2 \
    --feature-path "$save_dir" --resume \
    --epochs "$epochs"
