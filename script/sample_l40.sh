#!/bin/bash

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 30 > gpu_usage_sample_ddpm.log &
NVSMI_PID=$!
trap 'kill $NVSMI_PID 2>/dev/null' EXIT

# hugging face cache
export HF_HUB_CACHE=/share/m1/btang5/hf-cache        # just the model cache
export HF_HOME=/share/m1/btang5/hf                   # cache + tokens + everything (cache lands in $HF_HOME/hub)
export HF_HUB_OFFLINE=1                              # IMPORTANT
# python sample.py --image-size 256 --seed 1

# test run 
torchrun --nnodes=1 --nproc_per_node=1 sample_ddp.py --model DiT-XL/2 --num-fid-samples 50000 \
    --per-proc-batch-size 16 \
    --sample-dir samples/samples_l40