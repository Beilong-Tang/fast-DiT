#!/bin/bash

# hugging face cache
export HF_HUB_CACHE=/share/m1/btang5/hf-cache        # just the model cache
export HF_HOME=/share/m1/btang5/hf                   # cache + tokens + everything (cache lands in $HF_HOME/hub)
export HF_HUB_OFFLINE=1                              # IMPORTANT
# python sample.py --image-size 256 --seed 1


# use ema
python sample.py --image-size 256 --seed 1 --vae ema