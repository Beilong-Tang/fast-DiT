#!/bin/bash

ref_batch=/usr/local/usrapps/m1/btang5/software/guided-diffusion-fid-eval/precomputed_stats/VIRTUAL_imagenet256_labeled.npz

sample_batch=/usr/local/usrapps/m1/btang5/fast-DiT/samples/DiT-XL-2-pretrained-size-256-vae-ema-cfg-1.5-seed-0.npz

guided_evaluator $ref_batch $sample_batch