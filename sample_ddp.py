# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import time
import torch
import torch.distributed as dist
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from pathlib import Path
from blpytorch.utils.log import setup_logger
from blpytorch.utils.cuda_tracker import get_gpu_stats
import shutil
import glob


def create_npz_from_sample_folder(sample_dir, num=50_000, delete = True):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    files = glob.glob(f"{sample_dir}/*.npy")
    print(files)
    for f in tqdm(files, desc="Building .npz file from samples"):
        sample_np = np.load(f)
        samples.append(sample_np)
    samples = np.concatenate(samples)
    print(samples.shape)
    samples = samples[:num]
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    if delete:
        shutil.rmtree(sample_dir)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def calculate_len_exist_samples_and_start_index(sample_folder_dir, rank):
    exist_samples = [f for f in os.listdir(sample_folder_dir) if f.endswith("npy")]
    len_exist_samples = 0
    for f in exist_samples:
        bs = int((Path(f).stem.split('-')[-1]).split('x')[0])
        len_exist_samples += bs
    rank_files = [f for f in exist_samples if f.startswith(f"rank_{rank}")]
    rank_files_sorted = sorted(rank_files, key=lambda i: int(i.split('_')[3]))
    if rank_files_sorted:
        start_index = int(rank_files_sorted[-1].split('_')[3]) + 1
    else:
        start_index = 0
    return len_exist_samples, start_index

def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    logger = setup_logger("./logs/samples", rank) ## setup logging
    logger.info(f"tf32 enabled: {args.tf32}")
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    logger.info(f"Starting rank={rank}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        logger.info(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()
    len_exist_samples, start_index = calculate_len_exist_samples_and_start_index(sample_folder_dir, rank)
    seed = args.global_seed * dist.get_world_size() + rank + len_exist_samples
    torch.manual_seed(seed)

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil((args.num_fid_samples - len_exist_samples) / global_batch_size) * global_batch_size)
    if rank == 0:
        logger.info(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)

    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = len_exist_samples
    for index, _ in enumerate(pbar):
        # Sample inputs:
        time_start = time.time()
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        gpu_stats = get_gpu_stats()
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        np.save(f"{sample_folder_dir}/rank_{rank}_iter_{start_index:06d}_sample-{'x'.join([str(i) for i in samples.shape])}.npy", samples)
        start_index += 1
        total += global_batch_size
        logger.info(f"finished: {(index+1)/len(pbar)}, rate: {time.time() - time_start:.2f} sec/step, {gpu_stats}")

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples, delete=True)
        logger.info("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
