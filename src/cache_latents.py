#!/usr/bin/env python3
"""
Cache VAE latents for faster training.
Supports multi-node distributed processing.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from stage1.configuration_uniflow import UniFlowVisionConfig
from stage1.modeling_uniflow import UniFlowVisionModel
from utils.train_utils import prepare_dataloader, center_crop_arr
from utils.dist_utils import setup_distributed, cleanup_distributed


def parse_args():
    parser = argparse.ArgumentParser(description="Cache VAE latents for training")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to dataset (HuggingFace dataset path or ImageFolder)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save cached latents",
    )
    parser.add_argument(
        "--rae-config",
        type=str,
        default="src/stage1/config.json",
        help="Path to RAE config file",
    )
    parser.add_argument(
        "--rae-ckpt",
        type=str,
        required=True,
        help="Path to RAE checkpoint",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size for center crop",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--use-hf-dataset",
        action="store_true",
        help="Use HuggingFace dataset instead of ImageFolder",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to cache (train/validation/test)",
    )
    return parser.parse_args()


def cache_latents(
    rae,
    data_loader,
    output_dir,
    rank,
    world_size,
    device,
):
    """Cache latents with distributed processing."""
    os.makedirs(output_dir, exist_ok=True)

    rae.eval()

    # Progress bar only on rank 0
    if rank == 0:
        pbar = tqdm(total=len(data_loader), desc="Caching latents")

    processed_count = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)

            # Encode images and horizontally flipped images in parallel
            images_flip = images.flip(dims=[3])
            images_concat = torch.cat([images, images_flip], dim=0)
            latents_concat = rae.encode(images_concat.to(torch.bfloat16)).float()

            # Split back to original and flipped latents
            batch_size = images.shape[0]
            latents = latents_concat[:batch_size]
            latents_flip = latents_concat[batch_size:]

            # Save each sample in the batch
            for i in range(latents.shape[0]):
                # Create unique filename based on global index
                global_idx = (
                    batch_idx * data_loader.batch_size * world_size
                    + rank * data_loader.batch_size
                    + i
                )
                save_path = os.path.join(output_dir, f"latent_{global_idx:08d}.npz")

                np.savez(
                    save_path,
                    latent=latents[i].cpu().numpy(),
                    latent_flip=latents_flip[i].cpu().numpy(),
                    label=labels[i].cpu().numpy(),
                )
                processed_count += 1

            if rank == 0:
                pbar.update(1)

            # Synchronize across GPUs
            if dist.is_initialized():
                torch.cuda.synchronize()

    if rank == 0:
        pbar.close()

    # Gather statistics across all ranks
    if dist.is_initialized():
        total_processed = torch.tensor(processed_count, device=device)
        dist.all_reduce(total_processed, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(f"Total samples cached: {total_processed.item()}")
    else:
        print(f"Total samples cached: {processed_count}")


def main():
    args = parse_args()

    # Setup distributed training
    rank, world_size, device = setup_distributed()

    if rank == 0:
        print(f"Caching latents with {world_size} GPUs")
        print(f"Data path: {args.data_path}")
        print(f"Output directory: {args.output_dir}")

    # Load RAE model
    config = UniFlowVisionConfig.from_pretrained(args.rae_config)
    config.num_sampling_steps = '1'
    rae = UniFlowVisionModel._from_config(config, dtype=torch.bfloat16).to(device)

    # Load checkpoint
    state_dict = torch.load(args.rae_ckpt, map_location='cpu', mmap=True)

    # Handle different checkpoint formats
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        rae_state_dict = {
            k.replace('model.', '', 1): v
            for k, v in state_dict.items()
            if k.startswith('model.')
            and not any(x in k for x in ["lpips_loss", "teacher_mlp"])
        }
    else:
        rae_state_dict = state_dict

    rae.load_state_dict(rae_state_dict)
    rae.eval()

    if rank == 0:
        print("RAE model loaded successfully")

    # Prepare data loader
    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda pil_image: center_crop_arr(pil_image, args.image_size)
            ),
            transforms.ToTensor(),
        ]
    )

    loader, sampler = prepare_dataloader(
        Path(args.data_path),
        args.batch_size,
        args.num_workers,
        rank,
        world_size,
        transform=transform,
        use_hf_dataset=args.use_hf_dataset,
        split=args.split,
    )

    if rank == 0:
        print(f"Dataset loaded: {len(loader.dataset)} samples")
        print(f"Batches per GPU: {len(loader)}")

    # Cache latents
    cache_latents(
        rae=rae,
        data_loader=loader,
        output_dir=args.output_dir,
        rank=rank,
        world_size=world_size,
        device=device,
    )

    if rank == 0:
        print("Caching completed!")

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
