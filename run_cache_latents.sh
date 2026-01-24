#!/bin/bash
# Example script to run latent caching with distributed training
# This script should be run from the RAE root directory
# Modify the parameters according to your setup

# Number of GPUs
NUM_GPUS=8

# Data and output paths
DATA_PATH="/apdcephfs/share_300000800/datamultimodal/zhenpeng_data/imagenet-1k"
OUTPUT_DIR="/apdcephfs_sh2/share_300000800/data/multimodal/zhenpeng_data/imagenet-1k_cache_latent"
RAE_CKPT="../DeCo/dual_internvit_2b/exp_sem_gen_gate_c256_new_stage2_448px/epoch=0-step=40000.ckpt"

# Training parameters
IMAGE_SIZE=256
BATCH_SIZE=512
NUM_WORKERS=8

# Run with torchrun (recommended for multi-node)
torchrun --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=29500 \
    src/cache_latents.py \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --rae-ckpt "$RAE_CKPT" \
    --image-size $IMAGE_SIZE \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --use-hf-dataset

# For multi-node setup, use:
# Node 0:
# cd src && torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
#     --master_addr="MASTER_IP" --master_port=29500 \
#     cache_latents.py [args...]
#
# Node 1:
# cd src && torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
#     --master_addr="MASTER_IP" --master_port=29500 \
#     cache_latents.py [args...]
