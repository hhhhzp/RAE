#!/bin/bash

# Multi-node Multi-GPU Latent Caching Script for RAE
# Usage: 
#   On each node, set NODE_RANK and run this script
#   Node 0: NODE_RANK=0 bash run_cache_latents.sh
#   Node 1: NODE_RANK=1 bash run_cache_latents.sh
#   Node 2: NODE_RANK=2 bash run_cache_latents.sh
#   Node 3: NODE_RANK=3 bash run_cache_latents.sh

# Multi-node configuration
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond1
export SETUPTOOLS_USE_DISTUTILS=stdlib
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-29500}
export NNODES=${NNODES:-4}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export NODE_RANK=${NODE_RANK:-0}

# Data and output paths
DATA_PATH=${DATA_PATH:-"/apdcephfs/share_300000800/datamultimodal/zhenpeng_data/imagenet-1k"}
OUTPUT_DIR=${OUTPUT_DIR:-"/apdcephfs_sh2/share_300000800/data/multimodal/zhenpeng_data/imagenet-1k_cache_latent"}
RAE_CONFIG=${RAE_CONFIG:-"src/stage1/config.json"}
RAE_CKPT=${RAE_CKPT:-"../DeCo/dual_internvit_2b/exp_sem_gen_gate_c256_new_stage2_448px/epoch=0-step=40000.ckpt"}

# Caching parameters
IMAGE_SIZE=${IMAGE_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-8}
SPLIT=${SPLIT:-"train"}

# Launch latent caching with torchrun
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NGPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/cache_latents.py \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --rae-config "$RAE_CONFIG" \
    --rae-ckpt "$RAE_CKPT" \
    --image-size $IMAGE_SIZE \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --split $SPLIT \
    --use-hf-dataset
