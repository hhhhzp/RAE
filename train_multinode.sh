#!/bin/bash

# Multi-node Multi-GPU Training Script for RAE
# Usage: 
#   On each node, set NODE_RANK and run this script
#   Node 0: NODE_RANK=0 bash train_multinode.sh
#   Node 1: NODE_RANK=1 bash train_multinode.sh
#   Node 2: NODE_RANK=2 bash train_multinode.sh
#   Node 3: NODE_RANK=3 bash train_multinode.sh

# Multi-node configuration
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond1
export SETUPTOOLS_USE_DISTUTILS=stdlib
export WANDB_RESUME=auto
export WANDB_RUN_ID=88467262
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export MASTER_ADDR=${MASTER_ADDR:-"29.111.44.202"}
export MASTER_PORT=${MASTER_PORT:-28778}
export NNODES=${NNODES:-4}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export NODE_RANK=${NODE_RANK:-0}

# Experiment configuration
export WANDB_KEY='704a2d2634e43e681d6187f3b7c875f26cce2eec'
export ENTITY="zhenpenghuang"
export PROJECT="RAE"
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"RAE_2B"}

# Training configuration
CONFIG_PATH=${CONFIG_PATH:-"configs/stage2/training/ImageNet256/DiTDH-XL_DINOv2-B.yaml"}
DATA_PATH=${DATA_PATH:-"/apdcephfs/share_300000800/datamultimodal/zhenpeng_data/imagenet-1k"}
RESULTS_DIR=${RESULTS_DIR:-"ckpts/stage2"}
PRECISION=${PRECISION:-"fp32"}

# Launch training with torchrun
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NGPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/train.py \
    --config $CONFIG_PATH \
    --data-path $DATA_PATH \
    --results-dir $RESULTS_DIR \
    --precision $PRECISION \
    --compile \
    --wandb
