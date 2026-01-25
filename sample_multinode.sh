#!/bin/bash

# Multi-node Multi-GPU Sampling Script for RAE
# Usage: 
#   On each node, set NODE_RANK and run this script
#   Node 0: NODE_RANK=0 bash sample_multinode.sh
#   Node 1: NODE_RANK=1 bash sample_multinode.sh
#   Node 2: NODE_RANK=2 bash sample_multinode.sh
#   Node 3: NODE_RANK=3 bash sample_multinode.sh

# Multi-node configuration
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond1
export SETUPTOOLS_USE_DISTUTILS=stdlib
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export MASTER_ADDR=${MASTER_ADDR:-"29.111.44.202"}
export MASTER_PORT=${MASTER_PORT:-28779}
export NNODES=${NNODES:-4}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export NODE_RANK=${NODE_RANK:-0}

# Sampling configuration
CONFIG_PATH=${CONFIG_PATH:-"configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml"}
SAMPLE_DIR=${SAMPLE_DIR:-"samples"}
PRECISION=${PRECISION:-"fp32"}
LABEL_SAMPLING=${LABEL_SAMPLING:-"equal"}

# Launch sampling with torchrun
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NGPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/sample_ddp.py \
    --config $CONFIG_PATH \
    --sample-dir $SAMPLE_DIR \
    --precision $PRECISION \
    --label-sampling $LABEL_SAMPLING
