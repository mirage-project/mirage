#!/bin/bash

# Nsys profiling script for Qwen3 demo
# Usage: ./profile_qwen3.sh

export CUDA_VISIBLE_DEVICES=0

nsys profile \
    --trace=cuda,nvtx \
    --cuda-graph-trace=node \
    --output=mirage_online \
    python demo/qwen3/demo_online.py \
        --max-num-batched-requests 6 \
        --max-seq-length 1024 \
