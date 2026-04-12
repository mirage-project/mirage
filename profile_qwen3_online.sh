#!/bin/bash

# Nsys profiling script for Qwen3 demo
# Usage: ./profile_qwen3.sh

export CUDA_VISIBLE_DEVICES=0

nsys profile \
  -o qwen3_profile \
  --trace=cuda,nvtx,cublas,cudnn \
  --cuda-memory-usage=true \
  --stats=true \
  -f \
  python demo/qwen3/demo_online.py \
    --model Qwen/Qwen3-0.6B \
    --max-num-batched-requests 6 \
    --max-seq-length 1024