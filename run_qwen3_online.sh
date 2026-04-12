#!/bin/bash

# run qwen3 online serving test script
# Usage: ./run_qwen3_online.sh

export CUDA_VISIBLE_DEVICES=4

python demo/qwen3/demo_online.py \
    --model Qwen/Qwen3-0.6B \
    --max-num-batched-requests 6 \
    --max-seq-length 1024 \
