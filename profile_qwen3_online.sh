#!/bin/bash

# Nsys profiling script for Qwen3 demo
# Usage: ./profile_qwen3_online.sh

export CUDA_VISIBLE_DEVICES=0
export TMPDIR=~/vllm/mirage/nvidia

nsys profile \
    --trace=cuda,nvtx \
    --cuda-graph-trace=node \
    --output=mirage_online \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    python demo/qwen3/demo_online.py \
        --max-num-batched-requests 6 \
        --max-seq-length 1024 \
        --use-nsys
