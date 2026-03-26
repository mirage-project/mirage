#!/bin/bash
# Benchmark per-token latency across batch sizes 1, 2, 4, 8, 16
# Usage: bash demo/qwen3/bench_batch.sh

set -e

PROMPT="Write a detailed 20000 word essay on the history of artificial intelligence, covering its origins, key milestones, major researchers, and future prospects."
MODEL="Qwen/Qwen3-0.6B"
COMMON_FLAGS="--use-mirage --no-use-cutlass-kernel --model $MODEL --ignore-eos --max-seq-length 2048"

echo "============================================"
echo "Batch latency benchmark - $MODEL"
echo "============================================"
echo ""

RESULTS_FILE="bench_batch_results.txt"
> "$RESULTS_FILE"

for BS in 1 2 4 8 16; do
    MAX_TOKENS=8

    echo ">>> Running BS=$BS  max_num_batched_tokens=$MAX_TOKENS"
    python demo/qwen3/demo.py $COMMON_FLAGS \
        --max-num-batched-requests "$BS" \
        --max-num-batched-tokens "$MAX_TOKENS" \
        --prompt "$PROMPT" 2>&1 | tee /tmp/bench_bs${BS}.log

    # Extract the per-token latency line
    LATENCY=$(grep "per-token latency" /tmp/bench_bs${BS}.log | tail -1)
    echo "BS=$BS  tokens=$MAX_TOKENS  $LATENCY" | tee -a "$RESULTS_FILE"
    echo ""
done

echo "============================================"
echo "Summary"
echo "============================================"
cat "$RESULTS_FILE"
