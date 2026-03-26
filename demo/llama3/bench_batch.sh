#!/bin/bash
# Benchmark per-token latency across batch sizes 1, 2, 4, 8, 16
# Usage: bash demo/llama3/bench_batch.sh

set -e

MODEL="meta-llama/Llama-3.2-1B-Instruct"
COMMON_FLAGS="--use-mirage --no-use-cutlass-kernel --model $MODEL --ignore-eos --max-seq-length 2048"

echo "============================================"
echo "Batch latency benchmark - $MODEL"
echo "============================================"
echo ""

RESULTS_FILE="bench_batch_llama3_results.txt"
> "$RESULTS_FILE"

for BS in 1 2 4 8 16; do
    MAX_TOKENS=8

    echo ">>> Running BS=$BS  max_num_batched_tokens=$MAX_TOKENS"
    python demo/llama3/demo.py $COMMON_FLAGS \
        --max-num-batched-requests "$BS" \
        --max-num-batched-tokens "$MAX_TOKENS" 2>&1 | tee /tmp/bench_llama3_bs${BS}.log

    # Extract the per-token latency line
    LATENCY=$(grep "per-token latency" /tmp/bench_llama3_bs${BS}.log | tail -1)
    echo "BS=$BS  tokens=$MAX_TOKENS  $LATENCY" | tee -a "$RESULTS_FILE"
    echo ""
done

echo "============================================"
echo "Summary"
echo "============================================"
cat "$RESULTS_FILE"
