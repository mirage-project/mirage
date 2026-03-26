#!/bin/bash
# Simulate BS=16 tokens=16 latency from BS=8 and BS=16 tokens=8 runs
# The scheduling overhead per round ≈ (bs16_t8 - 2 * bs8_t8) per token
# Simulated bs16_t16 ≈ bs16_t8 - scheduling_overhead_per_round
set -e

QWEN_FLAGS="--use-mirage --no-use-cutlass-kernel --model Qwen/Qwen3-0.6B --ignore-eos --max-seq-length 2048"
LLAMA_FLAGS="--use-mirage --no-use-cutlass-kernel --model meta-llama/Llama-3.2-1B-Instruct --ignore-eos --max-seq-length 2048"

extract_latency() {
    grep "per-token latency" "$1" | tail -1 | grep -oP '[\d.]+(?= ms)'
}

echo "============================================"
echo "BS=16 Simulation Benchmark"
echo "============================================"

RESULTS_FILE="bench_bs16_sim_results.txt"
> "$RESULTS_FILE"

for MODEL_NAME in qwen llama; do
    if [ "$MODEL_NAME" = "qwen" ]; then
        FLAGS="$QWEN_FLAGS"
        DEMO="demo/qwen3/demo.py"
    else
        FLAGS="$LLAMA_FLAGS"
        DEMO="demo/llama3/demo.py"
    fi

    echo ""
    echo ">>> $MODEL_NAME: Running BS=8 tokens=8"
    python $DEMO $FLAGS --max-num-batched-requests 8 --max-num-batched-tokens 8 2>&1 | tee /tmp/sim_${MODEL_NAME}_bs8.log

    echo ""
    echo ">>> $MODEL_NAME: Running BS=16 tokens=8"
    python $DEMO $FLAGS --max-num-batched-requests 16 --max-num-batched-tokens 8 2>&1 | tee /tmp/sim_${MODEL_NAME}_bs16.log

    LAT_8=$(extract_latency /tmp/sim_${MODEL_NAME}_bs8.log)
    LAT_16=$(extract_latency /tmp/sim_${MODEL_NAME}_bs16.log)

    if [ -n "$LAT_8" ] && [ -n "$LAT_16" ]; then
        # bs16_t8 does 2 rounds, bs8_t8 does 1 round
        # scheduling overhead ≈ bs16_t8 - 2 * bs8_t8
        # simulated bs16_t16 = bs16_t8 - overhead = 2 * bs8_t8
        SIM=$(python3 -c "
bs8 = $LAT_8
bs16_t8 = $LAT_16
overhead = bs16_t8 - 2 * bs8
sim = bs16_t8 - overhead  # = 2 * bs8
print(f'BS=8  tokens=8:  {bs8:.3f} ms/token')
print(f'BS=16 tokens=8:  {bs16_t8:.3f} ms/token (actual, 2 rounds)')
print(f'Scheduling overhead per round: {overhead:.3f} ms/token')
print(f'BS=16 tokens=16: {sim:.3f} ms/token (simulated, 1 round)')
")
        echo ""
        echo "=== $MODEL_NAME simulation ==="
        echo "$SIM"
        echo ""
        echo "[$MODEL_NAME]" >> "$RESULTS_FILE"
        echo "$SIM" >> "$RESULTS_FILE"
        echo "" >> "$RESULTS_FILE"
    else
        echo "Failed to extract latency for $MODEL_NAME"
    fi
done

echo "============================================"
echo "Summary"
echo "============================================"
cat "$RESULTS_FILE"
