#!/bin/bash
set -e

cd "$(dirname "$0")/.."

OUTPUT_LEN=2048

echo "============================================"
echo "1. HuggingFace Transformers Baseline"
echo "============================================"
python demo/bench_transformers.py --output-len $OUTPUT_LEN

echo ""
echo "============================================"
echo "2. Mirage - Qwen3-0.6B (BS=1)"
echo "============================================"
python demo/qwen3/demo.py --use-mirage --model Qwen/Qwen3-0.6B --ignore-eos --max-seq-length $OUTPUT_LEN --max-num-batched-requests 1

echo ""
echo "============================================"
echo "3. Mirage - Qwen3-0.6B (BS=2)"
echo "============================================"
python demo/qwen3/demo.py --use-mirage --model Qwen/Qwen3-0.6B --ignore-eos --max-seq-length $OUTPUT_LEN --max-num-batched-requests 2

echo ""
echo "============================================"
echo "4. Mirage - Qwen3-0.6B (BS=4)"
echo "============================================"
python demo/qwen3/demo.py --use-mirage --model Qwen/Qwen3-0.6B --ignore-eos --max-seq-length $OUTPUT_LEN --max-num-batched-requests 4

echo ""
echo "============================================"
echo "5. Mirage - Qwen3-0.6B (BS=8)"
echo "============================================"
python demo/qwen3/demo.py --use-mirage --model Qwen/Qwen3-0.6B --ignore-eos --max-seq-length $OUTPUT_LEN --max-num-batched-requests 8

echo ""
echo "============================================"
echo "6. Mirage - Llama-3.2-1B (BS=1)"
echo "============================================"
python demo/llama3/demo.py --use-mirage --model meta-llama/Llama-3.2-1B-Instruct --ignore-eos --max-seq-length $OUTPUT_LEN --max-num-batched-requests 1

echo ""
echo "============================================"
echo "7. Mirage - Llama-3.2-1B (BS=2)"
echo "============================================"
python demo/llama3/demo.py --use-mirage --model meta-llama/Llama-3.2-1B-Instruct --ignore-eos --max-seq-length $OUTPUT_LEN --max-num-batched-requests 2

echo ""
echo "============================================"
echo "8. Mirage - Llama-3.2-1B (BS=4)"
echo "============================================"
python demo/llama3/demo.py --use-mirage --model meta-llama/Llama-3.2-1B-Instruct --ignore-eos --max-seq-length $OUTPUT_LEN --max-num-batched-requests 4

echo ""
echo "============================================"
echo "9. Mirage - Llama-3.2-1B (BS=8)"
echo "============================================"
python demo/llama3/demo.py --use-mirage --model meta-llama/Llama-3.2-1B-Instruct --ignore-eos --max-seq-length $OUTPUT_LEN --max-num-batched-requests 8

echo ""
echo "============================================"
echo "All tests complete!"
echo "============================================"
