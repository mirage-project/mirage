#!/bin/bash
# M3-I10 vLLM per-kernel decode profiling runner (catalyst-B200).
# usage: run_profile.sh <gpu_id> <tag> [extra python args...]
set -uo pipefail
GPUID="$1"; shift
TAG="$1"; shift

ROOT="$HOME/mpk-qwen35/m3i10-profile"
LOG="$ROOT/logs/${TAG}.log"
mkdir -p "$ROOT/logs" "$ROOT/traces/$TAG" "$ROOT/out/$TAG"

exec > "$LOG" 2>&1
set -x

export CUDA_VISIBLE_DEVICES="$GPUID"
export HF_HOME="$HOME/mpk-qwen35/hf"
export PATH="$HOME/.local/bin:/usr/local/cuda-12.8/bin:$PATH"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TOKENIZERS_PARALLELISM=false
cd "$HOME/mpk-qwen35"
source venv-vllm/bin/activate

echo "=== m3i10 profile tag=$TAG start: $(date) ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

python3 "$ROOT/scripts/profile_vllm_decode.py" \
  --trace-dir "$ROOT/traces/$TAG" \
  --out-dir "$ROOT/out/$TAG" \
  --log-file "$LOG" \
  "$@"
EXIT=$?
echo "=== M3I10_PROFILE_EXIT_CODE=$EXIT at $(date) ==="
exit $EXIT
