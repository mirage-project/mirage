#!/bin/bash
set -uo pipefail
GPUID="$1"; shift
R="$HOME/mpk-qwen35/m3i10-profile"
mkdir -p "$R/logs" "$R/sglang/out" "$R/sglang/traces"
exec > "$R/logs/sglang_probe.log" 2>&1
set -x
export CUDA_VISIBLE_DEVICES="$GPUID"
export HF_HOME="$HOME/mpk-qwen35/hf"
export PATH="$HOME/.local/bin:/usr/local/cuda-12.8/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
cd "$R/sglang"
source venv-sglang/bin/activate
date -Is
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv
echo "NOTE: co-tenant idle contexts present on all GPUs; this probe is FEASIBILITY, its throughput is NOT binding."

timeout 1500 python3 "$R/scripts/sglang_probe.py" \
   --out "$R/sglang/out/probe_bs1.json" \
   --profile-dir "$R/sglang/traces" "$@"
EXIT=$?
echo "=== SGLANG_PROBE_EXIT=$EXIT at $(date) ==="
exit $EXIT
