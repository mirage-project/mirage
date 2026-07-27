#!/bin/bash
set -uo pipefail
R="$HOME/mpk-qwen35/m3i10-profile"
exec > "$R/logs/analysis.log" 2>&1
cd "$R"
for BS in 1 16 8; do
  echo "############ decode bs$BS ############"
  python3 scripts/analyze_trace.py traces/main/decode_bs${BS}_win*.json \
      --out-dir out/main --label bs${BS} --top 70
done
for BS in 1 16 8; do
  echo "############ prefill bs$BS ############"
  python3 scripts/analyze_trace.py traces/main/prefill_bs${BS}_win0.json \
      --out-dir out/main --label prefill_bs${BS} --top 25 || echo "prefill bs$BS analysis failed"
done
echo "=== ANALYSIS_DONE ==="
