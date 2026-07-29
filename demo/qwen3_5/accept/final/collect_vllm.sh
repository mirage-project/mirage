#!/usr/bin/env bash
# FRESH vLLM COMPARATOR COLLECT (box-side) -- the primary AC-4/AC-5 comparator.
#
# Runs bench_vllm.py --mode sweep at the PINNED contract in the same window as
# the MPK runs.  bench_vllm.py itself refuses any deviation from that contract
# (enforce_binding_contract: workload 256/1024, batch sizes inside {1,2,4,8,16},
# >=3 reps, >=1 warmup rep, 5% dispersion bound) unless --exploratory is passed,
# which this script NEVER passes.
#
# The `exec > "$LOG" 2>&1` on the FIRST line of real work is load-bearing:
# bench_vllm.py re-reads that exact file to check the fp8-path markers that are
# logged inside the spawned EngineCore worker (bench-protocol.md 7), so the whole
# process tree's output has to land there.
#
# --language-model-only is NOT a free choice: it is passed the value the PINNED
# baseline table recorded, so the fresh run is the same comparator.  Any
# mismatch is caught again in score_perf.py's identity cross-check.
#
# EXIT: 0 collected, 3 could not collect.  It never decides AC-4/AC-5.
set -uo pipefail

BOX_ROOT="${MPK_BOX_ROOT:-$HOME/mpk-qwen35}"
VPY="${MPK_VLLM_PY:-$BOX_ROOT/venv-vllm/bin/python}"
# The interpreter's OWN bin dir goes on PATH -- the "activate" effect that
# matters here.  vLLM's inductor path shells out to `ninja`, which is a pip
# console script inside the venv, so calling venv-vllm/bin/python directly
# without this makes engine construction die with
# `FileNotFoundError: 'ninja'` inside determine_available_memory().  Found by
# running this collector for real (M4-I1 proof run).
export PATH="$(dirname "$VPY"):${CUDA_BIN:-/usr/local/cuda-12.8/bin}:$PATH"
export HF_HOME="${HF_HOME:-$BOX_ROOT/hf}"
export PYTHONUNBUFFERED=1

OUT=""; ACC=""; REPS=3; BSS="1,2,4,8,16"; LMO=""; RUNTAG="final-gate-fresh"
PINNED=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    --accept-dir) ACC="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    --batch-sizes) BSS="$2"; shift 2;;
    --pinned-baseline) PINNED="$2"; shift 2;;
    --run-tag) RUNTAG="$2"; shift 2;;
    --python) VPY="$2"; shift 2;;
    *) echo "collect_vllm.sh: unknown flag $1" >&2; exit 3;;
  esac
done
[ -n "$OUT" ] && [ -n "$ACC" ] && [ -n "$PINNED" ] || {
  echo "collect_vllm.sh: --out, --accept-dir and --pinned-baseline are required" >&2
  exit 3; }
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || {
  echo "collect_vllm.sh: CUDA_VISIBLE_DEVICES must be pinned by the guard" >&2; exit 3; }
[ -x "$VPY" ] || { echo "collect_vllm.sh: no vLLM interpreter at $VPY" >&2; exit 3; }
mkdir -p "$OUT"
LOG="$OUT/sweep.log"

# the pinned baseline's own language_model_only value -- not a knob we choose
LMO="$("$VPY" - "$PINNED" <<'EOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "full" / "summary.json"
try:
    print(((json.loads(p.read_text()).get("shared_meta") or {})
           .get("cli_args") or {}).get("language_model_only") or "")
except Exception:
    print("")
EOF
)"
[ -n "$LMO" ] || { echo "collect_vllm.sh: could not read language_model_only from $PINNED" >&2; exit 3; }

echo "=== fresh vLLM sweep GATE gpu=$CUDA_VISIBLE_DEVICES lmo=$LMO reps=$REPS bs=$BSS $(date -Is) ==="
echo "    log -> $LOG (bench_vllm.py re-reads it for the fp8-path markers)"
(
  exec > "$LOG" 2>&1
  echo "=== collect_vllm.sh wrapper: gpu=$CUDA_VISIBLE_DEVICES $(date -Is) ==="
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
  exec "$VPY" -u "$ACC/bench_vllm.py" --mode sweep \
      --output-dir "$OUT" --log-file "$LOG" --run-tag "$RUNTAG" \
      --language-model-only "$LMO" --batch-sizes "$BSS" --reps "$REPS"
)
RC=$?
echo "BENCH_VLLM_EXIT_CODE=$RC"
tail -20 "$LOG"
N=$(find "$OUT" -maxdepth 1 -name 'bs*.json' 2>/dev/null | wc -l)
echo "=== COLLECT_VLLM_DONE rc=$RC artifacts=$N $(date -Is) ==="
[ "$N" -gt 0 ] || exit 3
exit 0
