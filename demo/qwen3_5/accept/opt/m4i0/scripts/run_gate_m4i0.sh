#!/usr/bin/env bash
# M4-I0 driver: claim ONE stable-idle GPU with the repo guard, then run the
# fingerprint-scored cold AC-3 stability gate inside the isolated m4i0 clone.
set -uo pipefail
B=$HOME/mpk-qwen35
D=$B/mirage-m4i0
PY=$B/venv-rm/bin/python
M=$B/m4i0
export PYTHONPATH=$D/python
export HF_HOME=$B/hf
export PYTHONUNBUFFERED=1
export PATH=/usr/local/cuda-12.8/bin:$PATH
OUTNAME=${OUTNAME:?}
mkdir -p "$M/$OUTNAME"
cd "$D/demo/qwen3_5/accept"
echo "=== M4-I0 gate driver OUT=$OUTNAME $(date -Is) sha=$(git -C "$D" rev-parse --short HEAD) ==="
exec bash opt/m3i7/scripts/gpu_guard_i7.sh "${CANDS:-6,5,2,0,1,4,7,3}" -- \
  bash "harness/${GATESH:-gate_ac3_stable.sh}" \
    --out "$M/$OUTNAME" \
    --python "$PY" \
    --reps "${REPS:-3}" \
    --batch-sizes "${BSS:-1,2,4,8,16}" \
    --max-extra "${MAXEXTRA:-3}" \
    --kernel-root "$M/kernels-$OUTNAME" \
    ${CAP:+--per-request-token-cap "$CAP"}
