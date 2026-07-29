#!/usr/bin/env bash
# M4-I0 POSITIVE CONTROL: the campaign-2 census geometry (e4_full.py, bs4,
# msl 1280, 1024 new tokens, cold compile per rep, KV/GDN fingerprints) run in
# the SAME window as the AC-3-geometry gate.
#
# Purpose: a null result at the AC-3 geometry is uninterpretable on its own --
# it could mean the small geometry is not exposed, or it could mean the box is
# in a clean state tonight. This arm is the discriminator. Campaign 2 measured
# 10-16% per cold rep at THIS geometry.
set -uo pipefail
B=$HOME/mpk-qwen35
D=$B/mirage-m4i0
PY=$B/venv-rm/bin/python
M=$B/m4i0
OUTNAME=${OUTNAME:?}
REPS=${REPS:-20}
export PYTHONPATH=$D/python
export HF_HOME=$B/hf
export PYTHONUNBUFFERED=1
export PATH=/usr/local/cuda-12.8/bin:$PATH
export ACC=$D/demo/qwen3_5/accept
OUT=$M/$OUTNAME
mkdir -p "$OUT/logs" "$OUT/fps" "$M/kernels-$OUTNAME"
cd "$ACC"

one() {  # $1 = rep index
  local r="$1" t="pc_c$r" kd="$M/kernels-$OUTNAME/cold_pc_c$r"
  rm -rf "$kd"
  # per-rep drain: wait for the previous rep's context to actually go away
  for i in $(seq 1 60); do
    read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits -i "$PHYS" | tr -d ' ' | tr ',' ' ')
    if [ "${used:-999999}" -le "$((FLOOR + 600))" ] && [ "${util:-100}" -le 5 ]; then break; fi
    echo "  drain GPU $PHYS used=${used}MiB util=${util}% ($i)"; sleep 10
  done
  echo "  gpu_before: GPU $PHYS used=${used}MiB util=${util}%"
  "$PY" -u "$ACC/opt/m3i11/scripts/e4_full.py" \
      --out "$OUT/fps" --tag "$t" --bs 4 \
      --kernel-dir "$kd" --fresh-compile > "$OUT/logs/$t.log" 2>&1
  local rc=$?
  echo "  $t rc=$rc $(grep -o 'md5=[0-9a-f]*' "$OUT/logs/$t.log" | tail -1) $(date -Is)"
  [ "$rc" != 0 ] && tail -6 "$OUT/logs/$t.log" | sed 's/^/    /'
  rm -rf "$kd"
}

PHYS="${CUDA_VISIBLE_DEVICES:?guard must pin one device}"
FLOOR="${MPK_I7_GPU_FLOOR:-0}"
echo "=== POSITIVE CONTROL (1024-tok census) GPU=$PHYS floor=${FLOOR}MiB reps=$REPS $(date -Is) ==="
for r in $(seq 1 "$REPS"); do
  echo "=== pc rep $r/$REPS $(date -Is) ==="
  one "$r"
  df -h /raid | tail -1
done
echo "=== fingerprint census $(date -Is) ==="
"$PY" "$ACC/opt/m3i11/scripts/i11b_fpdiff.py" "$OUT/fps" 2>&1 | tee "$OUT/fpdiff.txt"
echo "=== POSCONTROL_DONE $(date -Is) ==="
