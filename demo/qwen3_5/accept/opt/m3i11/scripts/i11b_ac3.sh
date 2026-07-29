#!/usr/bin/env bash
# M3-I11 campaign 2: AC-3 sweep at all five batch sizes for one arm, then the
# AC-3 verdict and a per-case byte-diff against the committed pre-fix dumps.
# usage: ARM=fix|ctrl GPU=<id> bash i11b_ac3.sh
set -uo pipefail
ARM=${ARM:?}
GPU=${GPU:?}
B=$HOME/mpk-qwen35
MIRAGE=$B/m3i11b-$ARM
PY=$B/venv-rm/bin/python
export PYTHONPATH=$MIRAGE/python
export HF_HOME=$B/hf
export PYTHONUNBUFFERED=1
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_VISIBLE_DEVICES=$GPU
M=$B/m3i11b-out/$ARM
mkdir -p "$M/logs" "$M/ac3_dumps"
rm -rf "$M/ac3_dumps"/*
cd "$MIRAGE/demo/qwen3_5/accept"

echo "=== ARM=$ARM GPU=$GPU sha=$(git -C $MIRAGE rev-parse --short HEAD) $(date -Is) ==="
grep -n "store_async_wait<0>\|tma_store_wait<0>" \
  "$MIRAGE/include/mirage/persistent_kernel/tasks/blackwell/linear_sm100_mpk.cuh"

for BS in 1 2 4 8 16; do
  echo "=== AC3 bs=$BS $(date -Is) ==="
  $PY -u mpk_engine_run.py --batch-size $BS --max-seq-length 132 \
      --out-dir "$M/ac3_dumps" --kernel-dir "$M/kernel_ac3_bs$BS" \
      > "$M/logs/ac3_bs$BS.log" 2>&1
  echo "rc=$? $(date -Is)"
  tail -3 "$M/logs/ac3_bs$BS.log"
done

echo "=== run_ac3.py $(date -Is) ==="
$PY -u harness/run_ac3.py --engine-dump-dir "$M/ac3_dumps" \
    --batch-sizes 1,2,4,8,16 --output-json "$M/run_report.json" \
    > "$M/logs/run_ac3.log" 2>&1
echo "rc=$?"
tail -25 "$M/logs/run_ac3.log"

echo "=== per-case byte-diff vs committed results/dumps_final (pre-fix baseline) ==="
$PY -u "$B/m3i2a/bytediff.py" \
    "$MIRAGE/demo/qwen3_5/accept/results/dumps_final" "$M/ac3_dumps" 1,2,4,8,16 \
    > "$M/logs/bytediff.log" 2>&1
echo "rc=$?"
cat "$M/logs/bytediff.log"
echo "=== md5 of each dump ==="
md5sum "$M/ac3_dumps"/bs*.json
du -sh "$M"
echo "=== AC3_DRIVER_DONE ARM=$ARM $(date -Is) ==="
