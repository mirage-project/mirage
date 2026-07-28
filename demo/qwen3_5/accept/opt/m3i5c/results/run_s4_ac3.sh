#!/usr/bin/env bash
set -uo pipefail
MIRAGE=$HOME/mpk-qwen35/mirage-i5c-run
PY=$HOME/mpk-qwen35/venv-rm/bin/python
export PYTHONPATH=$MIRAGE/python
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
export PATH=/usr/local/cuda-12.8/bin:$PATH
GPU=${GPU:?}
export CUDA_VISIBLE_DEVICES=$GPU
M=$HOME/mpk-qwen35/m3i5c
cd $MIRAGE/demo/qwen3_5/accept
for BS in 1 2 4 8 16; do
  echo "=== AC3 bs=$BS $(date -Is) ==="
  $PY -u mpk_engine_run.py --batch-size $BS --max-seq-length 132 \
      --out-dir $M/ac3_dumps --kernel-dir $M/kernel_ac3_bs$BS \
      > $M/logs/s4_bs$BS.log 2>&1
  echo "rc=$? $(date -Is)"
  tail -3 $M/logs/s4_bs$BS.log
done
echo "=== run_ac3.py ==="
$PY -u harness/run_ac3.py --engine-dump-dir $M/ac3_dumps \
    --batch-sizes 1,2,4,8,16 --output-json $M/run_report.json > $M/logs/s4_run_ac3.log 2>&1
echo "rc=$?"
tail -30 $M/logs/s4_run_ac3.log
echo "=== bytediff.py vs committed dumps_final ==="
$PY -u $HOME/mpk-qwen35/m3i2a/bytediff.py \
    $MIRAGE/demo/qwen3_5/accept/results/dumps_final $M/ac3_dumps 1,2,4,8,16 \
    > $M/logs/s4_bytediff.log 2>&1
echo "rc=$?"
cat $M/logs/s4_bytediff.log
echo "S4_DRIVER_DONE"
