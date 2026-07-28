#!/usr/bin/env bash
set -uo pipefail
MIRAGE=$HOME/mpk-qwen35/mirage-i5c-run
PY=$HOME/mpk-qwen35/venv-rm/bin/python
export PYTHONPATH=$MIRAGE/python:$MIRAGE/tests/runtime_python/blackwell/sm100_moe_block_qwen35
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
export PATH=/usr/local/cuda-13.0/bin:$PATH
GPU=${GPU:?}
export CUDA_VISIBLE_DEVICES=$GPU
M=$HOME/mpk-qwen35/m3i5c
cd $MIRAGE/tests/runtime_python/blackwell/sm100_moe_block_qwen35
echo "=== rebuild ext (clean state) ==="
$PY setup.py build_ext --inplace > $M/logs/s3_build.log 2>&1; echo "build rc=$?"
echo "=== rows=16, iters=2000 ==="
$PY $MIRAGE/demo/qwen3_5/accept/opt/m3i5c/stress_compaction.py --iters 2000 --rows 16 \
    --out $M/stress_rows16.json 2>&1 | tee $M/logs/s3_rows16.log
echo "rc=$?"
for R in 32 64 128; do
  echo "=== rows=$R, iters=1000 ==="
  $PY $MIRAGE/demo/qwen3_5/accept/opt/m3i5c/stress_compaction.py --iters 1000 --rows $R \
      --out $M/stress_rows$R.json 2>&1 | tee $M/logs/s3_rows$R.log
  echo "rc=$?"
done
echo "S3_DRIVER_DONE"
