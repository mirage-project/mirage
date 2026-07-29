#!/usr/bin/env bash
# M3-I11 campaign 2 in-runtime unit gate for the changed kernel.
#
# These two test_mode graphs run linear_layer (task linear_sm100 ->
# linear_sm100_mpk, the kernel this issue changes) inside the REAL persistent
# runtime as a producer whose output a later task consumes -- exactly the
# pattern the TMA store-visibility defect affects -- and check the result
# against a torch reference. That makes them a better instrument for this change
# than the standalone sm100_linear kernel-wrapper tests, which launch the task
# with no consumer at all (and which hang at this SHA in BOTH arms, see report).
# usage: GPU=<id> REPS=<n> bash i11b_testmode.sh
set -uo pipefail
GPU=${GPU:?}
REPS=${REPS:-3}
B=$HOME/mpk-qwen35
PY=$B/venv-rm/bin/python
export HF_HOME=$B/hf
export PYTHONUNBUFFERED=1
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_VISIBLE_DEVICES=$GPU
M=$B/m3i11b-out/testmode
mkdir -p "$M"

TESTS="tests/runtime_python/test_mode/test_diamond_fork_join_testmode.py
tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py"

for arm in ctrl fix; do
  mirage=$B/m3i11b-$arm
  echo "=== ARM=$arm $(date -Is) ==="
  grep -n "store_async_wait<0>\|tma_store_wait<0>" \
    "$mirage/include/mirage/persistent_kernel/tasks/blackwell/linear_sm100_mpk.cuh"
  cd "$mirage"
  for t in $TESTS; do
    n=$(basename "$t" .py)
    for r in $(seq 1 "$REPS"); do
      PYTHONPATH=$mirage/python timeout 900 "$PY" "$t" \
          > "$M/${n}_${arm}_r${r}.log" 2>&1
      rc=$?
      echo "  $n arm=$arm rep=$r rc=$rc PASSED=$(grep -ci 'PASS' "$M/${n}_${arm}_r${r}.log") FAILED=$(grep -ci 'FAIL\|Traceback' "$M/${n}_${arm}_r${r}.log")  $(date -Is)"
      grep -iE "^(PASSED|FAILED|PASS|FAIL)" "$M/${n}_${arm}_r${r}.log" | tail -3 | sed 's/^/      /'
      if [ "$rc" != 0 ]; then tail -6 "$M/${n}_${arm}_r${r}.log" | sed 's/^/      /'; fi
    done
  done
done
echo "=== TESTMODE_DRIVER_DONE $(date -Is) ==="
