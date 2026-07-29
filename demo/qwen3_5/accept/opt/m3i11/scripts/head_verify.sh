#!/usr/bin/env bash
# Verify origin HEAD in a FRESH clone with a FRESH extension build, to separate
# "source regression" from "stale compiled extension".
# usage: SHA=<sha> GPU=<id> [AC3=1] bash head_verify.sh
set -uo pipefail
SHA=${SHA:?}
GPU=${GPU:?}
AC3=${AC3:-0}
B=$HOME/mpk-qwen35
SRC=$B/mirage
D=$B/mirage-i11
PY=$B/venv-rm/bin/python
Z3=$B/venv-rm/lib/python3.12/site-packages/z3
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME=$B/hf PYTHONUNBUFFERED=1
export CC=/usr/bin/gcc CXX=/usr/bin/g++
M=$B/m3i11b-out/headverify
mkdir -p "$M"

echo "=== fresh clone at $SHA $(date -Is) ==="
rm -rf "$D"
git clone --quiet --local --no-checkout "$SRC" "$D"
git -C "$D" remote add gh git@github.com:bill810975/mirage.git 2>/dev/null || true
git -C "$D" fetch --quiet gh qwen3-5_support || echo "WARN: gh fetch failed"
git -C "$D" checkout --quiet "$SHA" || { echo "CHECKOUT FAILED"; exit 2; }
for d in cutlass json z3; do rmdir "$D/deps/$d" 2>/dev/null; ln -sfn "$SRC/deps/$d" "$D/deps/$d"; done
mkdir -p "$D/python/mirage/lib"
cp -f "$SRC"/python/mirage/lib/*.so "$D/python/mirage/lib/"
echo "clone HEAD: $(git -C "$D" rev-parse --short HEAD)  $(git -C "$D" log --oneline -1 | cut -c1-70)"

echo "=== FRESH extension build $(date -Is) ==="
cd "$D"
"$PY" setup.py build_ext --inplace > "$M/build_$SHA.log" 2>&1
echo "BUILD_EXIT=$? $(date -Is)"
tail -3 "$M/build_$SHA.log"
ls -la --time-style=+%F_%T "$D"/python/mirage/core.cpython-*.so
md5sum "$D"/python/mirage/core.cpython-*.so

export PYTHONPATH=$D/python
export CUDA_VISIBLE_DEVICES=$GPU
echo "=== graph-build smoke: bs1 msl132 on GPU$GPU $(date -Is) ==="
cd "$D/demo/qwen3_5/accept"
rm -rf "$M/smoke_$SHA"; mkdir -p "$M/smoke_$SHA"
"$PY" -u mpk_engine_run.py --batch-size 1 --max-seq-length 132 \
   --out-dir "$M/smoke_$SHA" --kernel-dir "$M/kd_${SHA}_bs1" \
   > "$M/smoke_$SHA.log" 2>&1
rc=$?
echo "SMOKE_EXIT=$rc $(date -Is)"
grep -iE "graph assembled|MPK INIT|bgraph inputs/outputs|runtime_error|wrote |Traceback" "$M/smoke_$SHA.log" | tail -8
[ "$rc" != 0 ] && tail -12 "$M/smoke_$SHA.log"

if [ "$AC3" = 1 ] && [ "$rc" = 0 ]; then
  echo "=== AC-3 arm at $SHA (all five bs) $(date -Is) ==="
  rm -rf "$M/ac3_$SHA"; mkdir -p "$M/ac3_$SHA"
  for BS in 1 2 4 8 16; do
    "$PY" -u mpk_engine_run.py --batch-size $BS --max-seq-length 132 \
        --out-dir "$M/ac3_$SHA" --kernel-dir "$M/kd_${SHA}_ac3bs$BS" \
        > "$M/ac3_bs${BS}_$SHA.log" 2>&1
    echo "  bs=$BS rc=$? $(date -Is)"
  done
  "$PY" -u harness/run_ac3.py --engine-dump-dir "$M/ac3_$SHA" \
      --batch-sizes 1,2,4,8,16 --output-json "$M/run_report_$SHA.json" \
      > "$M/run_ac3_$SHA.log" 2>&1
  echo "run_ac3 rc=$?"
  grep -E "FAIL|waiver-request|overall" "$M/run_ac3_$SHA.log" | tail -12
  "$PY" -u "$B/m3i2a/bytediff.py" \
      "$D/demo/qwen3_5/accept/results/dumps_final" "$M/ac3_$SHA" 1,2,4,8,16 \
      > "$M/bytediff_$SHA.log" 2>&1
  echo "bytediff rc=$?"
  grep -E "identical|counts|missing" "$M/bytediff_$SHA.log" | tail -10
fi
df -h /raid | tail -1
echo "=== HEAD_VERIFY_DONE SHA=$SHA $(date -Is) ==="
