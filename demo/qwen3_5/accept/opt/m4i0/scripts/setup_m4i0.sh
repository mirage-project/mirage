#!/usr/bin/env bash
# M4-I0: isolated clone at integrated HEAD with its OWN freshly-built C++
# extension (STALE-EXTENSION TRAP, .memory/main/b200-env.md). Recipe mirrored
# from i7/setup_i7.sh -- build in place with setup.py build_ext --inplace, then
# drive with PYTHONPATH=<clone>/python so no venv editable pointer is touched.
set -uo pipefail
SHA=${SHA:?}
B=$HOME/mpk-qwen35
SRC=$B/mirage
D=$B/mirage-m4i0
PY=$B/venv-rm/bin/python
M=$B/m4i0
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME=$B/hf PYTHONUNBUFFERED=1
mkdir -p "$M/logs"

echo "=== df before $(date -Is) ==="; df -h /raid | tail -1

echo "=== refresh the source mirror ==="
git -C "$SRC" fetch --quiet origin qwen3-5_support || echo "WARN: origin fetch failed"

echo "=== fresh clone at $SHA $(date -Is) ==="
rm -rf "$D"
git clone --quiet --local --no-checkout "$SRC" "$D"
git -C "$D" remote add gh git@github.com:bill810975/mirage.git 2>/dev/null || true
git -C "$D" fetch --quiet gh qwen3-5_support || echo "WARN: gh fetch failed"
git -C "$D" checkout --quiet "$SHA" || { echo "CHECKOUT FAILED"; exit 2; }
for d in cutlass json z3; do rmdir "$D/deps/$d" 2>/dev/null; ln -sfn "$SRC/deps/$d" "$D/deps/$d"; done
mkdir -p "$D/python/mirage/lib"
cp -f "$SRC"/python/mirage/lib/*.so "$D/python/mirage/lib/" 2>/dev/null
echo "clone HEAD: $(git -C "$D" rev-parse HEAD)"
git -C "$D" log --oneline -1

echo "=== FRESH extension build $(date -Is) ==="
cd "$D"
"$PY" setup.py build_ext --inplace > "$M/logs/build.log" 2>&1
echo "BUILD_EXIT=$? $(date -Is)"
tail -4 "$M/logs/build.log"
ls -la --time-style=+%F_%T "$D"/python/mirage/core.cpython-*.so
md5sum "$D"/python/mirage/core.cpython-*.so

echo "=== import sanity (must resolve to THIS clone) ==="
PYTHONPATH="$D/python" "$PY" -c "
import mirage, os
print('mirage.__file__ =', os.path.realpath(mirage.__file__))
from mirage.kernel import get_key_paths
r,i,dp = get_key_paths(); print('key paths ok:', os.path.isdir(i), os.path.isdir(dp))
" 2>&1 | tail -8

echo "=== provenance ==="
sha256sum "$D/demo/qwen3_5/accept/mpk_engine_run.py" \
          "$D/demo/qwen3_5/accept/harness/run_ac3.py" \
          "$D/python/mirage/mpk/models/qwen3_5/builder.py" \
          "$D/include/mirage/persistent_kernel/tasks/blackwell/linear_sm100_mpk.cuh" \
  | tee "$M/logs/provenance.txt"
grep -n "store_async_wait<0>\|tma_store_wait<0>" \
  "$D/include/mirage/persistent_kernel/tasks/blackwell/linear_sm100_mpk.cuh"
df -h /raid | tail -1
echo "=== SETUP_M4I0_DONE $(date -Is) ==="
