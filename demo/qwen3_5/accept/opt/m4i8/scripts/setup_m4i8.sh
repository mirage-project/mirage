#!/usr/bin/env bash
# M4-I8: isolated clone at origin/qwen3-5_support HEAD (5756c789) with its OWN
# freshly-built C++ extension (STALE-EXTENSION TRAP). Pattern from setup_m4i7.sh.
set -uo pipefail
B=$HOME/mpk-qwen35
SRC=$B/mirage
D=$B/mirage-m4i8
PY=$B/venv-rm/bin/python
M=$B/m4i8
PIN=${PIN:-5756c789bb9ed1afa1a0e790377c72cc1136079c}
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME=$B/hf PYTHONUNBUFFERED=1
mkdir -p "$M/logs"
echo "=== df before $(date -Is) ==="; df -h /raid / /var/tmp | tail -3
echo "=== fresh clone $(date -Is) ==="
rm -rf "$D"
git clone --quiet --local --no-checkout "$SRC" "$D" || { echo "CLONE FAILED"; exit 2; }
git -C "$D" remote add gh git@github.com:bill810975/mirage.git 2>/dev/null || true
git -C "$D" fetch --quiet gh qwen3-5_support || { echo "GH FETCH FAILED"; exit 2; }
git -C "$D" checkout --quiet -B m4i8 "$PIN" || { echo "CHECKOUT FAILED"; exit 2; }
echo "base: $(git -C "$D" rev-parse HEAD) $(git -C "$D" log -1 --format=%s)"
git -C "$D" status --short | head -5
for d in cutlass json z3; do rmdir "$D/deps/$d" 2>/dev/null; ln -sfn "$SRC/deps/$d" "$D/deps/$d"; done
mkdir -p "$D/python/mirage/lib"
cp -f "$SRC"/python/mirage/lib/*.so "$D/python/mirage/lib/" 2>/dev/null
echo "=== FRESH extension build $(date -Is) ==="
cd "$D"
"$PY" setup.py build_ext --inplace > "$M/logs/build.log" 2>&1
echo "BUILD_EXIT=$? $(date -Is)"
tail -5 "$M/logs/build.log"
ls -la --time-style=+%F_%T "$D"/python/mirage/core.cpython-*.so
md5sum "$D"/python/mirage/core.cpython-*.so
echo "=== import sanity (must resolve to THIS clone) ==="
PYTHONPATH="$D/python" "$PY" -c "
import mirage, os
print('mirage:', os.path.realpath(mirage.__file__))
import mirage.core as c; print('core:', os.path.realpath(c.__file__))
"
echo "=== df after $(date -Is) ==="; df -h /raid / /var/tmp | tail -3
echo "M4I8_SETUP_DONE $(date -Is)"
