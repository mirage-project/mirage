#!/usr/bin/env bash
# M4-I6 BASE arm: a second isolated clone pinned to the router integration's
# PARENT commit (5e48eaab, M4-I2 terminal), with its OWN freshly built C++
# extension. The A/B arms must differ in exactly one thing -- the router body --
# so the base arm gets the same clone/build recipe rather than a reverted header
# in the candidate tree (STALE-EXTENSION TRAP + "two arms sharing a kernel dir
# execute one binary", add-mpk-task "A Kernel Directory Must Carry Every
# Compile-Time -D Knob").
set -uo pipefail
B=$HOME/mpk-qwen35
SRC=$B/mirage-m4i6
D=$B/mirage-m4i6-base
PY=$B/venv-rm/bin/python
M=$B/m4i6
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME=$B/hf PYTHONUNBUFFERED=1
mkdir -p "$M/logs"

echo "=== base clone $(date -Is) ==="
rm -rf "$D"
git clone --quiet --local --no-checkout "$SRC" "$D"
git -C "$D" checkout --quiet -B m4i6base m4i6~1 || { echo "CHECKOUT FAILED"; exit 2; }
echo "HEAD: $(git -C "$D" rev-parse --short HEAD) $(git -C "$D" log -1 --format=%s)"
git -C "$D" status --short | head -5

for d in cutlass json z3; do rmdir "$D/deps/$d" 2>/dev/null; ln -sfn "$B/mirage/deps/$d" "$D/deps/$d"; done
mkdir -p "$D/python/mirage/lib"
cp -f "$B/mirage"/python/mirage/lib/*.so "$D/python/mirage/lib/" 2>/dev/null

echo "=== FRESH extension build $(date -Is) ==="
cd "$D"
"$PY" setup.py build_ext --inplace > "$M/logs/build_base.log" 2>&1
echo "BUILD_EXIT=$? $(date -Is)"
ls -la --time-style=+%F_%T "$D"/python/mirage/core.cpython-*.so
md5sum "$D"/python/mirage/core.cpython-*.so

echo "=== the two arms differ in exactly ONE file ==="
git -C "$B/mirage-m4i6" diff --stat m4i6~1 m4i6

echo "=== base router is the pre-M4-I6 body ==="
K="$D/include/mirage/persistent_kernel/tasks/blackwell/topk_softmax_sm100.cuh"
echo "  M4-I6 markers (must be 0): $(grep -c 'M4-I6' "$K")"
echo "  atomicAdd (must be 0, M3-I5c): $(grep -c atomicAdd "$K")"
sha256sum "$K" | tee -a "$M/logs/provenance.txt"
echo "=== SETUP_M4I6_BASE_DONE $(date -Is) ==="
