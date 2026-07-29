#!/usr/bin/env bash
# M4-I4: isolated clone at the policy-landing commit with its OWN freshly-built
# C++ extension (STALE-EXTENSION TRAP, .memory/main/b200-env.md). Recipe from
# opt/m4i0/scripts/setup_m4i0.sh; the only difference is that the commit under
# test is NOT on the remote (this issue commits locally, never pushes), so it
# arrives as a git bundle.
set -uo pipefail
BUNDLE=${BUNDLE:-$HOME/mpk-qwen35/m4i4.bundle}
SHA=${SHA:?the commit to test}
B=$HOME/mpk-qwen35
SRC=$B/mirage
D=$B/mirage-m4i4
PY=$B/venv-rm/bin/python
M=$B/m4i4
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME=$B/hf PYTHONUNBUFFERED=1
mkdir -p "$M/logs"

echo "=== df before $(date -Is) ==="; df -h /raid | tail -1

echo "=== fresh clone $(date -Is) ==="
rm -rf "$D"
git clone --quiet --local --no-checkout "$SRC" "$D"
git -C "$D" remote add gh git@github.com:bill810975/mirage.git 2>/dev/null || true
git -C "$D" fetch --quiet gh qwen3-5_support || echo "WARN: gh fetch failed"
git -C "$D" fetch --quiet "$BUNDLE" 'refs/heads/*:refs/remotes/bundle/*' \
  || { echo "BUNDLE FETCH FAILED"; exit 2; }
git -C "$D" checkout --quiet "$SHA" || { echo "CHECKOUT FAILED"; exit 2; }
for d in cutlass json z3; do rmdir "$D/deps/$d" 2>/dev/null; ln -sfn "$SRC/deps/$d" "$D/deps/$d"; done
mkdir -p "$D/python/mirage/lib"
cp -f "$SRC"/python/mirage/lib/*.so "$D/python/mirage/lib/" 2>/dev/null
echo "clone HEAD: $(git -C "$D" rev-parse HEAD)"
git -C "$D" log --oneline -1
git -C "$D" status --short | head -5

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
import inspect
print('runtime cap knob present:',
      'max_tokens_per_request' in inspect.signature(mirage.PersistentKernel.__init__).parameters)
" 2>&1 | tail -8

echo "=== the policy this tree ships ==="
"$PY" "$D/demo/qwen3_5/accept/admission_policy.py"
"$PY" "$D/demo/qwen3_5/accept/admission_policy.py" --describe
"$PY" "$D/demo/qwen3_5/accept/harness/tests/test_admission_policy.py" 2>&1 | tail -3

echo "=== provenance ==="
sha256sum "$D/demo/qwen3_5/accept/admission_policy.py" \
          "$D/demo/qwen3_5/accept/mpk_engine_run.py" \
          "$D/demo/qwen3_5/accept/harness/gate_ac3_stable.py" \
          "$D/demo/qwen3_5/accept/harness/gate_ac3_stable.sh" \
          "$D/python/mirage/mpk/persistent_kernel.py" \
          "$D/include/mirage/persistent_kernel/persistent_kernel.cuh" \
  | tee "$M/logs/provenance.txt"
df -h /raid | tail -1
echo "=== SETUP_M4I4_DONE $(date -Is) ==="
