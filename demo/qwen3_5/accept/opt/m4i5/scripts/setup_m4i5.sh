#!/usr/bin/env bash
# M4-I5 environment: an ISOLATED clone at the pinned basis, with its OWN freshly
# built C++ extension.
#
# Build recipe from ~/mpk-qwen35/i7/setup_i7.sh (M3-I7): build in place with
# `setup.py build_ext --inplace` and drive with PYTHONPATH=<clone>/python, so no
# venv's editable pointer is touched and two clones cannot fight over one .so
# (.memory/main/b200-env.md STALE-EXTENSION TRAP).
#
# Why the two-step base+overlay: the pinned basis is a LOCAL commit that is not
# on origin yet, and `remote_setup.sh` says such a commit must be shipped
# directly.  BASE is the newest ancestor this box can fetch; the caller then
# streams `git archive <pinned> -- <changed paths>` into $D.  Everything between
# BASE and the pinned sha is under demo/qwen3_5/accept/**, .claude/skills/** and
# .gitignore -- NO src/, include/ or python/ change -- so the C++ extension
# built at BASE is the extension of the pinned sha.  That invariant is asserted
# here rather than assumed.
set -uo pipefail
BASE=${BASE:?want the fetchable base sha}
PINNED=${PINNED:?want the pinned basis sha (for the record)}
B=$HOME/mpk-qwen35
SRC=$B/mirage
D=$B/mirage-m4i5
PY=$B/venv-rm/bin/python
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME=$B/hf PYTHONUNBUFFERED=1
export CC=/usr/bin/gcc CXX=/usr/bin/g++
M=$B/m4i5
mkdir -p "$M/logs"

echo "=== setup_m4i5 base=$BASE pinned=$PINNED $(date -Is) ==="
df -h /raid / | tail -2

git -C "$SRC" fetch --quiet origin qwen3-5_support || echo "WARN: origin fetch failed"

echo "=== fresh clone at BASE $(date -Is) ==="
rm -rf "$D" "$D.new.$$"
git clone --quiet --local --no-checkout "$SRC" "$D.new.$$" || exit 2
git -C "$D.new.$$" checkout --quiet "$BASE" || { echo "CHECKOUT FAILED for $BASE"; exit 2; }
for d in cutlass json z3; do
  rmdir "$D.new.$$/deps/$d" 2>/dev/null
  ln -sfn "$SRC/deps/$d" "$D.new.$$/deps/$d"
done
mkdir -p "$D.new.$$/python/mirage/lib"
cp -f "$SRC"/python/mirage/lib/*.so "$D.new.$$/python/mirage/lib/" 2>/dev/null
mv "$D.new.$$" "$D"
echo "clone HEAD: $(git -C "$D" rev-parse HEAD)"

echo "=== FRESH extension build $(date -Is) ==="
( cd "$D" && "$PY" setup.py build_ext --inplace ) > "$M/logs/build.log" 2>&1
echo "BUILD_EXIT=$? $(date -Is)"
tail -4 "$M/logs/build.log"
SO="$(ls "$D"/python/mirage/core.cpython-*.so 2>/dev/null | head -1)"
[ -n "$SO" ] || { echo "NO EXTENSION BUILT"; exit 2; }
ls -la --time-style=+%F_%T "$SO"; md5sum "$SO"

echo "=== import sanity (must resolve to THIS clone) ==="
PYTHONPATH="$D/python" "$PY" -c "
import mirage, os, sys
p = os.path.realpath(mirage.__file__); print('mirage.__file__ =', p)
sys.exit(0 if p.startswith(os.path.realpath('$D')) else 3)
" || { echo "IMPORT RESOLVES OUTSIDE THE CLONE"; exit 2; }
echo "=== SETUP DONE (awaiting overlay) $(date -Is) ==="
