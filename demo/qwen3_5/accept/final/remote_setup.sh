#!/usr/bin/env bash
# Box-side deploy for the final gate: an ISOLATED clone at the EXACT commit the
# gate is running for, with its OWN freshly built C++ extension.
#
# Why a fresh build every time the sha changes (STALE-EXTENSION TRAP,
# .memory/main/b200-env.md): the editable install builds a .so from src/kernel/*,
# and running new Python against an old .so produces
# `build_annotated_graph: bgraph inputs/outputs count mismatch` + rc=134 at every
# batch size -- a convincing fake "HEAD is broken" alarm.  A gate that measured
# that would report a false FAIL.
#
# Why a clone rather than the shared tree: `git reset --hard` in a shared clone
# clobbers a concurrent agent's uncommitted work, and three ferret chains plus
# other agents share this box.
#
# Idempotent: re-running for the same sha with an up-to-date .so is a no-op.
# Env: SHA (required), DEST, SRC, PY, GH_REMOTE.  Exit 0 ready / 2 failed.
set -uo pipefail
SHA="${SHA:?SHA is required}"
BOX_ROOT="${MPK_BOX_ROOT:-$HOME/mpk-qwen35}"
SRC="${SRC:-$BOX_ROOT/mirage}"
DEST="${DEST:-$BOX_ROOT/final-gate/tree-${SHA:0:12}}"
PY="${MPK_PY:-$BOX_ROOT/venv-rm/bin/python}"
GH_REMOTE="${GH_REMOTE:-git@github.com:bill810975/mirage.git}"
export PATH="${CUDA_BIN:-/usr/local/cuda-12.8/bin}:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export HF_HOME="${HF_HOME:-$BOX_ROOT/hf}"
mkdir -p "$(dirname "$DEST")"

echo "=== remote_setup sha=$SHA dest=$DEST $(date -Is) ==="
df -h /raid | tail -1

link_deps () {          # deps/* are submodules; borrow the mirror's checkouts
  for d in cutlass json z3; do
    rmdir "$1/deps/$d" 2>/dev/null
    ln -sfn "$SRC/deps/$d" "$1/deps/$d"
  done
  mkdir -p "$1/python/mirage/lib"
  cp -f "$SRC"/python/mirage/lib/*.so "$1/python/mirage/lib/" 2>/dev/null
}

# Resolve the request against the CLONE first.  Comparing $SHA to `rev-parse HEAD`
# as strings is wrong for a short sha -- it never matches, and the else branch
# below used to `rm -rf` a tree that was already correct.  Resolve both sides to
# full object ids instead, and never destroy anything before a checkout has
# actually succeeded (clone into .new, swap after).
CUR="$(git -C "$DEST" rev-parse HEAD 2>/dev/null || true)"
WANT="$(git -C "$DEST" rev-parse --verify --quiet "${SHA}^{commit}" 2>/dev/null || true)"
if [ -n "$CUR" ] && [ -n "$WANT" ] && [ "$CUR" = "$WANT" ]; then
  echo "clone already at $SHA ($CUR)"
  link_deps "$DEST"
elif [ -n "$WANT" ]; then
  echo "--- $SHA is already in this clone's object db; checking it out in place ---"
  git -C "$DEST" checkout --quiet "$WANT" || { echo "CHECKOUT FAILED for $SHA"; exit 2; }
  link_deps "$DEST"
else
  echo "--- fetch + fresh clone ---"
  if [ -d "$SRC/.git" ]; then
    git -C "$SRC" fetch --quiet origin qwen3-5_support || echo "WARN: origin fetch failed"
  fi
  NEW="$DEST.new.$$"
  rm -rf "$NEW"
  if [ -d "$SRC/.git" ]; then
    git clone --quiet --local --no-checkout "$SRC" "$NEW" || exit 2
  else
    git clone --quiet --no-checkout "$GH_REMOTE" "$NEW" || exit 2
  fi
  git -C "$NEW" remote add gh "$GH_REMOTE" 2>/dev/null || true
  git -C "$NEW" fetch --quiet gh qwen3-5_support || echo "WARN: gh fetch failed"
  if ! git -C "$NEW" checkout --quiet "$SHA"; then
    echo "CHECKOUT FAILED for $SHA -- the commit is not reachable from this box's"
    echo "mirror or from $GH_REMOTE.  A gate run needs its own commit to be"
    echo "fetchable here; an unpushed local commit must be shipped to $DEST"
    echo "directly (rsync) before running the gate."
    rm -rf "$NEW"
    exit 2
  fi
  link_deps "$NEW"
  rm -rf "$DEST"
  mv "$NEW" "$DEST"
fi
echo "clone HEAD: $(git -C "$DEST" rev-parse HEAD)"
echo "clone dirty: $(git -C "$DEST" status --porcelain | wc -l) path(s)"

SO="$(ls "$DEST"/python/mirage/core.cpython-*.so 2>/dev/null | head -1)"
NEWEST_SRC="$(find "$DEST/src" "$DEST/include" -newer "${SO:-$DEST/setup.py}" -type f 2>/dev/null | head -1)"
if [ -z "$SO" ] || [ -n "$NEWEST_SRC" ]; then
  echo "--- FRESH extension build (so=${SO:-none}, newer source: ${NEWEST_SRC:-n/a}) ---"
  ( cd "$DEST" && "$PY" setup.py build_ext --inplace ) > "$DEST/build_final_gate.log" 2>&1
  echo "BUILD_EXIT=$?"
  tail -4 "$DEST/build_final_gate.log"
  SO="$(ls "$DEST"/python/mirage/core.cpython-*.so 2>/dev/null | head -1)"
fi
[ -n "$SO" ] || { echo "NO EXTENSION BUILT"; exit 2; }
ls -la --time-style=+%F_%T "$SO"
md5sum "$SO"

echo "--- import sanity (must resolve to THIS clone) ---"
PYTHONPATH="$DEST/python" "$PY" -c "
import mirage, os, sys
p = os.path.realpath(mirage.__file__)
print('mirage.__file__ =', p)
sys.exit(0 if p.startswith(os.path.realpath('$DEST')) else 3)
" || { echo "IMPORT RESOLVES OUTSIDE THE CLONE"; exit 2; }

printf '{"sha":"%s","dest":"%s","so":"%s","so_md5":"%s","utc":"%s","host":"%s"}\n' \
  "$(git -C "$DEST" rev-parse HEAD)" "$DEST" "$SO" \
  "$(md5sum "$SO" | cut -d' ' -f1)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" \
  > "$DEST/final_gate_deploy.json"
cat "$DEST/final_gate_deploy.json"
echo "=== REMOTE_SETUP_DONE $(date -Is) ==="
exit 0
