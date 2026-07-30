#!/usr/bin/env bash
# M4-I6: isolated clone at the router-integration commit with its OWN freshly
# built C++ extension (STALE-EXTENSION TRAP, .memory/main/b200-env.md). Recipe
# from setup_m4i2.sh / setup_m4i4.sh; the commits arrive as a format-patch
# series applied with `git am` on top of origin/qwen3-5_support, because this
# session cannot push. `git am` rewrites committer dates, so the SHAs differ
# from the authoring clone's; the TREE hash is content-only and is asserted
# below to prove the trees are identical.
set -uo pipefail
B=$HOME/mpk-qwen35
SRC=$B/mirage
D=$B/mirage-m4i6
PY=$B/venv-rm/bin/python
M=$B/m4i6
WANT_TREE=${WANT_TREE:?the expected HEAD tree hash}
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME=$B/hf PYTHONUNBUFFERED=1
mkdir -p "$M/logs"

echo "=== df before $(date -Is) ==="; df -h /raid / | tail -2

echo "=== patches $(date -Is) ==="
rm -rf "$M/patches"
tar xzf "$B/m4i6_patches.tgz" -C "$M"
ls "$M/patches" | sed 's/^/  /'

echo "=== fresh clone $(date -Is) ==="
rm -rf "$D"
git clone --quiet --local --no-checkout "$SRC" "$D"
git -C "$D" remote add gh git@github.com:bill810975/mirage.git 2>/dev/null || true
git -C "$D" fetch --quiet gh qwen3-5_support || { echo "GH FETCH FAILED"; exit 2; }
git -C "$D" checkout --quiet -B m4i6 FETCH_HEAD || { echo "CHECKOUT FAILED"; exit 2; }
echo "base: $(git -C "$D" rev-parse --short HEAD) $(git -C "$D" log -1 --format=%s)"
git -C "$D" -c user.name=m4i6 -c user.email=m4i6@local am --quiet "$M"/patches/*.patch \
  || { echo "AM FAILED"; git -C "$D" am --abort; exit 2; }
GOT_TREE=$(git -C "$D" rev-parse HEAD^{tree})
echo "HEAD:      $(git -C "$D" rev-parse HEAD)"
git -C "$D" log --oneline -4
echo "tree want: $WANT_TREE"
echo "tree got:  $GOT_TREE"
[ "$WANT_TREE" = "$GOT_TREE" ] || { echo "TREE MISMATCH -- content differs"; exit 3; }
echo "TREE_MATCH: this clone's content is bit-identical to the authoring clone"
git -C "$D" status --short | head -5

for d in cutlass json z3; do rmdir "$D/deps/$d" 2>/dev/null; ln -sfn "$SRC/deps/$d" "$D/deps/$d"; done
mkdir -p "$D/python/mirage/lib"
cp -f "$SRC"/python/mirage/lib/*.so "$D/python/mirage/lib/" 2>/dev/null

echo "=== FRESH extension build $(date -Is) ==="
cd "$D"
"$PY" setup.py build_ext --inplace > "$M/logs/build.log" 2>&1
echo "BUILD_EXIT=$? $(date -Is)"
tail -6 "$M/logs/build.log"
ls -la --time-style=+%F_%T "$D"/python/mirage/core.cpython-*.so
md5sum "$D"/python/mirage/core.cpython-*.so

echo "=== import sanity (must resolve to THIS clone) ==="
PYTHONPATH="$D/python" "$PY" -c "
import mirage, os
print('mirage.__file__ =', os.path.realpath(mirage.__file__))
" 2>&1 | tail -4

echo "=== router provenance: the imported body must be the v013 candidate ==="
K="$D/include/mirage/persistent_kernel/tasks/blackwell/topk_softmax_sm100.cuh"
for tag in 'M4-I6' \
           'v013) BOUNDARY-WARP CONVERGED' \
           'v009) SORTED-LANE POP' \
           'v006) Shared-memory shadow bitmask' \
           'v007) PADDING-ROW COMPUTE SKIP' \
           'row_tile_base < num_rows' \
           'base + __popc(active_mask' \
           'extern __shared__ __align__(16) char smem'; do
  printf '  [%s] %s\n' "$(grep -c "$tag" "$K")" "$tag"
done
echo "  FERRET_DIAG leftovers (must be 0): $(grep -c FERRET_DIAG "$K")"
echo "  atomicAdd in the router (must be 0): $(grep -c atomicAdd "$K")"

echo "=== provenance ==="
sha256sum "$K" \
          "$D/python/mirage/mpk/models/qwen3_5/builder.py" \
          "$D/python/mirage/mpk/persistent_kernel.py" \
          "$D/src/kernel/task_register.cc" \
          "$D/demo/qwen3_5/accept/harness/gate_ac3_stable.sh" \
  | tee "$M/logs/provenance.txt"
df -h /raid / | tail -2
echo "=== SETUP_M4I6_DONE $(date -Is) ==="
