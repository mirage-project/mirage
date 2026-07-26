#!/usr/bin/env bash
# Materialise the M3-I8 A/B arms on the B200 box from the repo clone.
# Run ONCE when the window is armed, BEFORE plan_m3i8.sh. No GPU touched.
#
#   bash stage_arms.sh [/path/to/mirage]
#
# base : MOE_GATE_PADDING_ROWS = False, moe_n_splits = 2   (pre-M3-I8 graph)
# v1   : MOE_GATE_PADDING_ROWS = True,  moe_n_splits = 2   (the issue's change)
# v2a  : v1 + moe_n_splits = 4                             (grid widen, staged)
# v2b  : v1 + moe_n_splits = 8                             (grid widen, staged)
#
# v2a/v2b are NOT in the tree. The right-sizing makes the live expert set small,
# and only then is it worth splitting each expert's N further -- see
# predictions.md P6. They ship as arms so they are measured before anything is
# claimed for them.
set -euo pipefail
REPO=${1:-$HOME/mpk-qwen35/mirage}
M=$HOME/mpk-qwen35/m3i8
SRC_PK=$REPO/python/mirage/mpk/persistent_kernel.py
SRC_B=$REPO/python/mirage/mpk/models/qwen3_5/builder.py

for f in "$SRC_PK" "$SRC_B"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done
grep -q "^MOE_GATE_PADDING_ROWS" "$SRC_B" || {
  echo "REFUSING: $SRC_B has no MOE_GATE_PADDING_ROWS -- sync the M3-I8 tree first" >&2
  exit 1; }

mk() {  # mk <arm> <gate> <splits>
  local A=$M/arms/$1
  mkdir -p "$A/mpk/models/qwen3_5"
  cp "$SRC_PK" "$A/mpk/persistent_kernel.py"
  sed -e "s/^MOE_GATE_PADDING_ROWS = .*/MOE_GATE_PADDING_ROWS = $2/" \
      -e "s/^\( *\)self\.moe_n_splits = .*/\1self.moe_n_splits = $3/" \
      "$SRC_B" > "$A/mpk/models/qwen3_5/builder.py"
  echo "--- arm $1"
  grep -n "^MOE_GATE_PADDING_ROWS\|self.moe_n_splits" \
       "$A/mpk/models/qwen3_5/builder.py"
  sha256sum "$A/mpk/persistent_kernel.py" "$A/mpk/models/qwen3_5/builder.py"
}

mk base False 2
mk v1   True  2
mk v2a  True  4
mk v2b  True  8
echo "arms staged under $M/arms"
