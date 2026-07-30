#!/usr/bin/env bash
# M4-I9 GATE 1c -- is the SHIPPED DEFAULT (MPK_FUSE_SILU_QUANT unset) unchanged?
#
# The fusion is gated on an env var read in the BUILDER, not on an #ifdef, and
# the change also touches code that is on every path: a new enum value in
# runtime_header.h, a new entry in runtime.cc's task_type_to_name, a new case in
# tma.cuh's switch, a new #include in blackwell/task_header.cuh, and a new
# `else` branch in the builder. "That is all inert when the flag is off" is
# exactly the claim a reviewer should not have to take on trust.
#
# So: compare the arm-A GENERATED TU and its SASS against a PRISTINE reference
# generated at the SAME geometry by a tree that has none of this issue's code --
# M4-I8's own clone at 5756c789, whose /var/tmp/m4i8_sweep/kernel_A_bs1 dir was
# produced by the same harness at the same msl=353 / bs1 / mbt=16 / page 256.
# 5756c789's default path is the same as 8ff7be39's: M4-I8's gate 1c proved its
# only header change is __LINE__ immediates on the default path.
#
# PASS = the generated TU is byte-identical, therefore the SASS is too. That is
# the strong form and it is available here because the default branch emits
# exactly the ops it emitted before.
#
# Compile-only: no CUDA API call, no launch, no GPU claim.
set -u
export PATH=/usr/local/cuda-12.8/bin:$PATH
M=${M:-$HOME/mpk-qwen35/m4i9}
MINE=${MINE:-/var/tmp/m4i9_sweep/kernel_A_bs1/test_rank0.cu}
REF=${REF:-/var/tmp/m4i8_sweep/kernel_A_bs1/test_rank0.cu}
OUT=$M/ptxas
mkdir -p "$OUT"

echo "########## M4-I9 gate 1c: shipped default unchanged  $(date -Is) ##########"
for f in "$MINE" "$REF"; do
  [ -f "$f" ] || { echo "MISSING TU: $f"; exit 2; }
  printf "  %s\n    sha256 %s  lines %s  bytes %s\n" "$f" \
    "$(sha256sum "$f" | cut -c1-32)" "$(wc -l < "$f")" "$(stat -c%s "$f")"
done

echo
echo "=== generated-TU diff (arm A, M4-I9 tree) vs (pristine, M4-I8 tree 5756c789) ==="
if cmp -s "$MINE" "$REF"; then
  echo "  BYTE-IDENTICAL. The shipped default emits exactly the same TU."
  NDIFF=0
else
  NDIFF=$(diff "$MINE" "$REF" | grep -c '^[<>]')
  echo "  DIFFERS: $NDIFF changed lines -- every one must be accounted for."
  diff "$MINE" "$REF" | head -60
fi

echo
echo "=== task-type census, both TUs (must match) ==="
for f in "$MINE" "$REF"; do
  printf "  %-46s silu=%s quantize=%s fused=%s\n" "$(basename "$(dirname "$f")")" \
    "$(grep -c 'TASK_SILU_MUL' "$f")" \
    "$(grep -c 'TASK_QUANTIZE_FP8_SM100' "$f")" \
    "$(grep -c 'TASK_MOE_SILU_MUL_QUANTIZE_FP8_SM100' "$f")"
done

echo
echo "=== graph shape, both trees (task/event counts must match) ==="
for d in "$(dirname "$MINE")" "$(dirname "$REF")"; do
  G=$d/task_graph_rank0.json
  [ -f "$G" ] || { echo "  $(basename "$d"): no graph"; continue; }
  $HOME/mpk-qwen35/venv-rm/bin/python - "$G" <<'PYEOF'
import json, sys, collections
g = json.load(open(sys.argv[1]))
c = collections.Counter(t["task_type"] for t in g["all_tasks"])
print(f"  {sys.argv[1].split('/')[-2]:<28s} tasks={len(g['all_tasks'])} "
      f"events={len(g['all_events'])} t118={c.get(118,0)} t275={c.get(275,0)} "
      f"t243={c.get(243,0)}")
PYEOF
done

echo
echo "GATE1C_VERDICT: $([ "$NDIFF" -eq 0 ] && echo 'PASS (byte-identical TU)' || echo "REVIEW ($NDIFF lines)")"
echo "CHECK_DEFAULT_M4I9_DONE $(date -Is)"
