#!/usr/bin/env bash
# M4-I6 GATE 1 -- the four router unit instruments, at per-bs invocations.
#
#   1a tests/.../sm100_moe/test_gate_topk.py                  (tie-aware)
#   1b tests/.../sm100_moe_sigmoid/test_gate_topk_sigmoid.py   (sibling kernel)
#   1c tests/.../sm100_moe_sigmoid/test_topk_sigmoid_testmode.py (full pipeline)
#   1d tests/.../sm100_moe_block_qwen35/test_router_oracle.py  (HF oracle)
#
# 1a and 1d are the ones that bind THIS change; 1b/1c cover the sigmoid sibling
# so a collateral regression in the shared header cannot pass unseen.
#
# WHY per-bs INVOCATIONS. 1a's own sweep is BATCH_SIZES x NUM_EXPERTS_LIST inside
# ONE process, which is exactly the coverage that matters for the row-tile loop
# (bs 17/33 sit past a pass) but pools every case into one CUDA context. The
# M3-I5b/I5c coverage discipline was one process per bs, so a per-bs failure
# cannot be masked by an earlier case's state and each cell has its own log. This
# driver runs BOTH: the full in-process sweep, then one process per bs over the
# AC-3 set {1,2,4,8,16} plus the past-the-cap sizes {17,33}.
#
# THE KNOWN TOOLCHAIN TRAP (add-mpk-task, "Step B"): these setup.py extensions
# link against the installed torch, so the nvcc that builds them must match
# `torch.version.cuda` (13.0 on this box), NOT the 12.8 the megakernel JIT uses.
# sm100_moe/setup.py takes a bare shutil.which("nvcc"), so PATH must be pinned
# per command or it silently takes whatever comes first -- or nothing.
set -uo pipefail
B=$HOME/mpk-qwen35
D=${D:-$B/mirage-m4i6}
M=${M:-$B/m4i6}
PY=$B/venv-rm/bin/python
ARM=${ARM:-cand}
export HF_HOME=$B/hf PYTHONUNBUFFERED=1
export QWEN35_ORACLE_DUMPS=${QWEN35_ORACLE_DUMPS:-$B/oracle-work/dumps}
# 1c compiles a MEGAKERNEL through the MPK JIT, which needs nvcc on PATH and
# resolves `import mirage` from sys.path. Both were traps on the first run:
# without PYTHONPATH the venv served mirage from $B/mirage-rm -- a DIFFERENT
# clone -- and without nvcc the JIT died with "nvcc not found". The JIT lane is
# 12.8 (the shipped megakernel toolchain, every driver in this campaign pins
# it); build_ext below overrides PATH per command to the torch-matched 13.0.
export PATH=/usr/local/cuda-12.8/bin:$PATH
export PYTHONPATH="$D/python${PYTHONPATH:+:$PYTHONPATH}"
G="$M/gate1_$ARM"
mkdir -p "$G"
: > "$G/summary.txt"

TMOE=$D/tests/runtime_python/blackwell/sm100_moe
TSIG=$D/tests/runtime_python/blackwell/sm100_moe_sigmoid
TQW=$D/tests/runtime_python/blackwell/sm100_moe_block_qwen35

echo "### M4-I6 GATE 1  arm=$ARM  clone=$D  $(date -Is)"
echo "### GPU claim: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader \
  | tee "$G/gpu_before.txt"
echo "### router header: $(sha256sum "$D/include/mirage/persistent_kernel/tasks/blackwell/topk_softmax_sm100.cuh" | cut -c1-16)"
"$PY" -c "import torch; print('### torch', torch.__version__, 'torch.version.cuda', torch.version.cuda)"

# nvcc must match torch.version.cuda -- see the header note.
TORCH_CUDA=$("$PY" -c "import torch; print(torch.version.cuda)")
CU=/usr/local/cuda-$TORCH_CUDA
[ -d "$CU" ] || { echo "REFUSING: no $CU for torch.version.cuda=$TORCH_CUDA"; exit 3; }
echo "### pinning CUDA_HOME=$CU (matches torch.version.cuda)"

build_ext () {  # $1 = test dir, $2 = tag
  echo "--- build $2 ($(basename "$1")) ---"
  cd "$1" || return 1
  rm -rf build ./*.so 2>/dev/null
  CUDA_HOME=$CU PATH=$CU/bin:$PATH "$PY" setup.py build_ext --inplace \
      > "$G/build_$2.log" 2>&1
  local rc=$?
  echo "BUILD_EXIT=$rc"
  [ $rc -ne 0 ] && { tail -25 "$G/build_$2.log"; return 1; }
  ls -la ./*.so | sed 's/^/  /'
  return 0
}

pass_fail () {  # $1 = log, $2 = label, $3 = rc
  local log="$1" label="$2" rc="$3" nf verdict
  nf=$(grep -cE '^(Test )?FAILED|AssertionError|^Traceback' "$log" 2>/dev/null)
  nf=${nf:-0}
  if [ "$rc" = 0 ] && [ "$nf" = 0 ]; then verdict=PASS; else verdict=FAIL; fi
  printf 'GATE1 %-46s rc=%-3s fail_lines=%-3s %s\n' \
      "$label" "$rc" "$nf" "$verdict" | tee -a "$G/summary.txt"
}

# ---------------------------------------------------------------- 1a
if build_ext "$TMOE" moe; then
  echo; echo "=== 1a test_gate_topk.py -- FULL in-process sweep (bs 1,8,9,16,17,33 x experts 128,256) ==="
  "$PY" test_gate_topk.py > "$G/1a_sweep.log" 2>&1; rc=$?
  grep -E '^(PASS|FAIL|bs=|All .* passed|=== SUMMARY)' "$G/1a_sweep.log" | tail -20
  pass_fail "$G/1a_sweep.log" "1a test_gate_topk FULL SWEEP" "$rc"

  echo; echo "=== 1a test_gate_topk.py -- ONE PROCESS PER BS ==="
  for bs in 1 2 4 8 16 17 33; do
    "$PY" - "$bs" > "$G/1a_bs$bs.log" 2>&1 <<'PYEOF'
import sys, runpy, types
bs = int(sys.argv[1])
mod = runpy.run_path("test_gate_topk.py", run_name="__loaded__")
# re-exec main() with a single-bs sweep: the module defines BATCH_SIZES at
# import time and main() reads the module global, so rebind it in the namespace
# the function closes over.
g = mod["main"].__globals__
g["BATCH_SIZES"] = [bs]
sys.argv = ["test_gate_topk.py"]
mod["main"]()
PYEOF
    pass_fail "$G/1a_bs$bs.log" "1a test_gate_topk bs=$bs" "$?"
  done
fi

# ---------------------------------------------------------------- 1b + 1c
if build_ext "$TSIG" sigmoid; then
  echo; echo "=== 1b test_gate_topk_sigmoid.py ==="
  "$PY" test_gate_topk_sigmoid.py > "$G/1b_sigmoid.log" 2>&1
  pass_fail "$G/1b_sigmoid.log" "1b test_gate_topk_sigmoid" "$?"
  tail -6 "$G/1b_sigmoid.log"

  echo; echo "=== 1c test_topk_sigmoid_testmode.py -- ONE PROCESS PER BS ==="
  for bs in 1 2 4 8 16 17; do
    "$PY" test_topk_sigmoid_testmode.py "$bs" > "$G/1c_bs$bs.log" 2>&1
    pass_fail "$G/1c_bs$bs.log" "1c test_topk_sigmoid_testmode bs=$bs" "$?"
  done
fi

# ---------------------------------------------------------------- 1d
if build_ext "$TQW" qwen35; then
  echo; echo "=== 1d test_router_oracle.py (HF dumps: $QWEN35_ORACLE_DUMPS) ==="
  "$PY" test_router_oracle.py > "$G/1d_oracle.log" 2>&1
  pass_fail "$G/1d_oracle.log" "1d test_router_oracle" "$?"
  tail -20 "$G/1d_oracle.log"
fi

echo
echo "=== GATE 1 SUMMARY  arm=$ARM ==="
cat "$G/summary.txt"
echo "  PASS: $(grep -c ' PASS$' "$G/summary.txt")  FAIL: $(grep -c ' FAIL$' "$G/summary.txt")"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tee "$G/gpu_after.txt"
echo "=== GATE1_M4I6_DONE arm=$ARM $(date -Is) ==="
grep -q ' FAIL$' "$G/summary.txt" && exit 1 || exit 0
