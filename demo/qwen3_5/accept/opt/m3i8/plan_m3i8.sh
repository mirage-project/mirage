#!/usr/bin/env bash
# M3-I8 validation plan -- WRITTEN, NOT ARMED.
#
# All 8 B200s are contended and M3-I2b owns the next window. This script is the
# prepared capture; the COORDINATOR sequences windows, so it refuses to start
# until it is explicitly armed:
#
#     M3I8_ARMED=1 nohup bash ~/mpk-qwen35/m3i8/plan_m3i8.sh \
#         > ~/mpk-qwen35/m3i8/logs/plan.log 2>&1 &
#
# It also refuses while M3-I2b still holds a GPU lock, so an accidental launch
# cannot land on top of the parked window.
#
# Stage order is VALUE order, so a short window still settles the issue:
#
#  0  rebuild + codegen identity: `base` must regenerate the pre-M3-I8 router
#     call byte-for-byte, `v1` must differ by exactly the qo_indptr argument
#  1  MECHANISM: mask_probe base vs v1, all five bs   -- the primary falsifier
#  2  AC-3 gate on v1 (inert-at-bs16 first, then the full sweep + byte diff)
#  3  perf bs8 then bs1, base then v1  -- bs8 is the ONE case the cost model
#     says crosses a worker-wave boundary; bs1 is where the backlog claimed
#     +37% and the model says ~0
#  4  perf bs 2,4,16
#  5  v2a (moe_n_splits=4): perf bs1,8 + AC-3
#  6  v2b (moe_n_splits=8): perf bs1,8
#  7  analysis
set -uo pipefail

M=$HOME/mpk-qwen35/m3i8
REPO=${REPO:-$HOME/mpk-qwen35/mirage}
CANDS=${CANDS:-6,0,1,2,3,4,5,7}
MAXWAIT_MIN=${MAXWAIT_MIN:-600}
STAGES=${STAGES:-0,1,2,3,4,5,6,7}
export PATH=/usr/local/cuda-12.8/bin:$PATH
PY=$HOME/mpk-qwen35/venv-mpk/bin/python

# ------------------------------------------------------------------ arming
if [ "${M3I8_ARMED:-0}" != "1" ]; then
  cat >&2 <<'EOF'
REFUSING: M3-I8's window is not armed.

This capture is prepared, reviewed and parked. M3-I2b owns the next GPU window;
the coordinator sequences them. Re-run with M3I8_ARMED=1 once I2b's window has
completed and its arms have been removed from the shared mirage clone.
EOF
  exit 98
fi
for L in "$HOME/mpk-qwen35/.gpu-locks/"M3-I2b.lock; do
  if [ -f "$L" ]; then
    echo "REFUSING: $L still exists -- M3-I2b's window has not been released." >&2
    exit 98
  fi
done
mkdir -p "$M/logs" "$M/arms"

has() { case ",$STAGES," in *,$1,*) return 0;; *) return 1;; esac; }

wait_for_gpu() {
  GPU=""
  for i in $(seq 1 "$MAXWAIT_MIN"); do
    for g in ${CANDS//,/ }; do
      ROW=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' -v gg="$g" '{gsub(/ /,"",$1)} $1+0==gg+0')
      U=$(echo "$ROW" | awk -F',' '{gsub(/ /,"",$2); print $2+0}')
      T=$(echo "$ROW" | awk -F',' '{gsub(/ /,"",$3); print $3+0}')
      if [ "$U" -le 500 ] && [ "$T" -le 5 ]; then GPU=$g; break; fi
    done
    [ -n "$GPU" ] && break
    [ $((i % 10)) -eq 1 ] && echo "waiting for a free GPU ($i min) $(date -Is)"
    sleep 60
  done
  if [ -z "$GPU" ]; then
    echo "REFUSING: no GPU became free within ${MAXWAIT_MIN} min" >&2; return 97
  fi
  echo "=== candidate GPU $GPU at $(date -Is); handing to the 3-sample guard ==="
}

run_guarded() {  # re-probes and claims the lock every time; fails closed (97)
  wait_for_gpu || return 97
  bash "$M/gpu_guard_m3i8.sh" "$GPU" -- "$@"
}

install_arm() {
  cp "$M/arms/$1/mpk/persistent_kernel.py" \
     "$REPO/python/mirage/mpk/persistent_kernel.py"
  cp "$M/arms/$1/mpk/models/qwen3_5/builder.py" \
     "$REPO/python/mirage/mpk/models/qwen3_5/builder.py"
  find "$REPO/python" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
  echo "--- installed arm $1"
}

# ---------------------------------------------------------------- stage 0
# The one structural difference from M3-I2b: M3-I8 changes C++
# (src/kernel/task_register.cc) and a task header, so the clone must be rebuilt
# ONCE. After that every arm is pure Python again -- `base` is the same binary
# with MOE_GATE_PADDING_ROWS = False.
if has 0; then
  echo "########## STAGE 0 rebuild + codegen identity $(date -Is)"
  ( cd "$REPO" && $PY -m pip install --no-build-isolation -e . ) \
      > "$M/logs/rebuild.log" 2>&1
  RC=$?; echo "rebuild rc=$RC"; tail -20 "$M/logs/rebuild.log"
  [ "$RC" -ne 0 ] && { echo "REBUILD FAILED" >&2; exit "$RC"; }

  # `base` must emit the pre-M3-I8 router call. M3-I1's own compiled kernel is
  # the reference; the ONLY line that may differ is the quantize template args
  # (that is M3-I2b's change, already in the tree).
  install_arm base
  run_guarded env HF_HOME=$HOME/mpk-qwen35/hf PYTHONUNBUFFERED=1 \
    $PY -u "$HOME/mpk-qwen35/m3i1/opt/profile_wave.py" --batch-size 1 \
    --prompt-ids p06-poem --out-dir "$M/noprof_base" \
    --kernel-dir "$M/kernel_base_bs1_noprof" --rep 0 --no-profiler \
    > "$M/logs/base_codegen.log" 2>&1
  echo "base build rc=$?"
  for K in "$M/kernel_base_bs1_noprof" "$HOME/mpk-qwen35/m3i1/kernel_bs1_noprof"; do
    echo "--- $K"
    grep -A11 "topk_softmax_task_impl" "$K/test_rank0.cu" | sort -u \
      | tee "$M/logs/router_call_$(basename "$K").txt"
  done
  diff "$M/logs/router_call_kernel_base_bs1_noprof.txt" \
       "$M/logs/router_call_kernel_bs1_noprof.txt" \
    && echo "CODEGEN IDENTITY OK: base emits the pre-M3-I8 router call" \
    || { echo "CODEGEN IDENTITY FAILED: base is not a baseline" >&2; exit 92; }
fi

# ---------------------------------------------------------------- stage 1
if has 1; then
  echo "########## STAGE 1 MECHANISM: activated-expert mask $(date -Is)"
  for ARM in base v1; do
    install_arm "$ARM"
    for BS in 1 2 4 8 16; do
      run_guarded env HF_HOME=$HOME/mpk-qwen35/hf PYTHONUNBUFFERED=1 \
        $PY -u "$REPO/demo/qwen3_5/accept/opt/m3i8/mask_probe.py" \
        --batch-size "$BS" --out "$M/masks/mask_${ARM}_bs${BS}.json" \
        --kernel-dir "$M/kernel_${ARM}_mask_bs$BS" \
        >> "$M/logs/stage1_${ARM}.log" 2>&1
      echo "--- mask arm=$ARM bs=$BS rc=$?"
    done
  done
  grep -hE "\"activated_mean\"|\"hard_cap\"|\"cap_respected\"|\"rows_marked_mean\"" \
       "$M/logs/stage1_base.log" "$M/logs/stage1_v1.log" | tail -40
fi

# ---------------------------------------------------------------- stage 2
if has 2; then
  echo "########## STAGE 2 AC-3 gate (v1) $(date -Is)"
  install_arm v1
  run_guarded bash "$M/ac3_m3i8.sh" v1 > "$M/logs/stage2.log" 2>&1
  echo "########## ac3 v1 rc=$?"
  grep -E "^#####|identical|counts|CHANGED|cap_respected|per-token latency" \
       "$M/logs/stage2.log" | tail -40
fi

# ---------------------------------------------------------------- stage 3
if has 3; then
  echo "########## STAGE 3 perf bs8 then bs1 $(date -Is)"
  for BS in 8 1; do
    for ARM in base v1; do
      BSLIST=$BS run_guarded bash "$M/run_m3i8.sh" "$ARM" all \
        >> "$M/logs/stage3.log" 2>&1
      echo "--- stage3 bs=$BS arm=$ARM rc=$? $(date -Is)"
    done
  done
  $PY "$M/analyze_m3i8.py" "$M" base v1 > "$M/logs/analyze_stage3.log" 2>&1
  head -40 "$M/logs/analyze_stage3.log"
fi

# ---------------------------------------------------------------- stage 4
if has 4; then
  echo "########## STAGE 4 perf bs 2,4,16 $(date -Is)"
  for BS in 2 4 16; do
    for ARM in base v1; do
      BSLIST=$BS run_guarded bash "$M/run_m3i8.sh" "$ARM" all \
        >> "$M/logs/stage4.log" 2>&1
      echo "--- stage4 bs=$BS arm=$ARM rc=$? $(date -Is)"
    done
  done
fi

# ---------------------------------------------------------------- stage 5
if has 5; then
  echo "########## STAGE 5 v2a (moe_n_splits=4) $(date -Is)"
  for BS in 1 8; do
    BSLIST=$BS run_guarded bash "$M/run_m3i8.sh" v2a all \
      >> "$M/logs/stage5.log" 2>&1
    echo "--- stage5 perf bs=$BS v2a rc=$?"
  done
  install_arm v2a
  run_guarded bash "$M/ac3_m3i8.sh" v2a > "$M/logs/stage5_ac3.log" 2>&1
  echo "########## ac3 v2a rc=$?"
  grep -E "^#####|identical|counts|CHANGED" "$M/logs/stage5_ac3.log" | tail -30
fi

# ---------------------------------------------------------------- stage 6
if has 6; then
  echo "########## STAGE 6 v2b (moe_n_splits=8) $(date -Is)"
  for BS in 1 8; do
    BSLIST=$BS run_guarded bash "$M/run_m3i8.sh" v2b all \
      >> "$M/logs/stage6.log" 2>&1
    echo "--- stage6 perf bs=$BS v2b rc=$?"
  done
fi

# ---------------------------------------------------------------- stage 7
if has 7; then
  echo "########## STAGE 7 analysis $(date -Is)"
  $PY "$M/analyze_m3i8.py" "$M" base v1 v2a v2b > "$M/logs/analyze_final.log" 2>&1
  cat "$M/logs/analyze_final.log"
fi

echo "M3I8_PLAN_DONE=$(date -Is)"
