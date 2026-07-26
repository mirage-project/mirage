#!/usr/bin/env bash
# M3-I2b full plan. Waits for an EXCLUSIVE GPU, then runs the stages in value
# order so that a short window still yields a usable, gate-complete result.
#
#   nohup bash ~/mpk-qwen35/m3i2b/plan_m3i2b.sh > ~/mpk-qwen35/m3i2b/logs/plan.log 2>&1 &
#
# 1  oracle: quantize row_partition 2-D + 3-D byte gate            (~5 min)
# 2  perf bs1, base then v1, 3 prof + 3 noprof reps               (~12 min)
# 3  AC-3 full sweep on v1 + per-case byte diff + Qwen3-8B CI     (~25 min)
# 4  perf bs 2,4,8,16, base then v1 per bs (interleaved)          (~45 min)
# 5  perf bs1 v2 (widen gdn_conv + moe_combine) + AC-3 on v2      (~30 min)
# 6  analysis
set -uo pipefail
M=$HOME/mpk-qwen35/m3i2b
CANDS=${CANDS:-6,0,1,2,3,4,5,7}
MAXWAIT_MIN=${MAXWAIT_MIN:-600}
STAGES=${STAGES:-1,2,3,4,5,6}
export PATH=/usr/local/cuda-12.8/bin:$PATH
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
cd "$M"

install_arm() {
  cp "$M/arms/$1/mpk/persistent_kernel.py" \
     "$HOME/mpk-qwen35/mirage/python/mirage/mpk/persistent_kernel.py"
  cp "$M/arms/$1/mpk/models/qwen3_5/builder.py" \
     "$HOME/mpk-qwen35/mirage/python/mirage/mpk/models/qwen3_5/builder.py"
  find "$HOME/mpk-qwen35/mirage/python" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
  echo "--- installed arm $1"
}

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
  bash "$M/gpu_guard_m3i2b.sh" "$GPU" -- "$@"
}

has() { case ",$STAGES," in *,$1,*) return 0;; *) return 1;; esac; }

# ---------------------------------------------------------------- stage 1
if has 1; then
  echo "########## STAGE 1 oracle (v1) $(date -Is)"
  install_arm v1
  run_guarded env HF_HOME=$HOME/mpk-qwen35/hf PYTHONUNBUFFERED=1 \
    $PY -u "$HOME/mpk-qwen35/mirage/tests/runtime_python/blackwell/sm100_linear_fp8_blockscale/test_linear_fp8_blockscale_testmode.py" \
    > "$M/logs/oracle_v1.log" 2>&1
  RC=$?
  echo "########## oracle rc=$RC"; tail -20 "$M/logs/oracle_v1.log"
  if [ "$RC" -ne 0 ]; then
    echo "ORACLE FAILED -- stopping. The lever is void if the bytes moved." >&2
    exit "$RC"
  fi
fi

# ---------------------------------------------------------------- stage 2
if has 2; then
  echo "########## STAGE 2 perf bs1 $(date -Is)"
  for ARM in base v1; do
    BSLIST=1 run_guarded bash "$M/run_m3i2b.sh" "$ARM" all >> "$M/logs/stage2.log" 2>&1
    echo "--- stage2 arm=$ARM rc=$?"
  done
  $PY "$M/analyze_m3i2b.py" "$M" base v1 > "$M/logs/analyze_stage2.log" 2>&1
  head -30 "$M/logs/analyze_stage2.log"
fi

# ---------------------------------------------------------------- stage 3
if has 3; then
  echo "########## STAGE 3 AC-3 gate (v1) $(date -Is)"
  install_arm v1
  run_guarded bash "$M/ac3_m3i2b.sh" v1 > "$M/logs/stage3.log" 2>&1
  echo "########## ac3 v1 rc=$?"
  grep -E "^#####|identical|counts|CHANGED|per-token latency" "$M/logs/stage3.log" | tail -40
fi

# ---------------------------------------------------------------- stage 4
if has 4; then
  echo "########## STAGE 4 perf bs 2,4,8,16 $(date -Is)"
  for BS in 2 4 8 16; do
    for ARM in base v1; do
      BSLIST=$BS run_guarded bash "$M/run_m3i2b.sh" "$ARM" all >> "$M/logs/stage4.log" 2>&1
      echo "--- stage4 bs=$BS arm=$ARM rc=$? $(date -Is)"
    done
  done
fi

# ---------------------------------------------------------------- stage 5
if has 5; then
  echo "########## STAGE 5 v2 (widen gdn_conv + moe_combine) $(date -Is)"
  BSLIST=1 run_guarded bash "$M/run_m3i2b.sh" v2 all >> "$M/logs/stage5.log" 2>&1
  echo "--- stage5 perf bs1 v2 rc=$?"
  BSLIST=16 run_guarded bash "$M/run_m3i2b.sh" v2 all >> "$M/logs/stage5.log" 2>&1
  echo "--- stage5 perf bs16 v2 rc=$?"
  install_arm v2
  run_guarded bash "$M/ac3_m3i2b.sh" v2 > "$M/logs/stage5_ac3.log" 2>&1
  echo "########## ac3 v2 rc=$?"
  grep -E "^#####|identical|counts|CHANGED|per-token latency" "$M/logs/stage5_ac3.log" | tail -40
fi

# ---------------------------------------------------------------- stage 6
if has 6; then
  echo "########## STAGE 6 analysis $(date -Is)"
  $PY "$M/analyze_m3i2b.py" "$M" base v1 v2 > "$M/logs/analyze_final.log" 2>&1
  cat "$M/logs/analyze_final.log"
fi

echo "M3I2B_PLAN_DONE=$(date -Is)"
