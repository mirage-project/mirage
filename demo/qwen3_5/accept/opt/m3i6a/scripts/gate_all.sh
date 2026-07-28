#!/usr/bin/env bash
# M3-I6a gates, in the order the issue pins them, under ONE GPU claim:
#   1. attention unit / oracle / test-mode instruments   (gate_unit.sh)
#   2. FULL AC-3 sweep at all five batch sizes + per-case byte diff vs the
#      committed results/dumps_final
#   3. per-bs e2e A/B, 3 reps, median, at BOTH geometries
#        geometry A = AC-3 geometry   (msl=132, the AC-3 reference prompts)
#        geometry B = matched 256/1024 (256-token synthetic prompt, msl=353,
#                     96 decode steps -- M3-I10 armA's committed basis)
#
# Arms are MPK_ATTN_Q_PASS=4 (base = the shipped value) and 2 (candidate =
# the new default).  The knob changes the generated code, so every arm gets its
# own kernel dir, and profiled/unprofiled lanes never share one (compile-time
# knob -- M3-I3's recorded trap).
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=$HOME/mpk-qwen35/mirage-i6a
ACC=$T/demo/qwen3_5/accept
OPT=$ACC/opt
export MPK_ACCEPT_DIR=$ACC
export PYTHONPATH=$T/python
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
M=$HOME/mpk-qwen35/i6a
G=$M/gates
SEED_BASE=20260725
PHASES="${PHASES:-unit ac3 perf}"
mkdir -p "$G" "$M/logs"

echo "########## M3-I6a GATES gpu=$GPU $(date -Is) ##########"
echo "tree: $(git -C "$T" log --oneline -1)"
sha256sum "$T/include/mirage/persistent_kernel/tasks/blackwell/attention_sm100.cuh" \
          "$T/python/mirage/mpk/models/qwen3_5/builder.py"
$PY -c "
import sys; sys.path.insert(0,'$T/python')
import re
src=open('$T/python/mirage/mpk/models/qwen3_5/builder.py').read()
m=re.search(r'self\.attn_q_pass = (\d+)', src)
print('builder default attn_q_pass =', m.group(1))
"
AVAIL=$(df -BG --output=avail /raid | tail -1 | tr -dc '0-9')
echo "df /raid avail=${AVAIL}G"
[ "${AVAIL:-0}" -lt 10 ] && { echo "REFUSING: /raid headroom low" >&2; exit 96; }

# ---------------------------------------------------------------- 1. unit ----
case " $PHASES " in *" unit "*)
  echo; echo "########## PHASE 1: unit / oracle / test-mode $(date -Is) ##########"
  bash "$M/gate_unit.sh" "$GPU" 2>&1 | tee "$M/logs/gate_unit.log" | tail -60
;; esac

# ----------------------------------------------------------------- 2. AC-3 ---
run_ac3_arm () {
  local QP="$1"
  local DUMPS=$M/ac3/dumps_qp${QP}
  mkdir -p "$DUMPS" "$M/ac3/logs"
  echo "##### AC-3 arm MPK_ATTN_Q_PASS=$QP $(date -Is)"
  for BS in 16 1 2 4 8; do
    local KDIR=$M/ac3/kernel_qp${QP}_bs${BS}
    echo "##### AC-3 wave bs=$BS $(date -Is)"
    ( cd "$ACC" && MPK_ATTN_Q_PASS=$QP timeout 2400 $PY -u mpk_engine_run.py \
        --batch-size "$BS" --out-dir "$DUMPS" --kernel-dir "$KDIR" \
        --max-seq-length 132 ) > "$M/ac3/logs/qp${QP}_bs${BS}.log" 2>&1
    echo "##### rc=$? bs=$BS $(date -Is)"; tail -3 "$M/ac3/logs/qp${QP}_bs${BS}.log"
    # audit the pass size actually compiled in
    grep -o "multitoken_paged_attention_sm100_task_impl<bfloat16[^(]*" \
         "$KDIR/test_rank0.cu" 2>/dev/null | head -1
  done
  echo "##### AC-3 GATE arm qp$QP $(date -Is)"
  $PY -u "$ACC/harness/run_ac3.py" --engine-dump-dir "$DUMPS" \
      --batch-sizes 1,2,4,8,16 --output-json "$M/ac3/run_report_qp${QP}.json" \
      > "$M/ac3/logs/qp${QP}_gate.log" 2>&1
  echo "##### run_ac3 rc=$?"; tail -24 "$M/ac3/logs/qp${QP}_gate.log"
  echo "##### PER-CASE BYTE DIFF vs committed results/dumps_final"
  $PY -u "$HOME/mpk-qwen35/m3i2a/bytediff.py" "$ACC/results/dumps_final" \
      "$DUMPS" 1,2,4,8,16 > "$M/ac3/bytediff_qp${QP}.json" \
      2> "$M/ac3/logs/qp${QP}_bytediff.err"
  echo "##### bytediff rc=$?"; tail -3 "$M/ac3/logs/qp${QP}_bytediff.err"
  $PY - "$M/ac3/bytediff_qp${QP}.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("identical:", d.get("identical"), " missing:", d.get("missing"))
print("counts:", json.dumps(d.get("counts")))
bad = {k: v for k, v in d.get("per_case", {}).items() if v != "identical"}
print("CHANGED:", json.dumps(bad, indent=1) if bad else "none")
EOF
}

case " $PHASES " in *" ac3 "*)
  echo; echo "########## PHASE 2: FULL AC-3 SWEEP $(date -Is) ##########"
  run_ac3_arm 2
;; esac

# ----------------------------------------------------------------- 3. perf ---
# geometry A: the AC-3 geometry -- the reference prompt ids, msl=132.
# geometry B: matched 256/1024 -- 256-token synthetic prompt, msl=353.
perf_geomA () {
  local QP="$1" BS="$2" REP="$3"
  local KDIR=$M/perf/kernel_A_qp${QP}_bs${BS}
  local TAG=A_qp${QP}_bs${BS}_rep${REP}
  local RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  ( cd "$ACC" && MPK_ATTN_Q_PASS=$QP timeout 2400 $PY -u mpk_engine_run.py \
      --batch-size "$BS" --out-dir "$M/perf/dumpsA_qp${QP}" \
      --kernel-dir "$KDIR" --max-seq-length 132 $RK \
      --dump-name "bs${BS}_rep${REP}" ) > "$M/perf/logs/${TAG}.log" 2>&1
  echo "  [$TAG] rc=$? $(grep -hoE 'wall[_= ]*[0-9.]+ *(ms)?' "$M/perf/logs/${TAG}.log" | tail -2 | tr '\n' ' ')"
}
perf_geomB () {
  local QP="$1" BS="$2" REP="$3"
  local KDIR=$M/perf/kernel_B_qp${QP}_bs${BS}_noprof
  local TAG=B_qp${QP}_bs${BS}_rep${REP}
  local SEED=$((SEED_BASE + BS*1000 + REP))
  local RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  MPK_ATTN_Q_PASS=$QP timeout 2400 $PY -u "$OPT/profile_wave.py" \
      --batch-size "$BS" --max-seq-length 353 --max-new-tokens 96 --mbt 16 \
      --page-size 256 --synthetic-prompt-len 256 --synthetic-seed "$SEED" \
      --out-dir "$M/perf/noprofB_qp${QP}" --kernel-dir "$KDIR" \
      --rep "$REP" --no-profiler $RK > "$M/perf/logs/${TAG}.log" 2>&1
  echo "  [$TAG] rc=$? $(grep -h 'wall=' "$M/perf/logs/${TAG}.log" | tail -1)"
  local f=$M/perf/noprofB_qp${QP}/meta_bs${BS}_rep${REP}.json
  [ -f "$f" ] && mv "$f" "$M/perf/noprofB_qp${QP}/meta_bs${BS}_rep${REP}_qp${QP}.json"
}

case " $PHASES " in *" perf "*)
  echo; echo "########## PHASE 3: e2e A/B, 3 reps, both geometries $(date -Is) ##########"
  mkdir -p "$M/perf/logs"
  for QP in 4 2; do
    mkdir -p "$M/perf/dumpsA_qp${QP}" "$M/perf/noprofB_qp${QP}"
  done
  echo "--- geometry A (AC-3: reference prompts, msl=132) ---"
  for BS in 1 2 4 8 16; do for QP in 4 2; do for REP in 0 1 2; do
    perf_geomA "$QP" "$BS" "$REP"
  done; done; done
  echo "--- geometry B (matched 256/1024: synth 256, msl=353, 96 decode steps) ---"
  for BS in 1 2 4 8 16; do for QP in 4 2; do for REP in 0 1 2; do
    perf_geomB "$QP" "$BS" "$REP"
  done; done; done
;; esac

echo; echo "########## GATE_ALL_DONE $(date -Is) ##########"
