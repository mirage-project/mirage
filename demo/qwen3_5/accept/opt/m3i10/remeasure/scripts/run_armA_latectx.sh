#!/usr/bin/env bash
# F2 closure capture: MPK decode window re-centered into context ~556-896 to
# match the vLLM reference table's own sampled context, isolating the
# attention finding from the arm-A-vs-vLLM context mismatch (arm A sampled
# 257-352; vLLM sampled 556-896).
#
# Derivation (cheapest correct form, per coordinator sign-off): keep prompt_len
# = 256 (same synthetic prompts, same seed formula), run D=640 total decode
# steps (max_seq_length = 256 + 640 + 1 = 897) so context spans [257, 896]
# across the wave, then in POST-PROCESSING take the final-96-iteration window
# of that decode run (context ~[801,896]) -- inside the vLLM window's own
# span, cheapest way to reach it without profiling the vLLM-infeasible full
# 1024-decode MPK never needed either. Profiled only, 1 rep/bs: this is a
# targeted closure measurement (attention's context sensitivity + a spot check
# that GDN/GEMM stay flat), not the primary statistical deliverable -- arm A's
# 3-rep reps already showed <0.2% wave-to-wave spread, and event counts are
# provably seed-independent (M3-I10 tier-1 evidence), so 1 rep suffices here.
set -uo pipefail

M=$HOME/mpk-qwen35/m3i10-remeasure
MIRAGE=$HOME/mpk-qwen35/mirage-rm
OPT=$MIRAGE/demo/qwen3_5/accept/opt
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export MPK_ACCEPT_DIR=$MIRAGE/demo/qwen3_5/accept
export PYTHONUNBUFFERED=1
PY=$HOME/mpk-qwen35/venv-rm/bin/python
SEED_BASE=20260725
MSL=897

mkdir -p "$M/prof_Alate" "$M/logs"

AVAIL=$(df -BG --output=avail /raid | tail -1 | tr -dc '0-9')
echo "df /raid avail=${AVAIL}G"
if [ "${AVAIL:-0}" -lt 10 ]; then
  echo "REFUSING: /raid headroom ${AVAIL}G < 10G" >&2; exit 96
fi

for BS in ${BSLIST:-1 8 16}; do
  SEED=$((SEED_BASE + BS*1000 + 0))
  KDIR=$M/kernel_Alate_bs${BS}_prof
  RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  echo "===== [Alate] PROFILED bs=$BS msl=$MSL rep=0 ${RK:-(compile)} $(date -Is) ====="
  $PY -u "$OPT/profile_wave.py" --batch-size "$BS" \
      --max-seq-length "$MSL" --max-new-tokens 96 --mbt 16 --page-size 256 \
      --synthetic-prompt-len 256 --synthetic-seed "$SEED" \
      --out-dir "$M/prof_Alate" --kernel-dir "$KDIR" \
      --rep 0 --slots 200000000 --save-raw $RK \
      > "$M/logs/Alate_prof_bs${BS}_rep0.log" 2>&1
  echo "rc=$? $(grep -h 'profiler:\|wall=' "$M/logs/Alate_prof_bs${BS}_rep0.log" | tail -2)"
done
echo "M3I10RM_LATECTX_EXIT=0 $(date -Is)"
