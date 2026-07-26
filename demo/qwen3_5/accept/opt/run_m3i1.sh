#!/usr/bin/env bash
# M3-I1 capture driver (runs ON the B200 box, under the GPU guard).
#
#   bash gpu_guard_m3i1.sh 6,0,1 -- bash run_m3i1.sh all
#
# One wave per PROCESS at every batch size (HAZARD-WAVE-RESET: an in-process
# second wave deadlocks at bs>=4).  The AC-3 geometry is pinned verbatim:
# msl=132, mbt=16, page_size=256, max_new_tokens=64, the reference prompts.
# Per batch size the prompt set is the first `bs` prompts in ascending length,
# i.e. exactly wave 0 of a full AC-3 sweep; bs16 takes all ten and the adapter
# pads to 16 slots by repeating, as it does in the committed M2 run.
#
# Kernels are compiled once per (batch size, profiled?) and reused for every
# repetition.  Profiled and unprofiled kernels are DIFFERENT binaries
# (-DMPK_ENABLE_PROFILING + -DMPK_PROFILER_BUFFER_ENTRIES), so the unprofiled
# control gets its own dirs, copied from the M2-I9 e51cb86 build so the M2
# evidence dirs are never written to.
set -uo pipefail

M=$HOME/mpk-qwen35/m3i1
OPT=$M/opt
MIRAGE=$HOME/mpk-qwen35/mirage
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export MPK_ACCEPT_DIR=$MIRAGE/demo/qwen3_5/accept
export PYTHONUNBUFFERED=1
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
SLOTS=${SLOTS:-48000000}
REPS=${REPS:-3}
MODE=${1:-all}

mkdir -p "$M/prof" "$M/noprof" "$M/tables" "$M/logs"

AVAIL=$(df -BG --output=avail /raid | tail -1 | tr -dc '0-9')
echo "df /raid avail=${AVAIL}G"
if [ "${AVAIL:-0}" -lt 10 ]; then
  echo "REFUSING: /raid headroom ${AVAIL}G < 10G" >&2; exit 96
fi

ids_for_bs() {
  case "$1" in
    1)  echo "p06-poem" ;;
    2)  echo "p06-poem,p01-history" ;;
    4)  echo "p06-poem,p01-history,p04-chinese,p09-translate" ;;
    8)  echo "p06-poem,p01-history,p04-chinese,p09-translate,p07-format,p05-cuda,p08-science,p10-logic" ;;
    16) echo "p06-poem,p01-history,p04-chinese,p09-translate,p07-format,p05-cuda,p08-science,p10-logic,p03-python,p02-math" ;;
    *)  echo "BAD" ;;
  esac
}

run_profiled() {
  local BS=$1 REP=$2
  local KDIR=$M/kernel_bs${BS}_prof
  local RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  local RAW=""; [ "$REP" -eq 0 ] && RAW="--save-raw"
  echo "===== PROFILED bs=$BS rep=$REP ${RK:-(compile)} ====="
  $PY -u "$OPT/profile_wave.py" --batch-size "$BS" \
      --prompt-ids "$(ids_for_bs "$BS")" --out-dir "$M/prof" \
      --kernel-dir "$KDIR" --rep "$REP" --slots "$SLOTS" $RK $RAW \
      > "$M/logs/prof_bs${BS}_rep${REP}.log" 2>&1
  echo "rc=$? $(grep -h 'profiler:\|wall=' "$M/logs/prof_bs${BS}_rep${REP}.log" | tail -2)"
}

run_noprof() {
  local BS=$1 REP=$2
  local KDIR=$M/kernel_bs${BS}_noprof
  if [ ! -f "$KDIR/task_graph_rank0.json" ] && \
     [ -f "$HOME/mpk-qwen35/m2i9/kernel_bs${BS}/task_graph_rank0.json" ]; then
    echo "seeding $KDIR from the M2-I9 e51cb86 build"
    cp -r "$HOME/mpk-qwen35/m2i9/kernel_bs${BS}" "$KDIR"
  fi
  local RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  echo "===== UNPROFILED bs=$BS rep=$REP ${RK:-(compile)} ====="
  $PY -u "$OPT/profile_wave.py" --batch-size "$BS" \
      --prompt-ids "$(ids_for_bs "$BS")" --out-dir "$M/noprof" \
      --kernel-dir "$KDIR" --rep "$REP" --no-profiler $RK \
      > "$M/logs/noprof_bs${BS}_rep${REP}.log" 2>&1
  echo "rc=$? $(grep -h 'wall=' "$M/logs/noprof_bs${BS}_rep${REP}.log" | tail -1)"
}

parse_bs() {
  local BS=$1
  echo "===== PARSE bs=$BS ====="
  $PY -u "$OPT/parse_profile.py" \
      --raw "$M/prof/raw_bs${BS}_rep0.npz" \
      --meta "$M/prof/meta_bs${BS}_rep0.json" \
      --names "$M/prof/task_names.json" \
      --out-prefix "$M/tables/bs${BS}" \
      > "$M/logs/parse_bs${BS}.log" 2>&1
  echo "rc=$?"; tail -5 "$M/logs/parse_bs${BS}.log"
}

for BS in ${BSLIST:-1 2 4 8 16}; do
  case "$MODE" in
    all|prof)
      for R in $(seq 0 $((REPS - 1))); do run_profiled "$BS" "$R"; done ;;
  esac
  case "$MODE" in
    all|noprof)
      for R in $(seq 0 $((REPS - 1))); do run_noprof "$BS" "$R"; done ;;
  esac
  case "$MODE" in
    all|parse) parse_bs "$BS" ;;
  esac
done
echo "M3I1_DRIVER_EXIT=0"
