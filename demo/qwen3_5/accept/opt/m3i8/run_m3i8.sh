#!/usr/bin/env bash
# M3-I8 A/B capture driver (runs ON the B200 box, under the GPU guard).
#
#   bash gpu_guard_m3i8.sh 6,0,1,2,3,4,5,7 -- bash run_m3i8.sh <arm> <mode>
#
# arm  = base | v1 | v2a | v2b       (source snapshot under $M/arms/<arm>/)
# mode = all | prof | noprof | parse
#
# Reuses M3-I1's capture pipeline VERBATIM (profile_wave.py / parse_profile.py /
# concurrency.py) so the before/after numbers come from the same instrument.
# One wave per PROCESS, AC-3 geometry pinned (msl=132, mbt=16, page 256, 64 new
# tokens), the same ascending-length prompt sets as I1 and I2b.
#
# ONE DIFFERENCE FROM M3-I2b: this issue also changes C++
# (`src/kernel/task_register.cc`) and a task header, so the box's mirage clone
# must be REBUILT ONCE before any arm runs (see `plan_m3i8.sh` stage 0). After
# that rebuild the arms are pure Python again -- `base` is the same binary with
# `MOE_GATE_PADDING_ROWS = False`, which regenerates the pre-M3-I8 router call
# byte-for-byte.
set -uo pipefail

M=$HOME/mpk-qwen35/m3i8
OPT=$HOME/mpk-qwen35/m3i1/opt          # I1's instrument, unchanged
MIRAGE=$HOME/mpk-qwen35/mirage
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export MPK_ACCEPT_DIR=$MIRAGE/demo/qwen3_5/accept
export PYTHONUNBUFFERED=1
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
SLOTS=${SLOTS:-48000000}
REPS=${REPS:-3}
ARM=${1:?arm}
MODE=${2:-all}

mkdir -p "$M/prof_$ARM" "$M/noprof_$ARM" "$M/tables_$ARM" "$M/logs"

AVAIL=$(df -BG --output=avail /raid | tail -1 | tr -dc '0-9')
echo "df /raid avail=${AVAIL}G"
if [ "${AVAIL:-0}" -lt 10 ]; then
  echo "REFUSING: /raid headroom ${AVAIL}G < 10G" >&2; exit 96
fi

# --- install the arm's source snapshot -------------------------------------
A=$M/arms/$ARM
[ -d "$A" ] || { echo "REFUSING: no arm snapshot at $A" >&2; exit 95; }
cp "$A/mpk/persistent_kernel.py" "$MIRAGE/python/mirage/mpk/persistent_kernel.py"
cp "$A/mpk/models/qwen3_5/builder.py" \
   "$MIRAGE/python/mirage/mpk/models/qwen3_5/builder.py"
find "$MIRAGE/python" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
echo "=== arm=$ARM installed; sha256 ==="
sha256sum "$MIRAGE/python/mirage/mpk/persistent_kernel.py" \
          "$MIRAGE/python/mirage/mpk/models/qwen3_5/builder.py" \
  | tee "$M/logs/arm_${ARM}_sha256.txt"
grep -n "^MOE_GATE_PADDING_ROWS\|self.moe_n_splits" \
     "$MIRAGE/python/mirage/mpk/models/qwen3_5/builder.py" | tee -a \
     "$M/logs/arm_${ARM}_sha256.txt"

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
  local KDIR=$M/kernel_${ARM}_bs${BS}_prof
  local RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  local RAW=""; [ "$REP" -eq 0 ] && RAW="--save-raw"
  echo "===== [$ARM] PROFILED bs=$BS rep=$REP ${RK:-(compile)} $(date -Is) ====="
  $PY -u "$OPT/profile_wave.py" --batch-size "$BS" \
      --prompt-ids "$(ids_for_bs "$BS")" --out-dir "$M/prof_$ARM" \
      --kernel-dir "$KDIR" --rep "$REP" --slots "$SLOTS" $RK $RAW \
      > "$M/logs/${ARM}_prof_bs${BS}_rep${REP}.log" 2>&1
  echo "rc=$? $(grep -h 'profiler:\|wall=' "$M/logs/${ARM}_prof_bs${BS}_rep${REP}.log" | tail -2)"
}

run_noprof() {
  local BS=$1 REP=$2
  local KDIR=$M/kernel_${ARM}_bs${BS}_noprof
  local RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  echo "===== [$ARM] UNPROFILED bs=$BS rep=$REP ${RK:-(compile)} $(date -Is) ====="
  $PY -u "$OPT/profile_wave.py" --batch-size "$BS" \
      --prompt-ids "$(ids_for_bs "$BS")" --out-dir "$M/noprof_$ARM" \
      --kernel-dir "$KDIR" --rep "$REP" --no-profiler $RK \
      > "$M/logs/${ARM}_noprof_bs${BS}_rep${REP}.log" 2>&1
  echo "rc=$? $(grep -h 'wall=' "$M/logs/${ARM}_noprof_bs${BS}_rep${REP}.log" | tail -1)"
}

parse_bs() {
  local BS=$1
  echo "===== [$ARM] PARSE bs=$BS ====="
  $PY -u "$OPT/parse_profile.py" \
      --raw "$M/prof_$ARM/raw_bs${BS}_rep0.npz" \
      --meta "$M/prof_$ARM/meta_bs${BS}_rep0.json" \
      --names "$M/prof_$ARM/task_names.json" \
      --out-prefix "$M/tables_$ARM/bs${BS}" \
      > "$M/logs/${ARM}_parse_bs${BS}.log" 2>&1
  echo "rc=$?"; tail -4 "$M/logs/${ARM}_parse_bs${BS}.log"
  $PY -u "$OPT/concurrency.py" \
      "$M/prof_$ARM/raw_bs${BS}_rep0.npz" "$M/prof_$ARM/meta_bs${BS}_rep0.json" \
      "$M/prof_$ARM/task_names.json" "$M/tables_$ARM/bs${BS}_concurrency.json" \
      > "$M/logs/${ARM}_conc_bs${BS}.log" 2>&1
  echo "conc rc=$?"
}

for BS in ${BSLIST:-1 2 4 8 16}; do
  case "$MODE" in
    all|prof) for R in $(seq 0 $((REPS - 1))); do run_profiled "$BS" "$R"; done ;;
  esac
  case "$MODE" in
    all|noprof) for R in $(seq 0 $((REPS - 1))); do run_noprof "$BS" "$R"; done ;;
  esac
  case "$MODE" in
    all|parse) parse_bs "$BS" ;;
  esac
done
echo "M3I8_DRIVER_EXIT=0 arm=$ARM mode=$MODE $(date -Is)"
