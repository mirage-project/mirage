#!/usr/bin/env bash
# M4-I4 -- the admission-cap A/B that LANDS the policy, run under ONE GPU claim
# in an isolated clone (~/mpk-qwen35/mirage-m4i4) with its OWN freshly built
# extension (STALE-EXTENSION TRAP, .memory/main/b200-env.md).
#
# WHAT IT MEASURES, AND WHY THIS SHAPE
# ------------------------------------
# M3-I7 measured the cap at bs4/bs8/bs16 but assembled the bs4/bs8 A/B from TWO
# windows (uncapped from its perfM sweep, capped from a later capsweep probe), so
# the two arms never shared a window. This campaign interleaves the arms inside
# one claim at every batch size, and runs bs1/bs2 as well -- the policy excludes
# them and that exclusion needs its own evidence, not an argument.
#
# PHASES (select with PHASES="geomA geomM ac3gate"):
#
#   geomA    AC-3 geometry (10 pinned reference prompts, msl=132, 64 new tokens,
#            mbt=16). Every run also produces the AC-3 token dump, so the
#            correctness evidence and the wave-wall A/B come from the same runs.
#            arms: none (pre-policy uncapped) x auto (forced cap) at all 5 bs.
#
#   geomM    Pinned 256/1024 benchmark geometry -- the geometry the binding vLLM
#            table was captured at. Two configs per (bs, arm):
#              full : msl=1280, 1024 new tokens
#              pre  : msl=259,  2 new tokens (same prompts, prefill only)
#            decode tok/s = bs*(D_full - D_pre)/(wall_full - wall_pre), which is
#            vLLM's own tokens-over-decode-window definition (bench-protocol.md
#            "Decode-throughput measurement"). Prompts come from --reference via
#            make_matched_reference.py, the ONLY prompt source the adapter honours.
#
#   ac3gate  M4-I0's fingerprint-scored COLD-compile AC-3 stability gate, run at
#            the SHIPPED policy (its default is now "policy") so the landed
#            configuration is certified by the pinned machinery, not just by the
#            warm reps in geomA.
#
# DISCIPLINE
#   * The cap is a COMPILE-TIME define. Every kernel dir is keyed by its cap
#     value AND carries a cap.txt; kdir_for() refuses a dir whose recorded value
#     differs. Sharing a kernel dir between arms is the trap that made M3-I9
#     under-report this win 5x and M3-I7's first pass report the arms identical.
#   * >=3 reps, medians + every per-rep value, arms alternated inside one window.
#   * Drain gate before every run + gpu_before/gpu_after recorded per tag, with
#     the device index taken from the campaign's verified claim (probe_dev below
#     re-derives it from a real CUDA context at every phase start, so a
#     candidate-list mislabel like M3-I7's cannot recur).
#   * Evidence is pulled into the repo per cell, not at the end (I6a c3).
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=$HOME/mpk-qwen35/mirage-m4i4
ACC=$T/demo/qwen3_5/accept
OPT=$ACC/opt
export MPK_ACCEPT_DIR=$ACC
export PYTHONPATH=$T/python
PY=$HOME/mpk-qwen35/venv-rm/bin/python
M=$HOME/mpk-qwen35/m4i4
K=$M/kernels
PHASES="${PHASES:-geomA geomM ac3gate}"
REPS="${REPS:-3}"
BSS="${BSS:-1 2 4 8 16}"
ARMS="${ARMS:-none auto}"
mkdir -p "$M/logs" "$K" "$M/geomA" "$M/geomM" "$M/audit" "$M/prompts"

FLOOR="${MPK_M4I4_GPU_FLOOR:-0}"
DRAIN_LIMIT=$((FLOOR + 400))

echo "########## M4-I4 gpu=$GPU phases='$PHASES' reps=$REPS $(date -Is) ##########"
echo "tree: $(git -C "$T" rev-parse HEAD)"
git -C "$T" status --short | head -5
md5sum "$T"/python/mirage/core.cpython-*.so
$PY "$ACC/admission_policy.py"
echo "drain gate: foreign floor ${FLOOR}MiB, limit ${DRAIN_LIMIT}MiB"
MIN_RAID_G="${MIN_RAID_G:-6}"
AVAIL=$(df -BG --output=avail /raid | tail -1 | tr -dc '0-9')
echo "df /raid avail=${AVAIL}G (refuse below ${MIN_RAID_G}G)"
[ "${AVAIL:-0}" -lt "$MIN_RAID_G" ] && { echo "REFUSING: /raid headroom low" >&2; exit 96; }

# ---------------------------------------------------------------- helpers ---
probe_dev () {   # re-derive the physical device from a REAL CUDA context
  $PY - <<'EOF'
import os, subprocess, torch
torch.cuda.init()
uuid = None
try:
    uuid = torch.cuda.get_device_properties(0).uuid
    uuid = f"GPU-{uuid}" if not str(uuid).startswith("GPU-") else str(uuid)
except Exception:
    pass
rows = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name",
                       "--format=csv,noheader"], capture_output=True,
                      text=True).stdout.strip().splitlines()
phys = None
for r in rows:
    idx, u, name = [x.strip() for x in r.split(",", 2)]
    if uuid and str(uuid) in u:
        phys = idx
print(f"M4I4_DEV claim={os.environ.get('CUDA_VISIBLE_DEVICES')} "
      f"context_uuid={uuid} context_phys={phys}")
EOF
}

drain () {   # wait until OUR previous run's memory is gone (M3-I6a mechanism)
  local i used
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
           | awk -F',' -v g="$GPU" '{gsub(/ /,"",$1)} $1+0==g+0 {gsub(/ /,"",$2); print $2+0}')
    [ "${used:-999999}" -le "$DRAIN_LIMIT" ] && { [ "$i" -gt 1 ] && echo "    drained (${used}MiB after ${i} checks)"; return 0; }
    sleep 5
  done
  echo "    WARNING: device $GPU still at ${used}MiB (limit ${DRAIN_LIMIT}MiB) after 300s"
  return 1
}

audit () {   # device state a run is about to start on, keyed by tag
  local TAG="$1" WHEN="$2"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    | awk -F',' -v g="$GPU" -v t="$TAG" -v w="$WHEN" \
        '{gsub(/ /,"",$1)} $1+0==g+0 {print t" "w" gpu="$1" used="$2" util="$3}' \
    >> "$M/audit/gpu_audit.txt"
}

# kdir_for <geom> <bs> <arm> -> kernel dir whose identity INCLUDES the cap value.
# Refuses to reuse a dir compiled with a different cap (the compile-knob trap).
kdir_for () {
  local GEOM=$1 BS=$2 ARM=$3
  local VAL
  VAL=$($PY -c "
import sys; sys.path.insert(0, '$ACC')
import admission_policy as ap
v = ap.resolve_int('$ARM', 16, $BS)
print('off' if v is None else v)")
  local D="$K/${GEOM}_bs${BS}_cap${VAL}"
  mkdir -p "$D"
  if [ -f "$D/cap.txt" ]; then
    local SEEN; SEEN=$(cat "$D/cap.txt")
    if [ "$SEEN" != "$VAL" ]; then
      echo "REFUSING: kernel dir $D was compiled with cap=$SEEN, this run wants $VAL" >&2
      exit 95
    fi
  else
    echo "$VAL" > "$D/cap.txt"
  fi
  echo "$D"
}

# engine_run <tag> <outdir> <kdir> <bs> <msl> <mnt> <arm> [extra...]
engine_run () {
  local TAG=$1 OD=$2 KDIR=$3 BS=$4 MSL=$5 MNT=$6 ARM=$7; shift 7
  local RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  mkdir -p "$OD"
  drain; audit "$TAG" before
  local T0=$SECONDS
  ( cd "$ACC" && timeout 5400 $PY -u mpk_engine_run.py \
      --batch-size "$BS" --max-seq-length "$MSL" --max-new-tokens "$MNT" \
      --mbt 16 --page-size 256 --out-dir "$OD" --kernel-dir "$KDIR" $RK \
      --per-request-token-cap "$ARM" "$@" \
  ) > "$M/logs/${TAG}.log" 2>&1
  local RC=$?
  audit "$TAG" after
  echo "M4I4_RUN tag=$TAG rc=$RC arm=$ARM bs=$BS msl=$MSL kdir=$(basename "$KDIR")" \
       "compiled=$([ -n "$RK" ] && echo reuse || echo cold) secs=$((SECONDS-T0))" \
       "gpu=$GPU" | tee -a "$M/logs/run_index.txt"
  grep -h 'admission cap:' "$M/logs/${TAG}.log" | tail -1 | sed 's/^/    /'
  grep -h 'wave=' "$M/logs/${TAG}.log" | tail -1 | sed 's/^/    /'
  [ "$RC" -ne 0 ] && tail -6 "$M/logs/${TAG}.log" | sed 's/^/    !! /'
  return 0
}

# ============================================================= PHASE geomA ===
case " $PHASES " in *" geomA "*)
  echo; echo "########## PHASE geomA: AC-3 geometry A/B $(date -Is) ##########"
  probe_dev | tee -a "$M/audit/dev_probe.txt"
  for R in $(seq 0 $((REPS-1))); do
    for BS in $BSS; do
      for ARM in $ARMS; do
        KD=$(kdir_for A "$BS" "$ARM") || exit 95
        engine_run "A_bs${BS}_${ARM}_r${R}" "$M/geomA/$ARM/rep$R" "$KD" \
                   "$BS" 132 64 "$ARM"
      done
    done
  done
;; esac

# ============================================================= PHASE geomM ===
case " $PHASES " in *" geomM "*)
  echo; echo "########## PHASE geomM: pinned 256/1024 A/B $(date -Is) ##########"
  probe_dev | tee -a "$M/audit/dev_probe.txt"
  for BS in $BSS; do
    for R in $(seq 0 $((REPS-1))); do
      P=$M/prompts/synthref_bs${BS}_rep${R}.json
      [ -f "$P" ] || $PY "$OPT/m3i7/scripts/make_matched_reference.py" \
          --batch-size "$BS" --rep "$R" --out "$P" | sed 's/^/  /'
    done
  done
  for R in $(seq 0 $((REPS-1))); do
    for BS in $BSS; do
      P=$M/prompts/synthref_bs${BS}_rep${R}.json
      for ARM in $ARMS; do
        KF=$(kdir_for Mfull "$BS" "$ARM") || exit 95
        KP=$(kdir_for Mpre  "$BS" "$ARM") || exit 95
        engine_run "Mfull_bs${BS}_${ARM}_r${R}" "$M/geomM/$ARM/full" "$KF" \
                   "$BS" 1280 1024 "$ARM" --reference "$P" \
                   --dump-name "bs${BS}_rep${R}.json"
        engine_run "Mpre_bs${BS}_${ARM}_r${R}" "$M/geomM/$ARM/pre" "$KP" \
                   "$BS" 259 2 "$ARM" --reference "$P" \
                   --dump-name "bs${BS}_rep${R}.json"
      done
    done
  done
;; esac

# ============================================================ PHASE ac3gate ===
case " $PHASES " in *" ac3gate "*)
  echo; echo "########## PHASE ac3gate: cold fingerprint gate at the SHIPPED policy $(date -Is) ##########"
  probe_dev | tee -a "$M/audit/dev_probe.txt"
  drain
  MPK_I7_GPU_FLOOR="$FLOOR" PY="$PY" \
    bash "$ACC/harness/gate_ac3_stable.sh" --out "$M/ac3gate" \
         --reps "${GATE_REPS:-3}" --batch-sizes "$(echo $BSS | tr ' ' ',')" \
         --kernel-root "$M/kernels_cold" \
    > "$M/logs/ac3gate.log" 2>&1
  echo "ac3gate rc=$? $(date -Is)"
  tail -25 "$M/logs/ac3gate.log"
;; esac

echo; echo "########## M4I4_DONE phases='$PHASES' $(date -Is) ##########"
df -h /raid | tail -1
