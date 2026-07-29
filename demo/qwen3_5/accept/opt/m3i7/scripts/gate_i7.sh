#!/usr/bin/env bash
# M3-I7 -- the M3 milestone integration gate, run under ONE GPU claim in an
# ISOLATED clone (~/mpk-qwen35/mirage-i7) with its OWN freshly-built extension
# (STALE-EXTENSION TRAP, .memory/main/b200-env.md).
#
# PHASES (select with PHASES="ac3 perfA perfM prof late"):
#
#   ac3    FULL AC-3 sweep at all five batch sizes on integrated HEAD +
#          per-case byte diff vs the committed results/dumps_final.  Adds
#          (a) a bs16 CAPPED arm, because the pinned benchmark policy is
#          `--per-request-token-cap auto` at bs16 (docs/qwen35/bench-protocol.md,
#          M3-I9 landing) and the perf tables are measured in that
#          configuration, so correctness must be shown for it too; and
#          (b) 3 total reps at bs2 -- M3-I11's builder is concurrently
#          attributing a bs2/p08-science difference seen on a dirty GPU, and
#          the pinned rule is >=2 same-config reps before ANY "the tokens
#          changed" claim (M3-I9b).
#
#   perfA  e2e at the AC-3 geometry (10 pinned reference prompts, msl=132,
#          64 new tokens): 3 reps x 5 bs, plus a bs16 capped arm.
#
#   perfM  e2e at the PINNED matched 256/1024 benchmark geometry -- the
#          geometry the binding vLLM table was captured at.  Two configs per bs:
#            full : msl=1280, 1024 new tokens  (the benchmark job)
#            pre  : msl=259,  2 new tokens     (same prompts, prefill only)
#          Steady-state decode throughput is the SLOPE between them,
#            decode tok/s = bs*(D_full - D_pre) / (wall_full - wall_pre),
#          which is vLLM's own definition (tokens / decode-window seconds,
#          bench-protocol.md 5) rather than a whole-wave average that silently
#          bills prefill to decode.  Prompts come from --reference (the ONLY
#          prompt source mpk_engine_run.py honours; --prompts-file is read only
#          under --verify-chat-template) via make_matched_reference.py, so these
#          really are 256-token prompts drawn with the baseline's own sampler
#          and seed -- which M3-I9 stage 7 was not (see that script's docstring).
#          bs16 runs the pinned capped policy AND an uncapped arm.
#
#   prof   PROFILED capture at M3-I10 arm A's basis (msl=353 = 256-token
#          synthetic prompt + 96 decode steps + 1) for the per-stage
#          re-derivation, 3 reps x {1,8,16}, raw npz on rep0 only, plus a
#          3-rep UNPROFILED control at the same geometry (profiler overhead).
#
#   late   PROFILED late-context capture (msl=897, decode context ~801-896),
#          1 rep x {1,8,16} -- the basis ferret_targets.json's attention row
#          must be re-derived on (M3-I6a STALE-BASIS flag).
#
# Statistical rule everywhere: warmup implicit in the compile/first rep, >=3
# reps, medians + full range reported, arms alternated inside ONE window, and a
# per-rep DRAIN GATE + gpu_before audit so a rep that starts on a dirty device
# is discarded in analysis rather than silently averaged in (M3-I6a).
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=$HOME/mpk-qwen35/mirage-i7
ACC=$T/demo/qwen3_5/accept
OPT=$ACC/opt
export MPK_ACCEPT_DIR=$ACC
export PYTHONPATH=$T/python
PY=$HOME/mpk-qwen35/venv-rm/bin/python
M=$HOME/mpk-qwen35/i7
K=$M/kernels
PHASES="${PHASES:-ac3 perfA perfM prof late}"
REPS="${REPS:-3}"
SEED_BASE_PROF=20260725     # M3-I10 arm A's profile_wave seed formula
mkdir -p "$M/logs" "$K" "$M/ac3" "$M/perf" "$M/prof" "$M/audit"

echo "########## M3-I7 GATE gpu=$GPU phases='$PHASES' $(date -Is) ##########"
echo "tree: $(git -C "$T" rev-parse HEAD)"
git -C "$T" log --oneline -1
git -C "$T" status --short | head -5
$PY -c "import sys; sys.path.insert(0,'$T/python'); import mirage, os; print('mirage:', os.path.realpath(mirage.__file__))"
md5sum "$T"/python/mirage/core.cpython-*.so
# /raid is a shared 28T pool sitting at 100 % use; the free figure swings by
# several GB from other tenants. The ac3/perfA/perfM phases write only JSON
# dumps plus ~85 MB kernel dirs, so they need little; the profiled phases write
# multi-GB npz and parse them inline (see parse_raw) to keep peak usage to one
# raw at a time. MIN_RAID_G is therefore per-invocation, not a global constant.
MIN_RAID_G="${MIN_RAID_G:-4}"
AVAIL=$(df -BG --output=avail /raid | tail -1 | tr -dc '0-9')
echo "df /raid avail=${AVAIL}G (refuse below ${MIN_RAID_G}G)"
[ "${AVAIL:-0}" -lt "$MIN_RAID_G" ] && { echo "REFUSING: /raid headroom low" >&2; exit 96; }

# ---------------------------------------------------------------- helpers ---
# The drain gate exists to catch OUR OWN previous rep still tearing down -- M3-I6a
# found a 34 GB process mid-teardown producing a fake 2.1x regression. It must
# therefore measure growth above the device's foreign floor, not absolute
# occupancy: on a box where other users park idle CUDA contexts, an absolute
# 500 MiB test never clears and every rep pays the full 300 s timeout while
# still starting "dirty". FLOOR is what the guard measured at claim time.
FLOOR="${MPK_I7_GPU_FLOOR:-0}"
DRAIN_LIMIT=$((FLOOR + 400))
echo "drain gate: foreign floor ${FLOOR}MiB, limit ${DRAIN_LIMIT}MiB"

drain () {   # wait until OUR memory is gone (device back to its foreign floor)
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

# Parse a profiler raw into the per-stage tables IMMEDIATELY, so a multi-GB npz
# never has to coexist with the next one on a 100%-full filesystem. Uses the
# in-tree pipeline (parse_profile.py + concurrency.py), which since M3-I7 runs
# the corrected steady_window tie-break (see opt/schedule_sim.py).
parse_raw () {   # parse_raw <outdir> <bs> <rep> <tables-dir>
  local OD=$1 BS=$2 REP=$3 TD=$4
  local RAW=$OD/raw_bs${BS}_rep${REP}.npz
  local MET=$OD/meta_bs${BS}_rep${REP}.json
  [ -f "$RAW" ] || { echo "    (no raw for bs${BS} rep${REP}; nothing to parse)"; return 0; }
  mkdir -p "$TD"
  ( cd "$OPT" && $PY -u parse_profile.py --raw "$RAW" --meta "$MET" \
      --names "$OD/task_names.json" --out-prefix "$TD/bs${BS}" ) > "$TD/bs${BS}_parse.log" 2>&1
  echo "    parse rc=$? -> $TD/bs${BS}_attrib.json  ($(du -h "$RAW" | cut -f1) raw)"
  ( cd "$OPT" && $PY -u concurrency.py "$RAW" "$MET" "$OD/task_names.json" \
      "$TD/bs${BS}_concurrency.json" ) > "$TD/bs${BS}_conc.log" 2>&1
  echo "    conc  rc=$?"
  $PY - "$TD/bs${BS}_attrib.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
print(f"    bs{s['batch_size']} step_us={s['step_us']:.1f} window={s['steady_window']} "
      f"regime={s['steady_regime_live_prefill_decode_tokens']} "
      f"sched_model_agrees={s['schedule_model_agrees']}")
EOF
  df -h /raid | tail -1
}

audit () {   # record the device state a run is about to start on
  local TAG="$1"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    | sed "s/^/${TAG} /" >> "$M/audit/gpu_before.txt"
}

# engine_run <tag> <outdir> <kernel-dir> <bs> <msl> <mnt> [extra...]
engine_run () {
  local TAG=$1 OD=$2 KDIR=$3 BS=$4 MSL=$5 MNT=$6; shift 6
  local RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  mkdir -p "$OD"
  drain; audit "$TAG"
  ( cd "$ACC" && timeout 4200 $PY -u mpk_engine_run.py \
      --batch-size "$BS" --max-seq-length "$MSL" --max-new-tokens "$MNT" \
      --mbt 16 --page-size 256 --out-dir "$OD" --kernel-dir "$KDIR" $RK "$@" \
  ) > "$M/logs/${TAG}.log" 2>&1
  echo "  [$TAG] rc=$? $(grep -h 'wave=' "$M/logs/${TAG}.log" | tail -1)"
}

# wave_run <tag> <outdir> <kernel-dir> <bs> <msl> <rep> <slots|noprof> [extra...]
wave_run () {
  local TAG=$1 OD=$2 KDIR=$3 BS=$4 MSL=$5 REP=$6 MODE=$7; shift 7
  local RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  local SEED=$((SEED_BASE_PROF + BS*1000 + REP))
  local MODEFLAGS="--no-profiler"
  [ "$MODE" != "noprof" ] && MODEFLAGS="--slots $MODE"
  mkdir -p "$OD"
  drain; audit "$TAG"
  timeout 4200 $PY -u "$OPT/profile_wave.py" --batch-size "$BS" \
      --max-seq-length "$MSL" --max-new-tokens 96 --mbt 16 --page-size 256 \
      --synthetic-prompt-len 256 --synthetic-seed "$SEED" \
      --out-dir "$OD" --kernel-dir "$KDIR" --rep "$REP" $MODEFLAGS $RK "$@" \
      > "$M/logs/${TAG}.log" 2>&1
  echo "  [$TAG] rc=$? $(grep -h 'profiler:\|wall=' "$M/logs/${TAG}.log" | tail -2 | tr '\n' ' ')"
}

# ============================================================== PHASE ac3 ===
case " $PHASES " in *" ac3 "*)
  echo; echo "########## PHASE ac3: FULL AC-3 SWEEP at integrated HEAD $(date -Is) ##########"
  for BS in 1 2 4 8 16; do
    engine_run "ac3_bs${BS}" "$M/ac3/dumps" "$K/A_bs${BS}" "$BS" 132 64
  done
  # pinned benchmark policy arm (cap auto at bs16) -- separate dump tree so the
  # harness never sees an ambiguous tree (M3-I9b rule).
  engine_run "ac3_bs16_cap" "$M/ac3/dumps_cap16" "$K/A_bs16" 16 132 64 \
      --per-request-token-cap auto
  # bs2 determinism reps (independent data point for the I11 bs2/p08-science
  # question -- we do NOT re-root-cause it here, we measure it).
  for R in 1 2; do
    engine_run "ac3_bs2_rep${R}" "$M/ac3/dumps_bs2_rep${R}" "$K/A_bs2" 2 132 64
  done

  echo "##### AC-3 GATE (integrated HEAD, uncapped) $(date -Is)"
  $PY -u "$ACC/harness/run_ac3.py" --engine-dump-dir "$M/ac3/dumps" \
      --batch-sizes 1,2,4,8,16 --output-json "$M/ac3/run_report_head.json" \
      > "$M/logs/ac3_gate.log" 2>&1
  echo "##### run_ac3 rc=$?"; tail -30 "$M/logs/ac3_gate.log"

  for D in dumps dumps_cap16 dumps_bs2_rep1 dumps_bs2_rep2; do
    BSS=1,2,4,8,16; case "$D" in dumps_cap16) BSS=16;; dumps_bs2_rep*) BSS=2;; esac
    echo "##### PER-CASE BYTE DIFF ($D vs committed results/dumps_final)"
    $PY -u "$HOME/mpk-qwen35/m3i2a/bytediff.py" "$ACC/results/dumps_final" \
        "$M/ac3/$D" "$BSS" > "$M/ac3/bytediff_${D}.json" 2> "$M/logs/bytediff_${D}.err"
    echo "##### bytediff rc=$?"; tail -2 "$M/logs/bytediff_${D}.err"
    $PY - "$M/ac3/bytediff_${D}.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("  identical:", d.get("identical"), " missing:", d.get("missing"),
      " counts:", json.dumps(d.get("counts")))
bad = {k: v for k, v in d.get("per_case", {}).items() if v != "identical"}
print("  CHANGED:", json.dumps(bad) if bad else "none")
EOF
  done
;; esac

# ============================================================ PHASE perfA ===
case " $PHASES " in *" perfA "*)
  echo; echo "########## PHASE perfA: e2e at the AC-3 geometry $(date -Is) ##########"
  for R in $(seq 0 $((REPS-1))); do
    for BS in 1 2 4 8 16; do
      engine_run "pA_bs${BS}_rep${R}" "$M/perf/A" "$K/A_bs${BS}" "$BS" 132 64 \
          --dump-name "bs${BS}_rep${R}.json"
    done
    engine_run "pA_bs16cap_rep${R}" "$M/perf/A_cap16" "$K/A_bs16" 16 132 64 \
        --per-request-token-cap auto --dump-name "bs16_rep${R}.json"
  done
;; esac

# =========================================================== PHASE perfA1 ===
# M3-I1's EXACT AC-3-geometry shape: profile_wave.py --prompt-ids over the
# ascending-length reference subsets, msl=132, ONE wave per process. The full
# gate shape (phase perfA) runs all ten prompts in ceil(10/bs) waves and is NOT
# comparable to the committed baseline; this one is, wave wall for wave wall
# (opt/attribution.csv `wave_wall_ms_unprofiled`) and, on the profiled rep,
# steady decode step for steady decode step (`step_us` / `decode_tok_s`).
ids_for_bs () {
  case "$1" in
    1)  echo "p06-poem" ;;
    2)  echo "p06-poem,p01-history" ;;
    4)  echo "p06-poem,p01-history,p04-chinese,p09-translate" ;;
    8)  echo "p06-poem,p01-history,p04-chinese,p09-translate,p07-format,p05-cuda,p08-science,p10-logic" ;;
    16) echo "p06-poem,p01-history,p04-chinese,p09-translate,p07-format,p05-cuda,p08-science,p10-logic,p03-python,p02-math" ;;
  esac
}

# a1_run <tag> <outdir> <kdir> <bs> <rep> <slots|noprof> [extra...]
a1_run () {
  local TAG=$1 OD=$2 KDIR=$3 BS=$4 REP=$5 MODE=$6; shift 6
  local RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  local MODEFLAGS="--no-profiler"
  [ "$MODE" != "noprof" ] && MODEFLAGS="--slots $MODE"
  mkdir -p "$OD"
  drain; audit "$TAG"
  timeout 4200 $PY -u "$OPT/profile_wave.py" --batch-size "$BS" \
      --max-seq-length 132 --max-new-tokens 64 --mbt 16 --page-size 256 \
      --prompt-ids "$(ids_for_bs "$BS")" \
      --out-dir "$OD" --kernel-dir "$KDIR" --rep "$REP" $MODEFLAGS $RK "$@" \
      > "$M/logs/${TAG}.log" 2>&1
  echo "  [$TAG] rc=$? $(grep -h 'profiler:\|wall=' "$M/logs/${TAG}.log" | tail -2 | tr '\n' ' ')"
}

case " $PHASES " in *" perfA1 "*)
  echo; echo "########## PHASE perfA1: AC-3 geometry, M3-I1 shape $(date -Is) ##########"
  for BS in 1 2 4 8 16; do
    for R in $(seq 0 $((REPS-1))); do
      a1_run "pA1_bs${BS}_rep${R}" "$M/perf/A1" "$K/A1_bs${BS}_noprof" "$BS" "$R" noprof
    done
    # one PROFILED rep per bs -> the true steady decode step, parsed inline
    a1_run "pA1p_bs${BS}_rep0" "$M/perf/A1prof" "$K/A1_bs${BS}_prof" "$BS" 0 48000000 --save-raw
    parse_raw "$M/perf/A1prof" "$BS" 0 "$M/perf/A1/tables"
    mkdir -p "$M/perf/raw_stage"
    mv -f "$M/perf/A1prof/raw_bs${BS}_rep0.npz" "$M/perf/raw_stage/A1_raw_bs${BS}_rep0.npz" 2>/dev/null || true
  done
;; esac

# ============================================================ PHASE perfM ===
case " $PHASES " in *" perfM "*)
  echo; echo "########## PHASE perfM: pinned matched 256/1024 geometry $(date -Is) ##########"
  mkdir -p "$M/perf/M/prompts"
  for BS in 1 2 4 8 16; do
    for R in $(seq 0 $((REPS-1))); do
      $PY "$OPT/m3i7/scripts/make_matched_reference.py" --batch-size "$BS" \
          --rep "$R" --input-len 256 \
          --out "$M/perf/M/prompts/synthref_bs${BS}_rep${R}.json" \
          >> "$M/logs/perfM_prompts.log" 2>&1
    done
  done
  tail -3 "$M/logs/perfM_prompts.log"
  for R in $(seq 0 $((REPS-1))); do
    for BS in 1 2 4 8 16; do
      REF=$M/perf/M/prompts/synthref_bs${BS}_rep${R}.json
      CAP=(); [ "$BS" = 16 ] && CAP=(--per-request-token-cap auto)
      engine_run "pM_full_bs${BS}_rep${R}" "$M/perf/M/full" "$K/M_bs${BS}_full" \
          "$BS" 1280 1024 --reference "$REF" --dump-name "bs${BS}_rep${R}.json" "${CAP[@]}"
      engine_run "pM_pre_bs${BS}_rep${R}" "$M/perf/M/pre" "$K/M_bs${BS}_pre" \
          "$BS" 259 2 --reference "$REF" --dump-name "bs${BS}_rep${R}.json" "${CAP[@]}"
    done
    # bs16 uncapped arm (the A/B for the pinned cap policy at the benchmark geometry)
    REF=$M/perf/M/prompts/synthref_bs16_rep${R}.json
    engine_run "pM_full_bs16nocap_rep${R}" "$M/perf/M/full_nocap16" "$K/M_bs16_full" \
        16 1280 1024 --reference "$REF" --dump-name "bs16_rep${R}.json"
    engine_run "pM_pre_bs16nocap_rep${R}" "$M/perf/M/pre_nocap16" "$K/M_bs16_pre" \
        16 259 2 --reference "$REF" --dump-name "bs16_rep${R}.json"
  done
;; esac

# ============================================================ PHASE cap16 ===
# The bs16 admission-cap A/B, REDONE with per-arm kernel dirs.
#
# Defect this phase repairs (found by this gate, in this gate's own first pass):
# `--per-request-token-cap` is a COMPILE-TIME knob -- persistent_kernel.py:323
# turns it into `-DMPK_MAX_TOKENS_PER_REQUEST=<n>` on the JIT command line. The
# first pass pointed both arms at ONE kernel dir and let --reuse-kernel fire, so
# whichever arm compiled first decided what BOTH arms executed: at the AC-3
# geometry both ran uncapped, at the matched geometry both ran capped. The
# measured walls were identical to 0.05% for exactly that reason, while the
# CPU-side admission replay in the timings artifact still reported the arms as
# different (203 vs 131 predicted iterations) -- a replay of the policy, not of
# the binary that ran. This is the same compile-time-knob trap M3-I3 recorded
# and M3-I6a's gate script guarded against for MPK_ATTN_Q_PASS; it has to be
# obeyed for every compile-time knob, not just the one that bit us before.
#
# Each arm therefore gets its OWN kernel dir, and each dir's generated build
# command is grepped for the define so the arms are provably different binaries.
case " $PHASES " in *" cap16 "*)
  echo; echo "########## PHASE cap16: bs16 admission-cap A/B, per-arm kernels $(date -Is) ##########"
  mkdir -p "$M/perf/M/prompts"
  for R in $(seq 0 $((REPS-1))); do
    [ -f "$M/perf/M/prompts/synthref_bs16_rep${R}.json" ] || \
      $PY "$OPT/m3i7/scripts/make_matched_reference.py" --batch-size 16 --rep "$R" \
          --input-len 256 --out "$M/perf/M/prompts/synthref_bs16_rep${R}.json" >/dev/null 2>&1
  done
  for ARM in nocap cap; do
    CAP=(); [ "$ARM" = cap ] && CAP=(--per-request-token-cap auto)
    # AC-3 geometry: correctness + e2e
    engine_run "c16_ac3_${ARM}" "$M/cap16/ac3_${ARM}" "$K/C16_A_${ARM}" 16 132 64 "${CAP[@]}"
    for R in $(seq 0 $((REPS-1))); do
      engine_run "c16_A_${ARM}_rep${R}" "$M/cap16/A_${ARM}" "$K/C16_A_${ARM}" 16 132 64 \
          --dump-name "bs16_rep${R}.json" "${CAP[@]}"
      REF=$M/perf/M/prompts/synthref_bs16_rep${R}.json
      engine_run "c16_Mfull_${ARM}_rep${R}" "$M/cap16/Mfull_${ARM}" "$K/C16_M_${ARM}_full" \
          16 1280 1024 --reference "$REF" --dump-name "bs16_rep${R}.json" "${CAP[@]}"
      engine_run "c16_Mpre_${ARM}_rep${R}" "$M/cap16/Mpre_${ARM}" "$K/C16_M_${ARM}_pre" \
          16 259 2 --reference "$REF" --dump-name "bs16_rep${R}.json" "${CAP[@]}"
    done
  done
  echo "##### binary-identity audit: the define must be present in the cap arms only"
  for d in "$K"/C16_*; do
    n=$(grep -rlo "MPK_MAX_TOKENS_PER_REQUEST" "$d" 2>/dev/null | wc -l)
    v=$(grep -rho "MPK_MAX_TOKENS_PER_REQUEST=[0-9]*" "$d" 2>/dev/null | sort -u | tr '\n' ' ')
    echo "  $(basename "$d"): files_mentioning=$n  ${v:-<define absent>}"
  done
  echo "##### AC-3 byte diff, both arms"
  for ARM in nocap cap; do
    $PY -u "$HOME/mpk-qwen35/m3i2a/bytediff.py" "$ACC/results/dumps_final" \
        "$M/cap16/ac3_${ARM}" 16 > "$M/cap16/bytediff_${ARM}.json" 2>/dev/null
    $PY - "$M/cap16/bytediff_${ARM}.json" "$ARM" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
bad = {k: v for k, v in d.get("per_case", {}).items() if v != "identical"}
print(f"  arm={sys.argv[2]} identical={d.get('identical')} counts={json.dumps(d.get('counts'))} "
      f"CHANGED={json.dumps(bad) if bad else 'none'}")
EOF
  done
;; esac

# ============================================================= PHASE prof ===
case " $PHASES " in *" prof "*)
  echo; echo "########## PHASE prof: profiled per-stage capture, msl=353 $(date -Is) ##########"
  for BS in 1 8 16; do
    for R in $(seq 0 $((REPS-1))); do
      RAW=(); [ "$R" = 0 ] && RAW=(--save-raw)
      wave_run "prof_bs${BS}_rep${R}" "$M/prof/prof_A" "$K/P_bs${BS}_prof" \
          "$BS" 353 "$R" 96000000 "${RAW[@]}"
    done
    parse_raw "$M/prof/prof_A" "$BS" 0 "$M/prof/armA/tables"
    for R in $(seq 0 $((REPS-1))); do
      wave_run "noprof_bs${BS}_rep${R}" "$M/prof/noprof_A" "$K/P_bs${BS}_noprof" \
          "$BS" 353 "$R" noprof
    done
  done
;; esac

# ============================================================= PHASE late ===
case " $PHASES " in *" late "*)
  echo; echo "########## PHASE late: profiled late-context capture, msl=897 $(date -Is) ##########"
  for BS in 1 8 16; do
    wave_run "late_bs${BS}_rep0" "$M/prof/prof_Alate" "$K/L_bs${BS}_prof" \
        "$BS" 897 0 200000000 --save-raw
    parse_raw "$M/prof/prof_Alate" "$BS" 0 "$M/prof/armAlate/tables"
    # the late raws are 0.7-1.5 GB each and /raid is at 100 % with foreign
    # churn; stage the parsed one aside so the coordinator can pull it, and
    # let the NEXT capture start on a filesystem that is not already loaded.
    mkdir -p "$M/prof/raw_stage"
    mv -f "$M/prof/prof_Alate/raw_bs${BS}_rep0.npz" "$M/prof/raw_stage/" 2>/dev/null || true
    df -h /raid | tail -1
  done
;; esac

# ========================================================= PHASE capsweep ===
# Bounded probe: does the bs16 admission-cap win exist at bs4 and bs8 too?
#
# The pinned policy is cap at bs16, none below (bench-protocol.md, M3-I9
# landing), justified by (a) "the cap only changes prefill chunk boundaries at
# bs<16, perf-neutral-to-negative by design" and (b) a bs4 token flip. (b) was
# afterwards refuted -- M3-I9b root-caused that flip as one non-reproducible
# dump and showed the cap is bit-transparent at every bs. And (a) is a
# DECODE-side argument, while the bs16 win this gate just measured is a
# PREFILL-side one: uncapped admission hands the whole mbt budget to the lowest
# live slot, so requests prefill almost serially. The wave-iteration counts the
# adapter's own admission replay reports make the prediction concrete at bs16
# (1887 uncapped vs 1279 capped) and the same arithmetic says bs8 and bs4 should
# move too. Cheapest discriminating test: the benchmark geometry plus the AC-3
# byte diff, at bs4 and bs8, capped, against the uncapped runs already captured.
case " $PHASES " in *" capsweep "*)
  echo; echo "########## PHASE capsweep: cap at bs4/bs8, matched geometry $(date -Is) ##########"
  for BS in 4 8; do
    engine_run "cs_ac3_cap_bs${BS}" "$M/capsweep/ac3_bs${BS}" "$K/CS_A_cap_bs${BS}" \
        "$BS" 132 64 --per-request-token-cap auto
    for R in $(seq 0 $((REPS-1))); do
      REF=$M/perf/M/prompts/synthref_bs${BS}_rep${R}.json
      engine_run "cs_Mfull_cap_bs${BS}_rep${R}" "$M/capsweep/Mfull_bs${BS}" \
          "$K/CS_M_cap_bs${BS}_full" "$BS" 1280 1024 --reference "$REF" \
          --dump-name "bs${BS}_rep${R}.json" --per-request-token-cap auto
      engine_run "cs_Mpre_cap_bs${BS}_rep${R}" "$M/capsweep/Mpre_bs${BS}" \
          "$K/CS_M_cap_bs${BS}_pre" "$BS" 259 2 --reference "$REF" \
          --dump-name "bs${BS}_rep${R}.json" --per-request-token-cap auto
    done
    $PY -u "$HOME/mpk-qwen35/m3i2a/bytediff.py" "$ACC/results/dumps_final" \
        "$M/capsweep/ac3_bs${BS}" "$BS" > "$M/capsweep/bytediff_bs${BS}.json" 2>/dev/null
    $PY - "$M/capsweep/bytediff_bs${BS}.json" "$BS" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
bad = {k: v for k, v in d.get("per_case", {}).items() if v != "identical"}
print(f"  AC-3 bs{sys.argv[2]} capped: identical={d.get('identical')} "
      f"counts={json.dumps(d.get('counts'))} CHANGED={json.dumps(bad) if bad else 'none'}")
EOF
  done
;; esac

# ============================================================ PHASE late2 ===
# UNPROFILED reps at the late-context geometry. The per-stage decomposition
# rides the profiled rep0 (task-event counts are schedule-determined and
# seed-independent, and M3-I10 arm A measured <0.2% wave-to-wave spread), but
# the WAVE WALL that the decomposition is normalised against is a timing, and
# the pinned statistical rule wants >=3 reps with a median and a range for any
# timing. These are those reps, plus the profiler-overhead control.
case " $PHASES " in *" late2 "*)
  echo; echo "########## PHASE late2: unprofiled reps at msl=897 $(date -Is) ##########"
  for BS in 1 8 16; do
    for R in $(seq 0 $((REPS-1))); do
      wave_run "late2_bs${BS}_rep${R}" "$M/prof/noprof_Alate" "$K/L_bs${BS}_noprof" \
          "$BS" 897 "$R" noprof
    done
  done
;; esac

df -h /raid | tail -1
echo; echo "########## GATE_I7_DONE phases='$PHASES' $(date -Is) ##########"
