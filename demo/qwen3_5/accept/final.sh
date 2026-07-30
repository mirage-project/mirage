#!/usr/bin/env bash
# =============================================================================
# final.sh -- THE MECHANICAL FINAL GATE.  This is the file `.pm/accept.sh`
# (pinned, mode 0555) execs, so it is what AC-6 means by "the workspace harness
# at a fixed path".
#
# CONTRACT (from .pm/accept.sh, which is the authority for every value):
#   final.sh --model <id> --batch-sizes "1 2 4 8 16" --prompts <path>
#            --correct-new-tokens 64 --min-input-len 64 --min-output-len 256
#            --e2e-factor-max 1.25 --baseline vllm
#   exit 0  every criterion in scope PASSED
#   exit 1  a criterion FAILED (which one, with numbers) or an integrity violation
#   exit 3  NOT-APPLICABLE: a prerequisite genuinely could not run AND nothing failed
#   exit 2  usage error, or a deliberately NON-BINDING invocation (--rescore)
#
# WHAT IT ENFORCES
#   AC-3 (re-pinned 2026-07-29): coherence + a >=90% top-1 agreement floor with
#        every differing position accounted for + bit-exactness reported as a
#        diagnostic.  Scored by final/score_ac3.py over cold reps collected with
#        M4-I0's fingerprint-scored gate (harness/gate_ac3_stable.sh).
#   AC-4: mpk decode tok/s STRICTLY GREATER than vLLM at bs {1,2,4,8,16} on the
#        pinned 256/1024 workload.  Comparator: a FRESH vLLM sweep in the same
#        window (primary) cross-checked against the pinned baseline table, with a
#        DRIFT rule that fails the gate rather than picking a number.
#   AC-5: mpk e2e <= 1.25x vLLM e2e at every batch size.
#   Integrity first, fail fast: pinned contract re-read from .pm/accept.sh, prompt
#        digest, reference artifact, exactness baseline, clean tree, tool digests.
#
# STAGES (--stages, default all): integrity,ac3,perf,report
# MODES
#   --self-test    GPU-free: run every scorer's unit tests + the committed-fixture
#                  scenarios (including must-FAIL ones).  No engine, no model.
#   --rescore DIR  re-score an existing run directory.  ALWAYS exits 2: a rescore
#                  is not a fresh acceptance run and must never read as one.
#   --no-collect   score what is already in the run dir (no GPU work launched).
#   default        collect on the GPU box (remote by default), then score.
#
# WHY REMOTE BY DEFAULT: the pinned target is one B200 (sm_100) with the 37 GB FP8
# checkpoint, the built megakernel and the vLLM venv; this repo is checked out on a
# host without one.  --run-mode local runs everything in place when the gate is
# invoked ON the GPU box.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # .../demo/qwen3_5/accept
FINAL="$HERE/final"
REPO="$(cd "$HERE/../../.." && pwd)"
PY_LOCAL="${MPK_LOCAL_PY:-python3}"

# ---- the pinned contract, as handed to us (cross-checked against the pinned
#      .pm/accept.sh by final/integrity.py -- never trusted on its own) --------
MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
BATCH_SIZES="1 2 4 8 16"
PROMPTS=".pm/eval/prompts.jsonl"
CORRECT_NEW_TOKENS=64
MIN_INPUT_LEN=64
MIN_OUTPUT_LEN=256
E2E_FACTOR_MAX=1.25
BASELINE="vllm"

# ---- gate knobs -------------------------------------------------------------
STAGES="integrity,ac3,perf,report"
OUT=""
REPS=3
RUN_MODE="${MPK_GATE_RUN_MODE:-remote}"
HOST="${MPK_GATE_HOST:-catalyst-B200}"
BOX_ROOT="${MPK_BOX_ROOT:-\$HOME/mpk-qwen35}"     # expanded ON the box
CANDIDATES="${MPK_GATE_CANDIDATES:-6,5,2,3,1,0,7,4}"
AGENT_ROOT="${MPK_AGENT_ROOT:-}"
BASELINE_DIR="$HERE/baselines/vllm-0.25.1-20260725"
WORKLOAD_IN=256
WORKLOAD_OUT=1024
POLL_SECONDS="${MPK_GATE_POLL_SECONDS:-30}"
STAGE_TIMEOUT="${MPK_GATE_STAGE_TIMEOUT:-21600}"          # 6 h per collect stage
SELF_TEST=0
RESCORE=""
NO_COLLECT=0
SKIP_HF=0
NONBINDING=0

usage() { sed -n '2,45p' "${BASH_SOURCE[0]}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --batch-sizes) BATCH_SIZES="$2"; shift 2;;
    --prompts) PROMPTS="$2"; shift 2;;
    --correct-new-tokens) CORRECT_NEW_TOKENS="$2"; shift 2;;
    --min-input-len) MIN_INPUT_LEN="$2"; shift 2;;
    --min-output-len) MIN_OUTPUT_LEN="$2"; shift 2;;
    --e2e-factor-max) E2E_FACTOR_MAX="$2"; shift 2;;
    --baseline) BASELINE="$2"; shift 2;;
    --stages) STAGES="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    --run-mode) RUN_MODE="$2"; shift 2;;
    --host) HOST="$2"; shift 2;;
    --box-root) BOX_ROOT="$2"; shift 2;;
    --candidates) CANDIDATES="$2"; shift 2;;
    --agent-root) AGENT_ROOT="$2"; shift 2;;
    --baseline-dir) BASELINE_DIR="$2"; shift 2;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2;;
    --stage-timeout) STAGE_TIMEOUT="$2"; shift 2;;
    --non-binding) NONBINDING=1; shift;;
    --self-test) SELF_TEST=1; shift;;
    --rescore) RESCORE="$2"; shift 2;;
    --no-collect) NO_COLLECT=1; shift;;
    --skip-hf) SKIP_HF=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "final.sh: unknown flag $1" >&2; usage >&2; exit 2;;
  esac
done

BS_COMMA="$(echo "$BATCH_SIZES" | tr ' ' ',' | sed 's/,,*/,/g;s/^,//;s/,$//')"
in_stage() { case ",$STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }

# ---- resolve BOX_ROOT to a real absolute path (M4-I8) ------------------------
# The default carries a LITERAL, unexpanded $HOME so it can be expanded on the
# box.  But BOX_ROOT is also used locally to BUILD remote paths
# (REMOTE_RUN="$BOX_ROOT/final-gate/run-..."), and those land inside SINGLE
# quotes on the remote command line (`bash '$REMOTE_RUN/remote_setup.sh'`),
# where $HOME does not expand -- so the gate tried to run
# `$HOME/mpk-qwen35/final-gate/...` as a literal directory name and the
# M4-status run had to pass --box-root as an absolute path to work at all.
# Ask the box for its $HOME once, here, and substitute.  A caller-supplied
# absolute --box-root is left exactly as given.
case "$BOX_ROOT" in
  *'$HOME'*|*'${HOME}'*)
    if [ "$RUN_MODE" = "remote" ]; then
      _box_home="$(ssh -o BatchMode=yes "$HOST" 'printf %s "$HOME"' 2>/dev/null)"
    else
      _box_home="$HOME"
    fi
    if [ -z "$_box_home" ]; then
      echo "final.sh: cannot resolve \$HOME on $HOST to expand --box-root" >&2
      echo "          pass an absolute --box-root instead" >&2
      exit 2
    fi
    BOX_ROOT="${BOX_ROOT//\$\{HOME\}/$_box_home}"
    BOX_ROOT="${BOX_ROOT//\$HOME/$_box_home}"
    unset _box_home
    ;;
esac
case "$BOX_ROOT" in
  /*) ;;
  *) echo "final.sh: --box-root must be absolute, got '$BOX_ROOT'" >&2; exit 2;;
esac

# =============================================================== self-test ===
if [ "$SELF_TEST" = "1" ]; then
  echo "########## final.sh --self-test (GPU-free) $(date -Is) ##########"
  rc=0
  for t in "$FINAL"/tests/test_*.py; do
    echo "----- $(basename "$t")"
    "$PY_LOCAL" "$t" -v 2>&1 | tail -25 || rc=1
  done
  echo "----- reused machinery: opt/m4i0/scripts/test_gate_scorer.py"
  "$PY_LOCAL" "$HERE/opt/m4i0/scripts/test_gate_scorer.py" 2>&1 | tail -12 || rc=1
  echo "----- reused machinery: harness tests"
  for t in "$HERE"/harness/tests/test_*.py; do
    ( cd "$HERE/harness" && "$PY_LOCAL" "$t" 2>&1 | tail -3 ) || rc=1
  done
  echo "########## SELF_TEST $([ $rc -eq 0 ] && echo PASS || echo FAIL) ##########"
  exit $rc
fi

# ================================================================= run dir ===
if [ -n "$RESCORE" ]; then
  RUNDIR="$RESCORE"; NONBINDING=1; NO_COLLECT=1
  [ -d "$RUNDIR" ] || { echo "final.sh: --rescore dir not found: $RUNDIR" >&2; exit 2; }
else
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  RUNDIR="${OUT:-${MPK_FINAL_OUT:-$HOME/mpk-final-gate}/$STAMP}"
  mkdir -p "$RUNDIR" || { echo "final.sh: cannot create $RUNDIR" >&2; exit 2; }
fi
mkdir -p "$RUNDIR/ac3" "$RUNDIR/perf"

# ---- locate the agent repo (the .pm/ owner) --------------------------------
if [ -z "$AGENT_ROOT" ]; then
  d="$PWD"
  for _ in 1 2 3 4 5 6; do
    if [ -f "$d/.pm/accept.sh" ]; then AGENT_ROOT="$d"; break; fi
    d="$(dirname "$d")"
  done
fi
GIT_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
cat > "$RUNDIR/run_meta.json" <<EOF
{
  "gate": "demo/qwen3_5/accept/final.sh",
  "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "host": "$(hostname)",
  "run_mode": "$RUN_MODE",
  "remote_host": "$HOST",
  "git_sha": "$GIT_SHA",
  "repo": "$REPO",
  "agent_root": "${AGENT_ROOT:-null}",
  "stages": "$STAGES",
  "reps": $REPS,
  "batch_sizes": "$BS_COMMA",
  "workload": {"input_len": $WORKLOAD_IN, "output_len": $WORKLOAD_OUT},
  "e2e_factor_max": $E2E_FACTOR_MAX,
  "correct_new_tokens": $CORRECT_NEW_TOKENS,
  "baseline_dir": "$BASELINE_DIR",
  "run_dir": "$RUNDIR",
  "no_collect": $NO_COLLECT,
  "skip_hf": $SKIP_HF,
  "non_binding": $NONBINDING
}
EOF
echo "########## MPK FINAL GATE  sha=${GIT_SHA:0:12}  stages=$STAGES  $(date -Is) ##########"
echo "run dir: $RUNDIR"
cat "$RUNDIR/run_meta.json"

# ============================================================== integrity ===
INTEGRITY_JSON="$RUNDIR/integrity.json"
if in_stage integrity; then
  echo; echo "===== STAGE integrity $(date -Is) ====="
  AG=()
  [ -n "$AGENT_ROOT" ] && AG=(--agent-root "$AGENT_ROOT")
  [ "$NONBINDING" = "1" ] && AG+=(--non-binding)
  "$PY_LOCAL" "$FINAL/integrity.py" \
      --accept-dir "$HERE" --repo-root "$REPO" "${AG[@]+"${AG[@]}"}" \
      --baseline-dir "$BASELINE_DIR" --bench-vllm "$HERE/bench_vllm.py" \
      --model "$MODEL" --batch-sizes "$BATCH_SIZES" --prompts "$PROMPTS" \
      --correct-new-tokens "$CORRECT_NEW_TOKENS" \
      --min-input-len "$MIN_INPUT_LEN" --min-output-len "$MIN_OUTPUT_LEN" \
      --e2e-factor-max "$E2E_FACTOR_MAX" --baseline "$BASELINE" \
      --workload-input-len "$WORKLOAD_IN" --workload-output-len "$WORKLOAD_OUT" \
      --output-json "$INTEGRITY_JSON"
  IRC=$?
  if [ "$IRC" -ne 0 ]; then
    echo "===== INTEGRITY FAILED -- refusing to measure anything ====="
    NB0=()
    [ "$NONBINDING" = "1" ] && NB0=(--non-binding)
    "$PY_LOCAL" "$FINAL/report.py" --run-meta "$RUNDIR/run_meta.json" \
        --integrity "$INTEGRITY_JSON" --stages "$STAGES" \
        --output-json "$RUNDIR/report.json" --output-summary "$RUNDIR/summary.txt" \
        "${NB0[@]+"${NB0[@]}"}" || true
    exit 1
  fi
fi

# ======================================================== remote plumbing ===
REMOTE_TREE=""
REMOTE_RUN=""
rsh() { ssh -o BatchMode=yes "$HOST" "$@"; }

remote_setup() {
  echo; echo "===== deploy to $HOST at $GIT_SHA $(date -Is) ====="
  REMOTE_RUN="$BOX_ROOT/final-gate/run-$(basename "$RUNDIR")"
  rsh "mkdir -p '$REMOTE_RUN'" || return 2
  rsh "cat > '$REMOTE_RUN/remote_setup.sh'" < "$FINAL/remote_setup.sh" || return 2
  rsh "SHA='$GIT_SHA' MPK_BOX_ROOT=\"$BOX_ROOT\" bash '$REMOTE_RUN/remote_setup.sh'" \
      2>&1 | tee "$RUNDIR/remote_setup.log"
  REMOTE_TREE="$(grep -oE '"dest":"[^"]+"' "$RUNDIR/remote_setup.log" | tail -1 \
                 | cut -d'"' -f4)"
  [ -n "$REMOTE_TREE" ] || { echo "deploy failed: no clone path reported"; return 2; }
  echo "remote tree: $REMOTE_TREE"
  echo "remote run:  $REMOTE_RUN"
}

# rstage <name> <local-work-script>: ship it, run it under tmux, poll to the end.
# tmux (not `nohup` through a one-shot ssh, which does not reliably survive here
# -- bench-protocol.md 9) and the work script is a FILE, so nothing has to
# survive a second round of shell quoting.
rstage() {
  local NAME="$1" SCRIPT="$2"
  local RS="$REMOTE_RUN/stage_$NAME.sh" RL="$REMOTE_RUN/stage_$NAME.log"
  rsh "cat > '$RS'" < "$SCRIPT" || return 2
  rsh "tmux kill-session -t gate_$NAME >/dev/null 2>&1; rm -f '$RL'; \
       tmux new-session -d -s gate_$NAME bash '$RS'" || return 2
  echo "  launched $NAME in tmux session gate_$NAME (log $RL)"
  local waited=0 marker=""
  while [ "$waited" -lt "$STAGE_TIMEOUT" ]; do
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
    marker="$(rsh "grep -h GATE_STAGE_EXIT '$RL' 2>/dev/null | tail -1" || true)"
    if [ -n "$marker" ]; then
      echo "  $NAME finished after ${waited}s: $marker"
      rsh "tail -n 20 '$RL'" || true
      echo "${marker#GATE_STAGE_EXIT=}" > "$RUNDIR/.stage_$NAME.rc"
      return 0
    fi
    printf '  [%s] %ss: %s\n' "$NAME" "$waited" \
           "$(rsh "tail -n 1 '$RL' 2>/dev/null" | tr -d '\r' | cut -c1-120)"
  done
  echo "  STAGE TIMEOUT after ${waited}s -- session left running for inspection" >&2
  return 4
}

# ==================================================================== AC-3 ===
AC3_SCORE="$RUNDIR/ac3/ac3_score.json"
if in_stage ac3; then
  echo; echo "===== STAGE ac3 (collect + score) $(date -Is) ====="
  REVISION="$("$PY_LOCAL" -c "
import json,sys
print((json.load(open(sys.argv[1]))['meta'] or {}).get('revision',''))
" "$HERE/reference/reference_outputs.json")"
  if [ "$NO_COLLECT" = "0" ]; then
    if [ "$RUN_MODE" = "remote" ]; then
      [ -n "$REMOTE_TREE" ] || remote_setup || exit 3
      cat > "$RUNDIR/.work_ac3.sh" <<WORK
#!/usr/bin/env bash
{
  export MPK_BOX_ROOT="$BOX_ROOT"
  bash "$REMOTE_TREE/demo/qwen3_5/accept/final/collect_ac3.sh" \\
    --out "$REMOTE_RUN/ac3" \\
    --accept-dir "$REMOTE_TREE/demo/qwen3_5/accept" \\
    --reps $REPS --batch-sizes "$BS_COMMA" --candidates "$CANDIDATES" \\
    --model "$MODEL" --revision "$REVISION" $([ "$SKIP_HF" = 1 ] && echo --skip-hf)
} > "$REMOTE_RUN/stage_ac3.log" 2>&1
echo "GATE_STAGE_EXIT=\$?" >> "$REMOTE_RUN/stage_ac3.log"
WORK
      rstage ac3 "$RUNDIR/.work_ac3.sh" || exit 3
      echo "  pulling AC-3 artifacts back"
      rsync -az --exclude 'kernels/**' "$HOST:$REMOTE_RUN/ac3/" "$RUNDIR/ac3/" \
        || { echo "rsync of AC-3 artifacts failed" >&2; exit 3; }
    else
      MPK_BOX_ROOT="$BOX_ROOT" bash "$FINAL/collect_ac3.sh" --out "$RUNDIR/ac3" \
        --accept-dir "$HERE" --reps "$REPS" --batch-sizes "$BS_COMMA" \
        --candidates "$CANDIDATES" --model "$MODEL" --revision "$REVISION" \
        $([ "$SKIP_HF" = 1 ] && echo --skip-hf) || exit 3
    fi
  fi
  COH=()
  [ -f "$RUNDIR/ac3/coherence_inputs.json" ] && \
    COH=(--coherence "$RUNDIR/ac3/coherence_inputs.json")
  GR=()
  [ -f "$RUNDIR/ac3/sweep/gate_ac3_stable.json" ] && \
    GR=(--gate-report "$RUNDIR/ac3/sweep/gate_ac3_stable.json")
  EM=()
  [ -f "$FINAL/engine_margins.json" ] && EM=(--engine-margins "$FINAL/engine_margins.json")
  "$PY_LOCAL" "$FINAL/score_ac3.py" \
      --reference "$HERE/reference/reference_outputs.json" \
      --reps-root "$RUNDIR/ac3/sweep/reps" \
      --baseline "$HERE/results/dumps_final" \
      --mechanisms "$FINAL/mechanisms.json" \
      "${COH[@]+"${COH[@]}"}" "${GR[@]+"${GR[@]}"}" "${EM[@]+"${EM[@]}"}" \
      --batch-sizes "$BS_COMMA" --expect-new-tokens "$CORRECT_NEW_TOKENS" \
      --reps-required "$REPS" --output-json "$AC3_SCORE" || true
fi

# =============================================================== AC-4/AC-5 ===
PERF_SCORE="$RUNDIR/perf/perf_score.json"
if in_stage perf; then
  echo; echo "===== STAGE perf (AC-4/AC-5 collect + score) $(date -Is) ====="
  if [ "$NO_COLLECT" = "0" ]; then
    if [ "$RUN_MODE" = "remote" ]; then
      [ -n "$REMOTE_TREE" ] || remote_setup || exit 3
      cat > "$RUNDIR/.work_perf.sh" <<WORK
#!/usr/bin/env bash
{
  export MPK_BOX_ROOT="$BOX_ROOT"
  ACC="$REMOTE_TREE/demo/qwen3_5/accept"
  echo "### MPK arms ###"
  bash "\$ACC/opt/m3i7/scripts/gpu_guard_i7.sh" "$CANDIDATES" -- \\
    bash "\$ACC/final/collect_perf.sh" --out "$REMOTE_RUN/perf/mpk" \\
      --accept-dir "\$ACC" --reps $REPS --batch-sizes "$BS_COMMA" \\
      --input-len $WORKLOAD_IN --max-new-tokens $WORKLOAD_OUT
  MRC=\$?
  echo "### fresh vLLM comparator ###"
  bash "\$ACC/opt/m3i7/scripts/gpu_guard_i7.sh" "$CANDIDATES" -- \\
    bash "\$ACC/final/collect_vllm.sh" --out "$REMOTE_RUN/perf/vllm_fresh" \\
      --accept-dir "\$ACC" --reps $REPS --batch-sizes "$BS_COMMA" \\
      --pinned-baseline "\$ACC/baselines/vllm-0.25.1-20260725"
  VRC=\$?
  echo "MPK_COLLECT_RC=\$MRC VLLM_COLLECT_RC=\$VRC"
  [ \$MRC -eq 0 ] && [ \$VRC -eq 0 ]
} > "$REMOTE_RUN/stage_perf.log" 2>&1
echo "GATE_STAGE_EXIT=\$?" >> "$REMOTE_RUN/stage_perf.log"
WORK
      rstage perf "$RUNDIR/.work_perf.sh" || exit 3
      echo "  pulling perf artifacts back"
      rsync -az --exclude 'kernels/**' "$HOST:$REMOTE_RUN/perf/" "$RUNDIR/perf/" \
        || { echo "rsync of perf artifacts failed" >&2; exit 3; }
    else
      ACC="$HERE"
      bash "$ACC/opt/m3i7/scripts/gpu_guard_i7.sh" "$CANDIDATES" -- \
        bash "$FINAL/collect_perf.sh" --out "$RUNDIR/perf/mpk" --accept-dir "$ACC" \
          --reps "$REPS" --batch-sizes "$BS_COMMA" --input-len "$WORKLOAD_IN" \
          --max-new-tokens "$WORKLOAD_OUT" || exit 3
      bash "$ACC/opt/m3i7/scripts/gpu_guard_i7.sh" "$CANDIDATES" -- \
        bash "$FINAL/collect_vllm.sh" --out "$RUNDIR/perf/vllm_fresh" \
          --accept-dir "$ACC" --reps "$REPS" --batch-sizes "$BS_COMMA" \
          --pinned-baseline "$BASELINE_DIR" || exit 3
    fi
  fi
  "$PY_LOCAL" "$FINAL/score_perf.py" \
      --mpk-root "$RUNDIR/perf/mpk" --vllm-fresh "$RUNDIR/perf/vllm_fresh" \
      --vllm-pinned "$BASELINE_DIR" --bench-vllm "$HERE/bench_vllm.py" \
      --batch-sizes "$BS_COMMA" --e2e-factor-max "$E2E_FACTOR_MAX" \
      --reps-required "$REPS" --output-json "$PERF_SCORE" || true
fi

# ================================================================== report ===
echo; echo "===== STAGE report $(date -Is) ====="
NB=()
[ "$NONBINDING" = "1" ] && NB=(--non-binding)
"$PY_LOCAL" "$FINAL/report.py" --run-meta "$RUNDIR/run_meta.json" \
    $([ -f "$INTEGRITY_JSON" ] && echo --integrity "$INTEGRITY_JSON") \
    $([ -f "$AC3_SCORE" ] && echo --ac3 "$AC3_SCORE") \
    $([ -f "$PERF_SCORE" ] && echo --perf "$PERF_SCORE") \
    --stages "$STAGES" --output-json "$RUNDIR/report.json" \
    --output-summary "$RUNDIR/summary.txt" "${NB[@]+"${NB[@]}"}"
RC=$?
if [ "$NONBINDING" = "1" ]; then
  echo "NON-BINDING invocation (--rescore / --non-binding): the verdict above is"
  echo "informational -- the run deviated from the pinned contract or re-scored"
  echo "existing artifacts. Exiting 2 so it can never be mistaken for an AC-6"
  echo "acceptance result. (A criterion that FAILED still shows as FAIL above.)"
  exit 2
fi
echo "########## FINAL_GATE_DONE rc=$RC $(date -Is) ##########"
exit $RC
