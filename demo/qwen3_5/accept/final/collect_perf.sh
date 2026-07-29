#!/usr/bin/env bash
# AC-4/AC-5 MPK COLLECT (box-side).  Produces what score_perf.py consumes:
#
#   $OUT/full/timings_bs<N>_rep<R>.json   msl 1280, 1024 new tokens
#   $OUT/pre/timings_bs<N>_rep<R>.json    msl  259,    2 new tokens, same prompts
#   $OUT/prompts/synthref_bs<N>_rep<R>.json
#   $OUT/audit.json                       device identity, foreign floor, per-cell
#                                         gpu_before, cap policy + compile-define evidence
#   $OUT/logs/<tag>.log                   every run's own log (carries `GATE gpu=N`)
#
# MEASUREMENT DISCIPLINE -- inherited from M3-I7, defect for defect
#   1. Decode throughput is the PREFILL-SUBTRACTED SLOPE, so two arms are run per
#      (bs, rep) over the SAME prompts and INTERLEAVED inside one window: the
#      slope subtracts one arm from the other, so they must share a window or box
#      drift lands in the answer.
#   2. Prompts come from --reference via make_matched_reference.py.
#      `--prompts-file` is NOT a prompt source (mpk_engine_run.py:678, read only
#      under --verify-chat-template); M3-I9 measured the AC-3 prompts by mistake
#      that way.
#   3. `--per-request-token-cap` is a COMPILE-TIME define, so every arm gets its
#      OWN kernel dir and the define is grepped back out of each dir as evidence.
#      M3-I7's first pass shared one dir and measured one binary twice.
#   4. Per-rep DRAIN GATE + gpu_before recorded for the device THIS SESSION
#      actually pinned (probed once from the process's own CUDA context UUID,
#      never from a candidate list -- the M3-I7 mislabelling defect).  A rep that
#      starts above the session's foreign floor is excluded IN ANALYSIS by
#      score_perf.py, not silently averaged in.
#   5. >=3 reps per arm; per-rep values are retained and reported, medians are
#      derived, and no rep is ever re-rolled away.
#
# EXIT: 0 collected, 3 could not collect (no idle device, missing prompts, no
#       cell produced timings).  It never decides AC-4/AC-5.
set -uo pipefail

BOX_ROOT="${MPK_BOX_ROOT:-$HOME/mpk-qwen35}"
PY="${MPK_PY:-$BOX_ROOT/venv-rm/bin/python}"
# The interpreter's own bin dir goes on PATH for the same reason
# collect_vllm.sh does it: pip console scripts (ninja, cmake) live there and the
# JIT paths shell out to them.
export PATH="$(dirname "$PY"):${CUDA_BIN:-/usr/local/cuda-12.8/bin}:$PATH"
export HF_HOME="${HF_HOME:-$BOX_ROOT/hf}"
export PYTHONUNBUFFERED=1

OUT=""; ACC=""; REPS=3; BSS="1,2,4,8,16"; INPUT_LEN=256; MSL_FULL=1280
MSL_PRE=259; MNT_FULL=1024; MNT_PRE=2; MBT=16; PAGE=256; KROOT=""
# The admission cap is NOT decided here.  accept/admission_policy.py owns it and
# mpk_engine_run.py's default is "policy", so every cell runs the SHIPPED policy
# and records the compile-time value it actually ran with in its own timings
# artifact.  score_perf.py checks that recorded value against the same authority.
CAP_REQUEST="policy"
GPU="${CUDA_VISIBLE_DEVICES:-}"
FLOOR_ARG="${MPK_I7_GPU_FLOOR:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    --accept-dir) ACC="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    --batch-sizes) BSS="$2"; shift 2;;
    --input-len) INPUT_LEN="$2"; shift 2;;
    --msl-full) MSL_FULL="$2"; shift 2;;
    --msl-pre) MSL_PRE="$2"; shift 2;;
    --max-new-tokens) MNT_FULL="$2"; shift 2;;
    --per-request-token-cap) CAP_REQUEST="$2"; shift 2;;
    --kernel-root) KROOT="$2"; shift 2;;
    --python) PY="$2"; shift 2;;
    *) echo "collect_perf.sh: unknown flag $1" >&2; exit 3;;
  esac
done
[ -n "$OUT" ] && [ -n "$ACC" ] || {
  echo "collect_perf.sh: --out and --accept-dir are required" >&2; exit 3; }
[ -n "$GPU" ] || { echo "collect_perf.sh: CUDA_VISIBLE_DEVICES must be pinned by the guard" >&2; exit 3; }
[ -n "$KROOT" ] || KROOT="$OUT/kernels"
mkdir -p "$OUT/full" "$OUT/pre" "$OUT/prompts" "$OUT/logs" "$OUT/meta" "$KROOT"
TREE="$(cd "$ACC/../../.." && pwd)"
export PYTHONPATH="$TREE/python${PYTHONPATH:+:$PYTHONPATH}"

echo "########## AC-4/AC-5 MPK COLLECT  GATE gpu=$GPU  $(date -Is) ##########"
echo "tree=$TREE sha=$(git -C "$TREE" rev-parse HEAD 2>/dev/null)"
md5sum "$TREE"/python/mirage/core.cpython-*.so 2>/dev/null

# ---- device identity, from THIS process's own CUDA context (not a list) -----
DEVJSON="$OUT/meta/device.json"
"$PY" - "$DEVJSON" <<'EOF'
import json, os, subprocess, sys
info = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "uuid": None, "phys_index": None, "name": None}
try:
    import torch
    p = torch.cuda.get_device_properties(0)
    info["name"] = p.name
    info["uuid"] = str(getattr(p, "uuid", "") or "") or None
except Exception as e:
    info["error"] = f"{type(e).__name__}: {e}"
if info["uuid"]:
    try:
        rows = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid",
                               "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=60).stdout
        want = info["uuid"].replace("GPU-", "").strip()
        for line in rows.splitlines():
            idx, _, uu = line.partition(",")
            if want and want in uu.strip():
                info["phys_index"] = int(idx.strip()); break
    except Exception as e:
        info["nvidia_smi_error"] = f"{type(e).__name__}: {e}"
json.dump(info, open(sys.argv[1], "w"), indent=1)
print("[collect] device:", json.dumps(info))
EOF
PHYS="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('phys_index') or '')" "$DEVJSON")"
if [ -n "$PHYS" ] && [ "$PHYS" != "$GPU" ]; then
  echo "[collect] DEVICE CORRECTION: the process reports physical GPU $PHYS while"
  echo "          CUDA_VISIBLE_DEVICES claimed '$GPU'.  Adopting $PHYS -- the run's"
  echo "          own context is the truth (M3-I7 mislabelling defect)."
  GPU="$PHYS"
fi
gpu_used () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' '; }
if [ -z "$FLOOR_ARG" ]; then FLOOR_ARG="$(gpu_used)"; fi
FLOOR="${FLOOR_ARG:-0}"
DRAIN_LIMIT=$((FLOOR + 400))
echo "[collect] pinned GPU $GPU, foreign floor ${FLOOR}MiB, drain limit ${DRAIN_LIMIT}MiB"

drain () {   # wait until OUR previous rep's memory is really gone (M3-I6a)
  local i used lo hi s
  for i in $(seq 1 60); do
    lo=99999999; hi=-1
    for s in 1 2 3; do
      used="$(gpu_used)"; used="${used:-99999999}"
      [ "$used" -lt "$lo" ] && lo=$used
      [ "$used" -gt "$hi" ] && hi=$used
      [ "$s" -lt 3 ] && sleep 2
    done
    [ "$hi" -le "$DRAIN_LIMIT" ] && { [ "$i" -gt 1 ] && echo "    drained (${lo}..${hi}MiB after $i checks)"; return 0; }
    echo "    draining GPU $GPU: ${lo}..${hi}MiB > ${DRAIN_LIMIT}MiB (wait $i/60)"
    sleep 5
  done
  echo "    REFUSING this cell: device $GPU still at ${hi}MiB (limit ${DRAIN_LIMIT}MiB)."
  echo "    MPK claims every SM and spin-waits, so starting next to live foreign"
  echo "    work is a deadlock, not noise (SM-RESIDENCY LAW, M3-I2a)."
  return 1
}

# cell <tag> <arm> <bs> <rep> <msl> <mnt> <reffile> <capflag...>
cell () {
  local TAG=$1 ARM=$2 BS=$3 REP=$4 MSL=$5 MNT=$6 REF=$7; shift 7
  local KD="$KROOT/${ARM}_bs${BS}"
  local RK=""; [ -f "$KD/task_graph_rank0.json" ] && RK="--reuse-kernel"
  local BEFORE AFTER RC
  if ! drain; then
    printf '{"tag":"%s","arm":"%s","bs":%d,"rep":%d,"status":"error","note":"drain gate timed out; cell NOT launched","gpu_index":%s}\n' \
           "$TAG" "$ARM" "$BS" "$REP" "$GPU" > "$OUT/meta/$TAG.json"
    return 0
  fi
  BEFORE="$(gpu_used)"
  echo "--- $TAG arm=$ARM bs=$BS rep=$REP msl=$MSL mnt=$MNT gpu_before=${BEFORE}MiB $(date -Is) ---"
  ( cd "$ACC" && timeout 5400 "$PY" -u mpk_engine_run.py \
      --batch-size "$BS" --max-seq-length "$MSL" --max-new-tokens "$MNT" \
      --mbt "$MBT" --page-size "$PAGE" --reference "$REF" \
      --out-dir "$OUT/$ARM" --dump-name "bs${BS}_rep${REP}.json" \
      --per-request-token-cap "$CAP_REQUEST" \
      --kernel-dir "$KD" $RK "$@" ) > "$OUT/logs/$TAG.log" 2>&1
  RC=$?
  AFTER="$(gpu_used)"
  echo "  [$TAG] rc=$RC gpu_after=${AFTER}MiB $(grep -h 'wave=' "$OUT/logs/$TAG.log" | tail -1)"
  [ "$RC" -ne 0 ] && tail -6 "$OUT/logs/$TAG.log" | sed 's/^/    /'
  local DEFINE
  DEFINE="$(grep -rho 'MPK_MAX_TOKENS_PER_REQUEST=[0-9]*' "$KD" 2>/dev/null | sort -u | paste -sd, -)"
  printf '{"tag":"%s","arm":"%s","bs":%d,"rep":%d,"status":"%s","rc":%d,"gpu_index":%s,"gpu_before_mib":%s,"gpu_after_mib":%s,"kernel_dir":"%s","cap_define":"%s","cap_flag":"%s"}\n' \
         "$TAG" "$ARM" "$BS" "$REP" "$([ $RC -eq 0 ] && echo ok || echo error)" \
         "$RC" "$GPU" "${BEFORE:-null}" "${AFTER:-null}" "$KD" "${DEFINE:-}" "$*" \
         > "$OUT/meta/$TAG.json"
}

# ---- prompts: the pinned baseline sampler's own ids, per (bs, rep) ----------
for BS in ${BSS//,/ }; do
  for R in $(seq 0 $((REPS - 1))); do
    F="$OUT/prompts/synthref_bs${BS}_rep${R}.json"
    [ -f "$F" ] || "$PY" "$ACC/opt/m3i7/scripts/make_matched_reference.py" \
        --batch-size "$BS" --rep "$R" --input-len "$INPUT_LEN" --out "$F" \
        >> "$OUT/logs/prompts.log" 2>&1
    [ -f "$F" ] || { echo "COLLECT FAILED: could not materialise $F" >&2; exit 3; }
  done
done
tail -2 "$OUT/logs/prompts.log" 2>/dev/null

# ---- the sweep: both arms interleaved, rep-major -----------------------------
for R in $(seq 0 $((REPS - 1))); do
  for BS in ${BSS//,/ }; do
    REF="$OUT/prompts/synthref_bs${BS}_rep${R}.json"
    # One kernel dir per (arm, bs): msl AND the cap are compile-time, and the
    # policy resolves the cap per batch size, so no two cells that could differ
    # in either may share a dir (M3-I7's shared-kernel-dir defect).
    cell "full_bs${BS}_rep${R}" full "$BS" "$R" "$MSL_FULL" "$MNT_FULL" "$REF"
    cell "pre_bs${BS}_rep${R}"  pre  "$BS" "$R" "$MSL_PRE"  "$MNT_PRE"  "$REF"
  done
done

# ---- audit.json --------------------------------------------------------------
POLICY_JSON_FILE="$OUT/meta/admission_policy.json"
"$PY" "$ACC/admission_policy.py" > "$POLICY_JSON_FILE" 2>/dev/null \
  || echo '{"error":"admission_policy.py unreadable"}' > "$POLICY_JSON_FILE"
"$PY" - "$OUT" "$BSS" "$POLICY_JSON_FILE" "$DEVJSON" <<'EOF'
import json, sys, glob, os
out, bss, policy_file, devjson = sys.argv[1:5]
cells = {}
for p in sorted(glob.glob(os.path.join(out, "meta", "*.json"))):
    if p.endswith("device.json"):
        continue
    try:
        d = json.load(open(p))
    except Exception:
        continue
    cells[d["tag"]] = d
befores = [c["gpu_before_mib"] for c in cells.values()
           if isinstance(c.get("gpu_before_mib"), int)]
try:
    policy = json.load(open(policy_file))
except Exception as e:
    policy = {"error": f"{type(e).__name__}: {e}"}
doc = {"schema": "final/perf_audit/v1",
       "device": json.load(open(devjson)),
       "foreign_floor_mib": (min(befores) if befores else None),
       "foreign_floor_definition": "the quietest pre-run sample observed on the "
                                   "pinned device in THIS session "
                                   "(harness/gate_ac3_stable.py's method)",
       "admission_policy": policy,
       "cap_define_by_cell": {t: c.get("cap_define") for t, c in cells.items()},
       "cells": cells}
json.dump(doc, open(os.path.join(out, "audit.json"), "w"), indent=2)
n_ok = sum(1 for c in cells.values() if c.get("status") == "ok")
print(f"[collect] audit: {n_ok}/{len(cells)} cells ok, foreign floor "
      f"{doc['foreign_floor_mib']} MiB, admission policy "
      f"{json.dumps(policy.get('per_bs', policy))}")
EOF

NT=$(find "$OUT/full" "$OUT/pre" -name 'timings_bs*_rep*.json' 2>/dev/null | wc -l)
echo "[collect] timings artifacts: $NT"
echo "########## COLLECT_PERF_DONE $(date -Is) ##########"
[ "$NT" -gt 0 ] || exit 3
exit 0
