#!/usr/bin/env bash
# AC-3 COLLECT (box-side).  Produces the artifacts score_ac3.py consumes:
#
#   $OUT/preflight/<attempt>/gate_ac3_stable.json   device probe, >=3 cold reps at bs1
#   $OUT/sweep/{reps,logs,gate_ac3_stable.json}     the full cold sweep, all bs
#   $OUT/coherence_inputs.json                      HF text + perplexity (hf_score.py)
#   $OUT/collect_ac3.json                           what happened, per attempt
#
# It REUSES M4-I0's gate (harness/gate_ac3_stable.sh) for the cold-rep +
# fingerprint machinery rather than re-implementing it: cold compile per rep,
# KV/GDN wave-boundary fingerprints, quarantine-and-replace, per-rep device
# audit, launch ledger, drain gate.
#
# PRE-FLIGHT DEVICE PROBE + BOUNDED DEVICE-LEVEL RE-RUNS
#   docs/qwen35/bench-protocol.md, "M4 gate policy", items 1 and 3: probe the
#   candidate device with >=3 cold reps at bs1 and do not use a device that
#   diverges; at most 2 further attempts on fresh devices, every attempt
#   documented.  Under the RE-PINNED AC-3 a divergent device is no longer a gate
#   failure (divergence is in-band and reported), so the probe is a
#   device-selection instrument here -- kept because the goal's Plan Evolution
#   Log explicitly keeps it as a diagnostic.
#
# EXIT: 0 collected (whatever the gate's own token verdict was -- that is a
#       DIAGNOSTIC under the re-pinned AC-3 and is scored by score_ac3.py, not
#       here), 3 could not collect (no usable device / no reps / HF stage
#       impossible).  This script never decides AC-3.
set -uo pipefail

BOX_ROOT="${MPK_BOX_ROOT:-$HOME/mpk-qwen35}"
PY="${MPK_PY:-$BOX_ROOT/venv-rm/bin/python}"
export PATH="${CUDA_BIN:-/usr/local/cuda-12.8/bin}:$PATH"
export HF_HOME="${HF_HOME:-$BOX_ROOT/hf}"
export PYTHONUNBUFFERED=1

OUT=""; REPS=3; BSS="1,2,4,8,16"; CANDS="6,5,2,3,1,0,7,4"; MAX_DEVICE_ATTEMPTS=3
ACC=""; REVISION=""; MODEL="Qwen/Qwen3.5-35B-A3B-FP8"; PREFLIGHT_REPS=3
SKIP_HF=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    --accept-dir) ACC="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    --batch-sizes) BSS="$2"; shift 2;;
    --candidates) CANDS="$2"; shift 2;;
    --preflight-reps) PREFLIGHT_REPS="$2"; shift 2;;
    --max-device-attempts) MAX_DEVICE_ATTEMPTS="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --revision) REVISION="$2"; shift 2;;
    --python) PY="$2"; shift 2;;
    --skip-hf) SKIP_HF=1; shift;;
    *) echo "collect_ac3.sh: unknown flag $1" >&2; exit 3;;
  esac
done
[ -n "$OUT" ] && [ -n "$ACC" ] && [ -n "$REVISION" ] || {
  echo "collect_ac3.sh: --out, --accept-dir and --revision are required" >&2; exit 3; }
mkdir -p "$OUT"
export PYTHONPATH="$(cd "$ACC/../../.." && pwd)/python${PYTHONPATH:+:$PYTHONPATH}"

JQ() { "$PY" -c "
import json,sys
d=json.load(open(sys.argv[1]))
cur=d
for k in sys.argv[2].split('.'):
    cur=(cur or {}).get(k) if isinstance(cur,dict) else None
print('' if cur is None else cur)
" "$1" "$2" 2>/dev/null; }

REMAINING="$CANDS"
ATTEMPTS_JSON="$OUT/attempts.jsonl"
: > "$ATTEMPTS_JSON"
CHOSEN=""
for att in $(seq 1 "$MAX_DEVICE_ATTEMPTS"); do
  [ -n "$REMAINING" ] || break
  PDIR="$OUT/preflight/attempt$att"
  mkdir -p "$PDIR"
  echo "=== AC-3 pre-flight device probe, attempt $att (candidates $REMAINING) $(date -Is) ==="
  bash "$ACC/opt/m3i7/scripts/gpu_guard_i7.sh" "$REMAINING" -- \
    bash "$ACC/harness/gate_ac3_stable.sh" --out "$PDIR" --python "$PY" \
      --reps "$PREFLIGHT_REPS" --batch-sizes 1 --max-extra 0 \
      --kernel-root "$PDIR/kernels" > "$PDIR/preflight.log" 2>&1
  prc=$?
  DEV="$(grep -oE 'physical GPUs used \(from each run.s own device UUID\): \[[0-9]+' \
         "$PDIR/preflight.log" | grep -oE '[0-9]+$' | head -1)"
  [ -n "$DEV" ] || DEV="$(grep -oE 'claiming it' -B4 "$PDIR/preflight.log" \
                          | grep -oE 'GPU [0-9]+' | tail -1 | grep -oE '[0-9]+')"
  RATE="$(JQ "$PDIR/gate_ac3_stable.json" totals.fingerprint_divergence_rate)"
  VERD="$(JQ "$PDIR/gate_ac3_stable.json" verdict)"
  printf '{"attempt":%d,"candidates":"%s","device":"%s","rc":%d,"verdict":"%s","fingerprint_divergence_rate":"%s"}\n' \
         "$att" "$REMAINING" "${DEV:-unknown}" "$prc" "${VERD:-none}" "${RATE:-none}" \
         >> "$ATTEMPTS_JSON"
  echo "  probe: device=${DEV:-unknown} rc=$prc verdict=${VERD:-none} divergence=${RATE:-none}"
  if [ "$prc" = "97" ]; then
    echo "  no candidate device was stable-idle; nothing to re-run onto."
    break
  fi
  # A clean probe = every rep fingerprint-consistent (gate verdict STABLE).  A
  # FAIL verdict here is a TOKEN difference vs dumps_final, which the re-pinned
  # AC-3 tolerates -- so it does NOT disqualify the device; only fingerprint
  # divergence (UNSTABLE, or a non-zero rate) sends us to a fresh device.
  if [ "${RATE:-0}" = "0.0" ] || [ "${RATE:-0}" = "0" ]; then
    CHOSEN="$DEV"
    echo "  device $CHOSEN accepted for the sweep (probe divergence rate ${RATE})"
    break
  fi
  echo "  device ${DEV:-unknown} diverged in the probe -- dropping it and retrying"
  [ -n "$DEV" ] && REMAINING="$(echo "$REMAINING" | tr ',' '\n' | grep -vx "$DEV" \
                                | paste -sd, -)"
done

if [ -z "$CHOSEN" ]; then
  echo "PROCEEDING WITHOUT A CERTIFIED-CLEAN DEVICE: every probed device showed"
  echo "fingerprint divergence (or none was idle).  Recorded in attempts.jsonl."
  echo "Under the re-pinned AC-3 divergence is in-band, so the sweep still runs;"
  echo "the rate is reported.  Falling back to the guard's own choice."
  CHOSEN=""
fi

SWEEP="$OUT/sweep"
mkdir -p "$SWEEP"
echo "=== AC-3 full cold sweep bs=$BSS reps=$REPS $(date -Is) ==="
if [ -n "$CHOSEN" ]; then GUARD_LIST="$CHOSEN"; else GUARD_LIST="$CANDS"; fi
bash "$ACC/opt/m3i7/scripts/gpu_guard_i7.sh" "$GUARD_LIST" -- \
  bash "$ACC/harness/gate_ac3_stable.sh" --out "$SWEEP" --python "$PY" \
    --reps "$REPS" --batch-sizes "$BSS" --kernel-root "$SWEEP/kernels" \
  > "$SWEEP/sweep.log" 2>&1
SRC=$?
echo "  gate_ac3_stable rc=$SRC (0 STABLE / 1 token-diff / 2 UNSTABLE / 3 integrity)"
tail -25 "$SWEEP/sweep.log"

NREPS=$(find "$SWEEP/reps" -maxdepth 2 -name 'bs*.json' 2>/dev/null | wc -l)
echo "  reps with a token dump: $NREPS"
if [ "$NREPS" -eq 0 ]; then
  echo "COLLECT FAILED: the sweep produced no token dump at all." >&2
  printf '{"stage":"ac3","ok":false,"why":"no rep produced a dump","gate_rc":%d}\n' "$SRC" \
    > "$OUT/collect_ac3.json"
  exit 3
fi

HFRC=0
if [ "$SKIP_HF" = "1" ]; then
  echo "=== HF coherence stage SKIPPED (--skip-hf): AC-3(a) perplexity/byte-soup"
  echo "    inputs will be missing and score_ac3.py will report NOT_EVALUABLE ==="
  HFRC=99
else
  echo "=== HF coherence stage (text + perplexity under the pinned HF model) $(date -Is) ==="
  bash "$ACC/opt/m3i7/scripts/gpu_guard_i7.sh" "$GUARD_LIST" -- \
    "$PY" -u "$ACC/final/hf_score.py" \
      --reference "$ACC/reference/reference_outputs.json" \
      --reps-root "$SWEEP/reps" --batch-sizes "$BSS" \
      --model "$MODEL" --revision "$REVISION" \
      --output-json "$OUT/coherence_inputs.json" > "$OUT/hf_score.log" 2>&1
  HFRC=$?
  echo "  hf_score rc=$HFRC"
  tail -12 "$OUT/hf_score.log"
fi

"$PY" - "$OUT/collect_ac3.json" "$SRC" "$HFRC" "$NREPS" "${CHOSEN:-}" \
       "$ATTEMPTS_JSON" <<'EOF'
import json, sys, time
out, src, hfrc, nreps, dev, attempts = sys.argv[1:7]
rows = []
try:
    rows = [json.loads(l) for l in open(attempts) if l.strip()]
except OSError:
    pass
json.dump({"stage": "ac3", "ok": True, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "gate_rc": int(src), "hf_score_rc": int(hfrc), "reps_with_dump": int(nreps),
           "device_chosen": dev or None, "preflight_attempts": rows},
          open(out, "w"), indent=2)
EOF
echo "=== COLLECT_AC3_DONE $(date -Is) ==="
exit 0
