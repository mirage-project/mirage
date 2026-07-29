#!/usr/bin/env bash
# gate_ac3_stable.sh -- the fingerprint-scored COLD-run AC-3 correctness gate.
#
# CONTRACT
# --------
#   Runs the AC-3 sweep (10 pinned reference prompts, msl 132, 64 new tokens) at
#   every batch size in the AC-3 set, with a COLD kernel compile for every rep --
#   the class that M3-I11 campaign 2 measured at 10-16% divergence per rep -- and
#   >=REPS independent reps (separate processes, fresh CUDA context) per bs.
#   Each rep captures the KV/GDN wave-boundary fingerprint alongside its token
#   dump.
#
#   PASS requires BOTH:
#     (a) every rep's token ids byte-identical, per case, to the committed
#         baseline results/dumps_final/bs<N>.json -- AC-3 itself, unrelaxed; and
#     (b) REPS reps per bs whose KV/GDN fingerprints are identical to each other,
#         key for key.
#
#   Any rep whose fingerprint deviates from the per-bs consensus is QUARANTINED
#   (kept in the record, never deleted), an extra rep is launched in its place,
#   and the observed divergence RATE is reported. The gate cannot silently pass a
#   run whose trajectory diverged: fingerprint divergence with clean tokens is
#   still counted, reported, and re-run, and fingerprint divergence that reaches
#   the tokens is a hard FAIL.
#
#   Output: $OUT/gate_ac3_stable.json -- per-rep fingerprints and devices,
#   per-case token verdicts, quarantine/divergence counts, the reps needed per
#   bs, and the STABLE/UNSTABLE/FAIL verdict. Exit 0 STABLE, 1 FAIL (tokens),
#   2 UNSTABLE (could not reach REPS fingerprint-identical reps), 3 integrity.
#
# WHY (docs/qwen35/bench-protocol.md, "Determinism protocol v2")
#   Token md5 is a ~2%-sensitive detector for this engine's nondeterminism; the
#   KV/GDN fingerprint is ~100%. Campaign 2's fix arm read 0/59 clean by md5 and
#   6/59 divergent by fingerprint. A gate scored on tokens alone reports a clean
#   number for a run whose arithmetic was not reproducible.
#
# USAGE
#   CUDA_VISIBLE_DEVICES must already be pinned to ONE stable-idle device -- use
#   the repo's guard, e.g.
#     bash opt/m3i7/scripts/gpu_guard_i7.sh 6,3,1,0,2,7,4,5 -- \
#       bash harness/gate_ac3_stable.sh --out /path/to/out
#
#   flags (all optional except --out):
#     --out DIR                 artifact root (reps/, logs/, kernels/, report)
#     --reps N                  fingerprint-consistent reps required per bs [3]
#     --batch-sizes a,b,c       [1,2,4,8,16]
#     --max-extra K             extra reps allowed per bs to replace quarantined
#                               ones before declaring UNSTABLE [3]
#     --baseline DIR            [<accept>/results/dumps_final]
#     --kernel-root DIR         cold kernel scratch [<out>/kernels]
#     --python PATH             interpreter [$PY, else python3]
#     --per-request-token-cap V policy | none | auto | <int>, passed through to
#                               admission_policy.py, which OWNS the policy.
#                               Default 'policy' = whatever the runtime ships, so
#                               the gate certifies the shipped configuration.
#                               'none' forces the pre-policy uncapped runtime,
#                               which is how results/dumps_final was produced;
#                               the cap was measured bit-transparent at bs4/8/16
#                               (M3-I7) and re-verified at all five bs (M4-I4).
#     --keep-kernels            do not delete each rep's kernel dir (needs
#                               ~110 MiB per rep; off by default, /raid is tight)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACC="$(cd "$HERE/.." && pwd)"

REPS=3
BATCH_SIZES="1,2,4,8,16"
MAX_EXTRA=3
OUT=""
BASELINE="$ACC/results/dumps_final"
KERNEL_ROOT=""
PY="${PY:-python3}"
# 'policy' delegates to admission_policy.py -- the gate certifies the SHIPPED
# configuration. Never hardcode a per-batch-size cap decision here.
CAP="policy"
KEEP_KERNELS=0
DRAIN_TRIES="${DRAIN_TRIES:-60}"
DRAIN_SLACK_MIB="${DRAIN_SLACK_MIB:-600}"
DRAIN_STABLE_MIB="${DRAIN_STABLE_MIB:-64}"

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    --reps) REPS="$2"; shift 2;;
    --batch-sizes) BATCH_SIZES="$2"; shift 2;;
    --max-extra) MAX_EXTRA="$2"; shift 2;;
    --baseline) BASELINE="$2"; shift 2;;
    --kernel-root) KERNEL_ROOT="$2"; shift 2;;
    --python) PY="$2"; shift 2;;
    --per-request-token-cap) CAP="$2"; shift 2;;
    --keep-kernels) KEEP_KERNELS=1; shift;;
    -h|--help) sed -n '1,60p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "unknown flag: $1" >&2; exit 3;;
  esac
done
[ -n "$OUT" ] || { echo "gate_ac3_stable.sh: --out is required" >&2; exit 3; }
[ -n "$KERNEL_ROOT" ] || KERNEL_ROOT="$OUT/kernels"
mkdir -p "$OUT/reps" "$OUT/logs" "$KERNEL_ROOT"

CAPARG=(--per-request-token-cap "$CAP")
POLICY_JSON="$("$PY" "$ACC/admission_policy.py" 2>/dev/null | tr -d '\n' | sed 's/  */ /g')"
[ -n "$POLICY_JSON" ] || POLICY_JSON='{"error":"admission_policy.py unreadable"}'

# ---------------------------------------------------------------------------
# the pinned device. CUDA_VISIBLE_DEVICES is the CLAIM; each rep re-derives the
# truth from its own CUDA context's device UUID and records it in its meta. If
# the two ever disagree the driver adopts the rep's value and says so loudly --
# M3-I7 shipped a table labelled with a candidate-list index while the guard had
# claimed a different device.
# ---------------------------------------------------------------------------
PHYS="${CUDA_VISIBLE_DEVICES:-}"
case "$PHYS" in
  ''|*,*|*GPU-*) echo "[gate] NOTE: CUDA_VISIBLE_DEVICES='$PHYS' is not a single"\
                      "index; the drain gate is disabled until rep 1 reports the"\
                      "device from its own context."; PHYS="";;
esac

gpu_row() {  # $1 = physical index -> "used util"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
             -i "$1" 2>/dev/null | tr -d ' ' | tr ',' ' '
}

FLOOR="${MPK_I7_GPU_FLOOR:-}"
if [ -n "$PHYS" ] && [ -z "$FLOOR" ]; then
  read -r FLOOR _ < <(gpu_row "$PHYS")
  FLOOR="${FLOOR:-0}"
fi
FLOOR="${FLOOR:-0}"

drain_gate() {
  # Wait until the pinned device has actually released the PREVIOUS rep's
  # memory. M3-I6a caught a 34 GB teardown still in flight producing a fake 2.1x
  # regression; a cold-compile stability census is exactly as vulnerable, since
  # a co-tenant footprint is one of the candidate mechanisms for the divergence
  # this gate is measuring.
  #
  # TWO conditions, checked over 3 samples 3 s apart:
  #   1. util <= 5% on every sample -- no live work on the device (SM-RESIDENCY
  #      LAW: MPK's megakernel claims every SM and spin-waits, so a co-tenant
  #      block is a self-sustaining deadlock, not just noise).
  #   2. resident memory either under floor+slack (the fast path) or STABLE
  #      across the samples. A FALLING reading is a previous tenant still
  #      tearing down -- that is the M3-I6a mechanism and the thing worth
  #      waiting for.
  # Condition 2's stability branch exists because the absolute test alone is
  # unsatisfiable when a foreign user parks an IDLE CUDA context on the device:
  # ~920 MiB on GPUs 2-6 on 2026-07-28 (gpu_guard_i7.sh's MAXMEM note) and again
  # ~940 MiB on 2026-07-29 during this gate's own campaign, which stalled every
  # rep for the full DRAIN_TRIES window. An idle context launches no blocks, so
  # it does not violate the SM-residency law. What it does do is make "the
  # device was clean" unprovable -- which is why every rep records gpu_before
  # AND gpu_after and the report flags any rep whose device GREW mid-run.
  local phys="$1" used util i s ok lo hi
  [ -n "$phys" ] || { echo "[gate] drain: physical index unknown, skipping"; return 0; }
  for i in $(seq 1 "$DRAIN_TRIES"); do
    ok=1; lo=99999999; hi=-1
    for s in 1 2 3; do
      read -r used util < <(gpu_row "$phys")
      used="${used:-99999999}"; util="${util:-100}"
      [ "$util" -gt 5 ] && { ok=0; break; }
      [ "$used" -lt "$lo" ] && lo="$used"
      [ "$used" -gt "$hi" ] && hi="$used"
      [ "$s" -lt 3 ] && sleep 3
    done
    if [ "$ok" -eq 1 ] && { [ "$hi" -le "$((FLOOR + DRAIN_SLACK_MIB))" ] \
                            || [ "$((hi - lo))" -le "$DRAIN_STABLE_MIB" ]; }; then
      echo "[gate] drained: GPU $phys used=${lo}..${hi}MiB util=${util}% (floor ${FLOOR}MiB, stable within ${DRAIN_STABLE_MIB}MiB)"
      return 0
    fi
    echo "[gate] draining GPU $phys: used=${lo}..${hi}MiB util=${util}% (floor ${FLOOR}+${DRAIN_SLACK_MIB}) -- wait ($i/$DRAIN_TRIES)"
    sleep 10
  done
  echo "[gate] REFUSING: drain gate timed out on GPU $phys (used=${lo}..${hi}MiB util=${util}%)."
  echo "[gate] The rep will NOT be launched. MPK's megakernel claims every SM and"
  echo "[gate] spin-waits, so starting it next to live foreign work is a"
  echo "[gate] self-sustaining deadlock, not just noise (SM-RESIDENCY LAW, M3-I2a)."
  echo "[gate] Seen for real on 2026-07-29: a 142 GB foreign job landed on a"
  echo "[gate] claimed device mid-campaign. Failing the rep closed is correct --"
  echo "[gate] the gate reports 'could not obtain a clean window', which is a"
  echo "[gate] true statement, where proceeding would have produced a hang or an"
  echo "[gate] OOM dressed up as a stability result."
  return 1
}

SHA="$(git -C "$ACC" rev-parse HEAD 2>/dev/null || echo unknown)"
cat > "$OUT/run_meta.json" <<EOF
{
  "driver": "gate_ac3_stable.sh",
  "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "host": "$(hostname)",
  "git_sha": "$SHA",
  "git_dirty": $( [ -n "$(git -C "$ACC" status --porcelain 2>/dev/null)" ] && echo true || echo false ),
  "cuda_visible_devices_claim": "${CUDA_VISIBLE_DEVICES:-}",
  "gpu_floor_mib": $FLOOR,
  "reps_required": $REPS,
  "max_extra": $MAX_EXTRA,
  "batch_sizes": "$BATCH_SIZES",
  "per_request_token_cap": "${CAP:-none}",
  "admission_policy": $POLICY_JSON,
  "baseline": "$BASELINE",
  "gate_py_sha256": "$(sha256sum "$HERE/gate_ac3_stable.py" | cut -d' ' -f1)",
  "gate_sh_sha256": "$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)"
}
EOF
echo "=== gate_ac3_stable  sha=$SHA  reps=$REPS  bs=$BATCH_SIZES  $(date -Is) ==="
cat "$OUT/run_meta.json"

accepted_for_bs() {  # $1 = bs -> count of fingerprint-consensus reps so far
  "$PY" "$HERE/gate_ac3_stable.py" score \
      --reps-root "$OUT/reps" --baseline "$BASELINE" \
      --batch-sizes "$1" --reps "$REPS" --max-extra "$MAX_EXTRA" \
      --output-json "$OUT/logs/partial_bs$1.json" \
      > "$OUT/logs/partial_bs$1.log" 2>&1
  "$PY" -c "
import json,sys
d=json.load(open(sys.argv[1]))
b=d['per_bs'][sys.argv[2]]
print(b['accepted'], b['quarantined'], b['errors'])
" "$OUT/logs/partial_bs$1.json" "$1" 2>/dev/null || echo "0 0 0"
}

for BS in ${BATCH_SIZES//,/ }; do
  echo ""
  echo "=== bs=$BS  need $REPS fingerprint-consistent cold reps  $(date -Is) ==="
  r=0
  MAX_ATTEMPTS=$((REPS + MAX_EXTRA))
  while [ "$r" -lt "$MAX_ATTEMPTS" ]; do
    r=$((r + 1))
    TAG="bs${BS}_r${r}"
    REPDIR="$OUT/reps/$TAG"
    # Append-only launch ledger, written BEFORE the rep dir exists. On
    # 2026-07-29 the shared /raid pool hit 0 bytes mid-campaign and one rep's
    # `mkdir -p` itself failed, so the rep left no directory and no meta and
    # vanished from the record -- silently shrinking the denominator of the
    # divergence rate. The scorer reconciles this ledger against the directories
    # it finds and reports any missing tag as a LOST rep.
    echo "$TAG" >> "$OUT/launched.txt"
    mkdir -p "$REPDIR"
    if ! drain_gate "$PHYS"; then
      printf '{"tag":"%s","status":"error","note":"drain gate timed out; rep NOT launched (device not idle)"}\n' \
             "$TAG" > "$REPDIR/meta_$TAG.json"
      echo "  $TAG recorded as a run error (no clean window)  $(date -Is)"
      continue
    fi
    KD="$KERNEL_ROOT/cold_$TAG"
    rm -rf "$KD"
    echo "--- $TAG cold compile + run $(date -Is) ---"
    "$PY" -u "$HERE/gate_ac3_stable.py" rep \
        --out "$REPDIR" --tag "$TAG" --bs "$BS" --rep "$r" \
        --kernel-dir "$KD" "${CAPARG[@]+"${CAPARG[@]}"}" \
        > "$OUT/logs/$TAG.log" 2>&1
    rc=$?
    [ "$KEEP_KERNELS" -eq 0 ] && rm -rf "$KD"
    if [ "$rc" -ne 0 ] && [ ! -f "$REPDIR/meta_$TAG.json" ]; then
      printf '{"tag":"%s","status":"error","rc":%d,"note":"rep process exited rc=%d"}\n' \
             "$TAG" "$rc" "$rc" > "$REPDIR/meta_$TAG.json"
    fi
    echo "  rc=$rc  $(grep -o 'md5=[0-9a-f]*' "$OUT/logs/$TAG.log" | tail -1)  $(date -Is)"
    [ "$rc" -ne 0 ] && tail -8 "$OUT/logs/$TAG.log" | sed 's/^/    /'

    # adopt the device the rep itself reports (see the M3-I7 note above)
    if [ -f "$REPDIR/meta_$TAG.json" ]; then
      TRUE_PHYS="$("$PY" -c "
import json,sys
try: print(json.load(open(sys.argv[1]))['device']['phys_index'])
except Exception: print('')
" "$REPDIR/meta_$TAG.json" 2>/dev/null)"
      if [ -n "$TRUE_PHYS" ] && [ "$TRUE_PHYS" != "$PHYS" ]; then
        echo "[gate] DEVICE CORRECTION: the run reports physical GPU $TRUE_PHYS,"\
             "CUDA_VISIBLE_DEVICES claimed '${PHYS:-unset}'. Adopting $TRUE_PHYS."
        PHYS="$TRUE_PHYS"
        read -r FLOOR _ < <(gpu_row "$PHYS"); FLOOR="${FLOOR:-0}"
      fi
    fi

    read -r ACC_N QUAR_N ERR_N <<< "$(accepted_for_bs "$BS")"
    echo "  bs=$BS so far: accepted=$ACC_N quarantined=$QUAR_N errors=$ERR_N (need $REPS)"
    [ "${ACC_N:-0}" -ge "$REPS" ] && break
    df -h "$KERNEL_ROOT" 2>/dev/null | tail -1
  done
done

echo ""
echo "=== scoring $(date -Is) ==="
"$PY" "$HERE/gate_ac3_stable.py" score \
    --reps-root "$OUT/reps" --baseline "$BASELINE" \
    --batch-sizes "$BATCH_SIZES" --reps "$REPS" --max-extra "$MAX_EXTRA" \
    --run-meta "$OUT/run_meta.json" \
    --output-json "$OUT/gate_ac3_stable.json"
RC=$?
echo "=== GATE_AC3_STABLE_DONE rc=$RC $(date -Is) ==="
exit $RC
