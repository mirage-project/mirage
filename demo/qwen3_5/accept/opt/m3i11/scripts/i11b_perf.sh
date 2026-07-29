#!/usr/bin/env bash
# M3-I11 campaign 2 perf gate: CTRL and FIX interleaved in the SAME window on
# the SAME GPU with warm (reused) kernels, >=3 measured reps per arm per batch
# size. Interleaving is the same-window control -- box load drifts, so
# arm-vs-arm must be compared rep-adjacent, not block-vs-block.
# usage: GPU=<id> REPS=<n> BSLIST="4 16" bash i11b_perf.sh
set -uo pipefail
GPU=${GPU:?}
REPS=${REPS:-3}
BSLIST=${BSLIST:-"4 16"}
B=$HOME/mpk-qwen35
PY=$B/venv-rm/bin/python
export HF_HOME=$B/hf
export PYTHONUNBUFFERED=1
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_VISIBLE_DEVICES=$GPU
M=$B/m3i11b-out/perf
mkdir -p "$M/logs" "$M/out" "$M/kernels"

run_one() {  # arm bs tag
  local arm="$1" bs="$2" tag="$3" fresh="$4"
  local mirage=$B/m3i11b-$arm
  local kd="$M/kernels/warm_${arm}_bs${bs}"
  local extra=""
  [ "$fresh" = "fresh" ] && { rm -rf "$kd"; extra="--fresh-compile"; }
  ACC=$mirage/demo/qwen3_5/accept PYTHONPATH=$mirage/python \
    "$PY" "$mirage/demo/qwen3_5/accept/opt/m3i11/scripts/e4_full.py" \
      --out "$M/out" --tag "$tag" --bs "$bs" --kernel-dir "$kd" $extra \
      > "$M/logs/${tag}.log" 2>&1
  echo "  $tag rc=$? $(grep -o 'md5=[0-9a-f]*' "$M/logs/${tag}.log" | tail -1) $(date -Is)"
}

for bs in $BSLIST; do
  echo "=== bs=$bs warm-up compiles $(date -Is) ==="
  run_one ctrl "$bs" "warmup_ctrl_bs${bs}" fresh
  run_one fix  "$bs" "warmup_fix_bs${bs}"  fresh
  for r in $(seq 1 "$REPS"); do
    echo "=== bs=$bs interleaved rep $r $(date -Is) ==="
    run_one ctrl "$bs" "perf_ctrl_bs${bs}_r${r}" warm
    run_one fix  "$bs" "perf_fix_bs${bs}_r${r}"  warm
  done
done

echo "=== PERF SUMMARY $(date -Is) ==="
"$PY" - "$M/out" <<'PY'
import glob, json, os, statistics, sys, re, collections
d = sys.argv[1]
rows = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(d, "meta_perf_*.json"))):
    m = json.load(open(f))
    tag = m["tag"]
    g = re.match(r"perf_(ctrl|fix)_bs(\d+)_r(\d+)", tag)
    if not g:
        continue
    arm, bs, rep = g.group(1), int(g.group(2)), int(g.group(3))
    # per-wave decode cost; the run-level figure is the token-weighted mean
    tot_ms = sum(t["wall_ms"] for t in m["timings"])
    tot_steps = sum(t["max_decode_steps"] for t in m["timings"])
    rows[(bs, arm)].append((rep, tot_ms / tot_steps, m["dump_md5"]))
for bs in sorted({b for b, _ in rows}):
    print(f"\nbs={bs}")
    for arm in ("ctrl", "fix"):
        v = sorted(rows.get((bs, arm), []))
        if not v:
            continue
        ms = [x[1] for x in v]
        print(f"  {arm:4s} ms/step per rep: "
              + " ".join(f"{x:.4f}" for x in ms)
              + f"   median={statistics.median(ms):.4f}")
        print(f"       md5: {sorted({x[2] for x in v})}")
    c = rows.get((bs, "ctrl"), []); f_ = rows.get((bs, "fix"), [])
    if c and f_:
        mc, mf = statistics.median([x[1] for x in c]), statistics.median([x[1] for x in f_])
        print(f"  median delta FIX vs CTRL: {(mf-mc)/mc*100:+.2f}%")
        pairs = [(b[1]-a[1])/a[1]*100 for a, b in zip(sorted(c), sorted(f_))]
        print(f"  rep-paired deltas: " + " ".join(f"{x:+.2f}%" for x in pairs)
              + f"   median={statistics.median(pairs):+.2f}%")
PY
echo "=== PERF_DRIVER_DONE $(date -Is) ==="
