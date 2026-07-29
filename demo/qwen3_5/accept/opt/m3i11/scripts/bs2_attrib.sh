#!/usr/bin/env bash
# M3-I11: attribute the HEAD bs2/p08-science AC-3 difference.
#
# 2x2 design on ONE clean GPU, 3 reps per arm, identical command:
#   A  HEAD (c80ebd68)        = fence + attention pass-size 4->2 (a86b1eb1)
#   B  HEAD minus fence       = pass-size only
#   C  0cdd52f0 tree          = fence only        (clone m3i11b-fix, 170ab325+fence)
#   D  170ab325               = neither           (clone m3i11b-ctrl)
# The pass-size change is pure Python (models/qwen3_5/builder.py) and there is
# no src/ change anywhere in 170ab325..HEAD, so all four arms share one
# compiled core.so -- the only thing that varies is what we want to vary.
# usage: GPU=<id> bash bs2_attrib.sh
set -uo pipefail
GPU=${GPU:?}
B=$HOME/mpk-qwen35
PY=$B/venv-rm/bin/python
export HF_HOME=$B/hf PYTHONUNBUFFERED=1
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_VISIBLE_DEVICES=$GPU
M=$B/m3i11b-out/bs2attrib
mkdir -p "$M"

# arm B: HEAD with the fence hunk reverted
NF=$B/mirage-i11-nofence
if [ ! -d "$NF" ]; then
  echo "=== building arm B (HEAD minus fence) ==="
  cp -a "$B/mirage-i11" "$NF"
  python3 - "$NF" <<'PY'
import re, sys
p = sys.argv[1] + "/include/mirage/persistent_kernel/tasks/blackwell/linear_sm100_mpk.cuh"
s = open(p).read()
assert "kernel::tma::store_async_wait<0>();" in s, "fence not present"
s = s.replace("kernel::tma::store_async_wait<0>();", "cute::tma_store_wait<0>();")
open(p, "w").write(s)
print("reverted fence in", p)
PY
fi

echo "=== arm fingerprints ==="
for a in "A:$B/mirage-i11" "B:$NF" "C:$B/m3i11b-fix" "D:$B/m3i11b-ctrl"; do
  n=${a%%:*}; d=${a#*:}
  printf "  %s %s  fence=%s  pass_size=%s\n" "$n" "$(basename $d)" \
    "$(grep -c 'store_async_wait<0>' $d/include/mirage/persistent_kernel/tasks/blackwell/linear_sm100_mpk.cuh)" \
    "$(grep -c 'max_tokens_per_pass' $d/python/mirage/mpk/models/qwen3_5/builder.py 2>/dev/null || echo 0)"
done

run_arm() {  # name dir
  local n="$1" d="$2"
  echo "=== arm $n ($(basename $d)) $(date -Is) ==="
  cd "$d/demo/qwen3_5/accept"
  for r in 1 2 3; do
    local o="$M/${n}_rep$r"; rm -rf "$o"; mkdir -p "$o"
    PYTHONPATH=$d/python "$PY" -u mpk_engine_run.py --batch-size 2 --max-seq-length 132 \
       --out-dir "$o" --kernel-dir "$M/kd_$n" $( [ $r -gt 1 ] && echo --reuse-kernel ) \
       > "$M/${n}_rep$r.log" 2>&1
    echo "  $n rep$r rc=$? md5=$(md5sum $o/bs2.json 2>/dev/null | cut -d' ' -f1) $(date -Is)"
  done
}

run_arm A "$B/mirage-i11"
run_arm B "$NF"
run_arm C "$B/m3i11b-fix"
run_arm D "$B/m3i11b-ctrl"

echo "=== ATTRIBUTION TABLE (per-prompt token md5, bs2) ==="
"$PY" - "$M" "$B/m3i11b-ctrl/demo/qwen3_5/accept/results/dumps_final/bs2.json" <<'PY'
import hashlib, json, os, sys
M, basep = sys.argv[1], sys.argv[2]
base = json.load(open(basep))
def h(d, p):
    v = d[p]
    v = v.get("token_ids", v) if isinstance(v, dict) else v
    return hashlib.md5(json.dumps(v).encode()).hexdigest()[:8]
arms = {}
for a in "ABCD":
    reps = []
    for r in (1, 2, 3):
        f = os.path.join(M, f"{a}_rep{r}", "bs2.json")
        if os.path.exists(f):
            reps.append(json.load(open(f)))
    arms[a] = reps
prompts = sorted(base)
print(f"{'prompt':15s} {'baseline':9s} " + " ".join(f"{a}(x{len(arms[a])})".ljust(28) for a in "ABCD"))
for p in prompts:
    b = h(base, p)
    cells = []
    for a in "ABCD":
        hs = [h(r, p) for r in arms[a]]
        u = sorted(set(hs))
        cell = "/".join(u) + ("" if len(u) == 1 else " UNSTABLE")
        cells.append(cell.ljust(28))
    row = f"{p:15s} {b:9s} " + " ".join(cells)
    if any(h(r, p) != b for a in "ABCD" for r in arms[a]):
        row += "  <== DIFFERS"
    print(row)
print()
print("legend: A=HEAD(fence+passsize)  B=passsize only  C=fence only  D=neither")
PY
echo "=== BS2_ATTRIB_DONE $(date -Is) ==="
