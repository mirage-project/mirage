#!/usr/bin/env bash
# M3-I9b stage C: REPRODUCTION CONTROL.
#
# Stage A/B found the cap bit-transparent -- same tokens, same full-vocab logit
# rows, same per-layer GDN/KV state -- at bs1 AND at bs4 wave 1, when the wave
# is run ALONE from a fresh runtime init.  The M3-I9 window saw p10-logic flip
# at bs4.  The two runs differ in exactly two things:
#
#   C1  the window ran ALL THREE bs4 waves in ONE process, with only
#       init_request_func() between them (mpk_engine_run._reset_runtime).  That
#       resets step / request_ids / qo_indptr / the page queue -- NOT the GDN
#       conv+recurrent state pools and NOT the KV cache.
#   C2  the probe exposes the lm-head buffer (attach_input instead of
#       new_tensor), which changes where argmax_in lives.
#
# C1 = the window's exact command on the window's own compiled kernel.
# C2 = the same command restricted to wave 1, so the ONLY delta vs C1 is the
#      two waves that ran before it.
# C3 = the uncapped control, same shape as C1.
set -uo pipefail
M=$HOME/mpk-qwen35/m3i9b
M9=$HOME/mpk-qwen35/m3i9
REPO=$HOME/mpk-qwen35/mirage
ACC=$REPO/demo/qwen3_5/accept
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
SCRIPTS=${SCRIPTS:-$M/scripts}
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
WAVE1=p07-format,p05-cuda,p08-science,p10-logic
CANDS=${CANDS:-6,5,4,3,2,1,0}
STAGES=${STAGES:-C1,C2,C3,C4}

mkdir -p "$M"/{logs,out}
has() { case ",$STAGES," in *,$1,*) return 0;; *) return 1;; esac; }
GPU=$(bash "$SCRIPTS/claim_gpu.sh" "$CANDS") || { echo "no GPU"; exit 97; }
export CUDA_VISIBLE_DEVICES="$GPU"
echo "=== M3-I9b stage C on GPU $GPU $(date -Is)"

run() {  # run <tag> <kernel-dir> [extra args]
  local tag=$1 kdir=$2; shift 2
  echo "=== $tag $(date -Is)"
  "$PY" "$ACC/mpk_engine_run.py" \
    --batch-size 4 --max-seq-length 132 --max-new-tokens 64 \
    --reference "$ACC/reference/reference_outputs.json" \
    --kernel-dir "$kdir" --reuse-kernel \
    --out-dir "$M/out/$tag" --dump-name bs4.json "$@" \
    > "$M/logs/$tag.log" 2>&1
  echo "   rc=$? -> $M/out/$tag/bs4.json"
}

has C1 && run c1_cap_3waves  "$M9/kernels/bs4_msl132_cap4" --per-request-token-cap auto
has C2 && run c2_cap_wave1   "$M9/kernels/bs4_msl132_cap4" --per-request-token-cap auto --prompt-ids "$WAVE1"
has C3 && run c3_base_3waves "$M9/kernels/bs4_msl132"
has C4 && run c4_base_wave1  "$M9/kernels/bs4_msl132" --prompt-ids "$WAVE1"

"$PY" - <<'EOF' | tee "$M/logs/cmp_C.txt"
import json, os
from pathlib import Path
M = Path(os.environ["HOME"]) / "mpk-qwen35/m3i9b"
ACC = Path(os.environ["HOME"]) / "mpk-qwen35/mirage/demo/qwen3_5/accept"
ref = json.load(open(ACC / "reference/reference_outputs.json"))["results"]
for tag in ("c1_cap_3waves", "c2_cap_wave1", "c3_base_3waves", "c4_base_wave1"):
    p = M / "out" / tag / "bs4.json"
    if not p.exists():
        print(f"{tag}: MISSING"); continue
    d = json.load(open(p))
    print(f"--- {tag} ({len(d)} prompts)")
    for pid, v in sorted(d.items()):
        r = ref[pid]["output_ids"]
        got = v["token_ids"]
        div = next((i for i, (a, b) in enumerate(zip(r, got)) if a != b), None)
        if div is not None:
            print(f"    {pid}: first divergence pos {div} "
                  f"ref={r[div]} got={got[div]} "
                  f"refmargin={ref[pid]['topk_logits_per_step'][div][0] - ref[pid]['topk_logits_per_step'][div][1]}")
    print("    all-match:",
          all(json.load(open(p))[pid]["token_ids"] == ref[pid]["output_ids"]
              for pid in d))
EOF
echo "=== stage C done $(date -Is)"
rm -f "$HOME/mpk-qwen35/.gpu-locks/M3-I9b.lock"
