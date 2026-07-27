#!/usr/bin/env bash
# M3-I9b window: root-cause the admission-cap token flip at bs4.
#
# STAGE A (bs1, p10-logic alone) isolates the ONE variable the cap changes at
# bs1: the prefill chunk decomposition.  Uncapped bs1 prefills 43 tokens as
# [16, 16, 11]; cap=4 prefills them as [4]*10 + [3] -- which is byte-for-byte
# the SAME decomposition slot 3 gets in the bs4 cap arm (protocol_sim).  So if
# bs1 reproduces the flip, chunk boundaries alone are sufficient and
# co-residency is not part of the mechanism; if it does not, co-residency is.
#
# STAGE B (bs4 wave 1, the wave the flip lives in) re-runs the real geometry
# and dumps every persistent per-layer prefill state for the layer bisect.
#
# Kernel dirs carry the cap value: `max_tokens_per_request` is a compile-time
# -D and load_mpk_kernel's reuse path does NOT validate it (M3-I9 runner
# backfill), so sharing a dir across cap values silently loads the wrong kernel.
set -uo pipefail
M=$HOME/mpk-qwen35/m3i9b
REPO=$HOME/mpk-qwen35/mirage
ACC=$REPO/demo/qwen3_5/accept
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
SCRIPTS=${SCRIPTS:-$M/scripts}
PROBE=$SCRIPTS/probe_chunk_numerics.py
export MPK_ACCEPT_DIR=$ACC
CMP=$SCRIPTS/compare_arms.py
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
STAGES=${STAGES:-A,B}
CANDS=${CANDS:-6,5,4,3,2,1,0}
WAVE1=p07-format,p05-cuda,p08-science,p10-logic

mkdir -p "$M"/{logs,out,kernels,raw}
has() { case ",$STAGES," in *,$1,*) return 0;; *) return 1;; esac; }

GPU=$(bash "$SCRIPTS/claim_gpu.sh" "$CANDS") || { echo "no GPU"; exit 97; }
export CUDA_VISIBLE_DEVICES="$GPU"
echo "=== M3-I9b on GPU $GPU, clone $(git -C "$REPO" rev-parse --short HEAD) $(date -Is)"

# ---------------------------------------------------------------- stage A
if has A; then
  for arm in base cap4; do
    capflag=""; suffix=""
    [ "$arm" = "cap4" ] && { capflag="--cap 4"; suffix="_cap4"; }
    echo "=== stage A / bs1 / $arm $(date -Is)"
    "$PY" "$PROBE" --bs 1 --mbt 16 --compile-msl 108 $capflag \
      --kernel-dir "$M/kernels/bs1_msl108${suffix}" --reuse-kernel \
      --prompts p10-logic --target p10-logic \
      --positions 0:64:1 --state-at 0 \
      --dump-rows 0,48,49,50 \
      --raw-dir "$M/raw/bs1_${arm}" \
      --out "$M/out/bs1_${arm}.json" 2>&1 | tail -80
  done
  "$PY" "$CMP" --curves "$M/out/bs1_base.json" "$M/out/bs1_cap4.json" \
      --states "$M/raw/bs1_base/state.pt" "$M/raw/bs1_cap4/state.pt" \
      --out "$M/out/cmp_bs1.json" | tee "$M/logs/cmp_bs1.txt"
fi

# ---------------------------------------------------------------- stage B
if has B; then
  for arm in base cap4; do
    capflag=""; suffix=""
    [ "$arm" = "cap4" ] && { capflag="--cap 4"; suffix="_cap4"; }
    echo "=== stage B / bs4 wave1 / $arm $(date -Is)"
    "$PY" "$PROBE" --bs 4 --mbt 16 --compile-msl 132 $capflag \
      --kernel-dir "$M/kernels/bs4_msl132${suffix}" --reuse-kernel \
      --prompts "$WAVE1" --target p10-logic \
      --positions 0,5,10,15,20,25,30,35,40,43,45,46,47,48,49,50,55,63 \
      --state-hash --state-at 0 \
      --raw-dir "$M/raw/bs4_${arm}" \
      --out "$M/out/bs4_${arm}.json" 2>&1 | tail -40
  done
  "$PY" "$CMP" --curves "$M/out/bs4_base.json" "$M/out/bs4_cap4.json" \
      --states "$M/raw/bs4_base/state.pt" "$M/raw/bs4_cap4/state.pt" \
      --out "$M/out/cmp_bs4.json" | tee "$M/logs/cmp_bs4.txt"
fi

echo "=== M3-I9b done $(date -Is)"
rm -f "$HOME/mpk-qwen35/.gpu-locks/M3-I9b.lock"
