#!/usr/bin/env bash
# M3-I11 E1: the in-process-replica discriminator.
#
# THREE processes, each decoding EIGHT byte-identical copies of p03-python as
# eight separate single-slot waves (bs=1, msl=1280, 1024 new tokens, warm cached
# kernel -- the exact .so the m3i9 s7 runs loaded).  The eight waves inside one
# process share one CUDA context, one set of device allocations, one compiled
# kernel, the same weights and the same launch config; they differ only in when
# they run.  So:
#   all 24 identical        -> not reproducible at this config
#   within-process equal,
#     across-process differ -> per-PROCESS constant perturbation  (H1 family:
#                              uninitialised memory / allocation-dependent)
#   within-process differ   -> per-STEP stochastic perturbation   (H3 family:
#                              order-dependent reduction)
#   wave0 odd, 1..7 equal   -> first-wave / residue effect
#
# NOTE the cap flag is deliberately NOT passed.  At bs=1, "auto" resolves to
# max(1, mbt//bs) = 16 = MPK_MAX_NUM_BATCHED_TOKENS, i.e. the compile-time
# default, and stage 7 pointed both arms at the SAME --kernel-dir
# (kernels/bs1_msl1280), so the s7 "base" and "cap" bs1 runs executed the
# identical binary on identical inputs.  Six such runs produced four distinct
# trajectories; this is that experiment with the arms collapsed.
set -uo pipefail
M=$HOME/mpk-qwen35/m3i11
REPO=$HOME/mpk-qwen35/mirage
ACC=$REPO/demo/qwen3_5/accept
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
KDIR=$HOME/mpk-qwen35/m3i9/kernels/bs1_msl1280
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU=${GPU:?set GPU}
export CUDA_VISIBLE_DEVICES=$GPU
NPROC=${NPROC:-3}

mkdir -p "$M/out" "$M/logs"
echo "kernel dir fingerprint:"; md5sum "$KDIR"/* | sed 's|.*/||'
for p in $(seq 1 "$NPROC"); do
  echo "=== E1 process $p on GPU $GPU $(date -Is)"
  "$PY" "$ACC/mpk_engine_run.py" \
      --batch-size 1 --max-seq-length 1280 --max-new-tokens 1024 \
      --reference "$M/ref_p03x8.json" \
      --kernel-dir "$KDIR" --reuse-kernel \
      --out-dir "$M/out/e1_p$p" --dump-name "bs1.json" \
      > "$M/logs/e1_p$p.log" 2>&1
  echo "rc=$? $(date -Is)"
done
echo "=== E1 done $(date -Is)"
