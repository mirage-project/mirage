#!/usr/bin/env bash
# M4-status: 6-rep re-collection of BOTH AC-4/AC-5 arms.
#
# WHY.  The 3-rep pass produced valid criteria verdicts but four arms above the
# protocol's 5% dispersion bound: mpk full bs1 8.30%, bs4 7.27%, bs8 14.63%,
# bs16 5.35%, mpk pre bs1 13.23%, and the fresh vLLM sweep's own dispersion
# check failed at bs8.  score_perf.py's documented remedy is to ADD reps, never
# to drop one (protocol 6, corrected 2026-07-25): at n>=6 the statistic switches
# from full range to IQR/median, so a single co-tenancy excursion stops
# dominating.  bs4 and bs8 both trended monotonically upward across reps while
# foreign load on GPUs 0 and 4 grew during the same window, which is a
# co-tenancy signature rather than a property of the tree -- exactly the case
# the escalation exists for.
#
# This cannot turn a FAIL into a PASS: the deficit is 1.6-1.8x, not a dispersion
# artefact.  Both datasets are retained.
#
# Kernel dirs are REUSED from the 3-rep pass so every rep runs the identical
# binary the first pass measured (and no rep pays a cold compile).
set -uo pipefail
export MPK_BOX_ROOT=/home/muhengl/mpk-qwen35
ACC=/home/muhengl/mpk-qwen35/final-gate/tree-6741b4ad8aae/demo/qwen3_5/accept
REMOTE_RUN=/home/muhengl/mpk-qwen35/final-gate/run-20260730T101424Z
CANDS="${CANDS:-5,2,7,6,1,0,3,4}"

echo "########## M4-status 6-rep re-collect $(date -Is) ##########"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader

echo "### MPK arms, 6 reps ###"
bash "$ACC/opt/m3i7/scripts/gpu_guard_i7.sh" "$CANDS" -- \
  bash "$ACC/final/collect_perf.sh" --out "$REMOTE_RUN/perf/mpk6" \
    --accept-dir "$ACC" --reps 6 --batch-sizes "1,2,4,8,16" \
    --input-len 256 --max-new-tokens 1024 \
    --kernel-root "$REMOTE_RUN/perf/mpk/kernels"
MRC=$?
echo "MPK6_RC=$MRC"

echo "### fresh vLLM comparator, 6 reps ###"
bash "$ACC/opt/m3i7/scripts/gpu_guard_i7.sh" "$CANDS" -- \
  bash "$ACC/final/collect_vllm.sh" --out "$REMOTE_RUN/perf/vllm_fresh6" \
    --accept-dir "$ACC" --reps 6 --batch-sizes "1,2,4,8,16" \
    --pinned-baseline "$ACC/baselines/vllm-0.25.1-20260725"
VRC=$?
echo "VLLM6_RC=$VRC"
echo "########## M4-status 6-rep re-collect DONE mpk=$MRC vllm=$VRC $(date -Is) ##########"
