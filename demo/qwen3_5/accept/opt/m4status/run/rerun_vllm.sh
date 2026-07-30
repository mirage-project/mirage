#!/usr/bin/env bash
# M4-status: re-run ONLY the gate's fresh vLLM comparator.
#
# WHY.  The gate's perf stage collected all 30 MPK cells cleanly (30/30 ok,
# foreign floor 62 MiB) and then the fresh vLLM half died before its first rep:
# bench_vllm.py's own preflight_gpu_check refused GPU 2 because a foreign
# process (1004 MiB, /home/bohanhou/my/bin/python) landed on the device between
# the guard's claim and the preflight.  That is the etiquette rule working, not
# a measurement result -- so the comparator is re-collected on a device that is
# exclusively free, and the run is then re-scored with final.sh --rescore.
#
# Same script, same flags, same pinned baseline as the gate's own invocation:
# nothing about the comparator's definition changes.
set -uo pipefail
export MPK_BOX_ROOT=/home/muhengl/mpk-qwen35
ACC=/home/muhengl/mpk-qwen35/final-gate/tree-6741b4ad8aae/demo/qwen3_5/accept
REMOTE_RUN=/home/muhengl/mpk-qwen35/final-gate/run-20260730T101424Z
CANDS="${CANDS:-5,2,7,6,1}"

echo "########## M4-status fresh vLLM re-collect $(date -Is) ##########"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
bash "$ACC/opt/m3i7/scripts/gpu_guard_i7.sh" "$CANDS" -- \
  bash "$ACC/final/collect_vllm.sh" --out "$REMOTE_RUN/perf/vllm_fresh" \
    --accept-dir "$ACC" --reps 3 --batch-sizes "1,2,4,8,16" \
    --pinned-baseline "$ACC/baselines/vllm-0.25.1-20260725"
echo "VLLM_RECOLLECT_RC=$?"
echo "########## M4-status fresh vLLM re-collect DONE $(date -Is) ##########"
