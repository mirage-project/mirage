#!/bin/bash
# M3-I10 part C: Nsight Compute detail on the vLLM kernels MPK must beat.
set -uo pipefail
GPUID="$1"; shift
BS="$1"; shift
R="$HOME/mpk-qwen35/m3i10-profile"
mkdir -p "$R/ncu" "$R/logs"
exec > "$R/logs/ncu_bs${BS}.log" 2>&1
set -x
export CUDA_VISIBLE_DEVICES="$GPUID"
export HF_HOME="$HOME/mpk-qwen35/hf"
export PATH="$HOME/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TOKENIZERS_PARALLELISM=false
cd "$HOME/mpk-qwen35"
source venv-vllm/bin/activate

KRE="regex:(cutlass_3x_gemm_fp8_blockwise|per_token_group_quant_8bit_kernel|bmm_E4m3_E4m3E4m3|bmm_Bfloat16_E4m3E4m3|fused_recurrent_gated_delta_rule|activationDeepSeekKernel|causal_conv1d_update|fmhaSm100fKernel|finalizeKernel|routingIndices)"

ncu --target-processes all \
    --profile-from-start off \
    --kernel-name "$KRE" \
    --launch-count 260 \
    --section SpeedOfLight --section LaunchStats --section Occupancy \
    --section MemoryWorkloadAnalysis \
    --export "$R/ncu/bs${BS}" --force-overwrite \
    python3 "$R/scripts/ncu_probe.py" --batch-size "$BS" --n-steps 1 --start-step 12 --enforce-eager --max-num-batched-tokens 1024
EXIT=$?
echo "=== NCU_EXIT=$EXIT at $(date) ==="
ls -la "$R/ncu/"
exit $EXIT
