#!/bin/bash

# This script profiles a specific CUDA kernel of a target application using NVIDIA Nsight Compute.
#
# Usage:
# ./analyze.sh <report_file.ncu-rep> <kernel_name> <command_to_profile...>
#
# Example:
# ./analyze.sh matmul.ncu-rep matmul_kernel python my_app.py

set -e
export MIRAGE_HOME=$(pwd)

function collect_ncu() {  
  local report_file=$1; shift
  local kernel_name=$1; shift
  local target_cmd=( "$@" )

  if [[ -n "$VIRTUAL_ENV" ]]; then
      source "$VIRTUAL_ENV/bin/activate"
  fi

  local ncu_path=$(which ncu 2>/dev/null)

  echo "Profiling kernel: $kernel_name"

  sudo env \
    MIRAGE_HOME="$MIRAGE_HOME" \
    TMPDIR="./ncu_tmp" \
    VIRTUAL_ENV="$VIRTUAL_ENV" \
    PATH="$VIRTUAL_ENV/bin:$PATH" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    "$ncu_path" \
      --set full \
      --force-overwrite \
      --target-processes all \
      --kernel-name "$kernel_name" \
      --kill no \
      --filter-mode global \
      --cache-control all \
      --clock-control base \
      --profile-from-start yes \
      --launch-count 1 \
      --apply-rules yes \
      --import-source no \
      --check-exit-code yes \
      --section MemoryWorkloadAnalysis_Chart \
      --section MemoryWorkloadAnalysis_Tables \
      --metrics "group:memory__chart,group:memory__shared_table,group:memory__first_level_cache_table,group:memory__l2_cache_table,group:memory__l2_cache_evict_policy_table,group:memory__dram_table" \
      --export "$report_file" \
      -- "${target_cmd[@]}"

  echo "-------------------------------------"
  echo "Profiling complete, report saved to $report_file"
}

collect_ncu "$@"