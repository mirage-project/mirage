#!/bin/bash
# Find idle GPUs (0% utilization and <=50% memory used) and output their
# indices as comma-separated list.
# Usage: ./scripts/find_idle_gpus.sh --num-gpus <N>

set -euo pipefail

NUM_GPUS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 --num-gpus <N>" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$NUM_GPUS" ]]; then
  echo "Usage: $0 --num-gpus <N>" >&2
  exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
  echo "Error: nvidia-smi not found." >&2
  exit 1
fi

IDLE_GPUS=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
                       --format=csv,noheader,nounits 2>/dev/null \
            | awk -F ', ' '$2 == 0 && $4 > 0 && $3 / $4 <= 0.5 {print $1}')

if [[ -z "$IDLE_GPUS" ]]; then
  echo "Error: no idle GPUs found." >&2
  exit 1
fi

IDLE_COUNT=$(echo "$IDLE_GPUS" | wc -l)

if [[ "$NUM_GPUS" -gt "$IDLE_COUNT" ]]; then
  echo "Error: need $NUM_GPUS idle GPUs, only $IDLE_COUNT available." >&2
  exit 1
fi

echo "$IDLE_GPUS" | head -n "$NUM_GPUS" | paste -sd, -
