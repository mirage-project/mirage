#!/bin/bash
# 3-sample stable-idle GPU guard. Prints the index of the first GPU that is
# idle across 3 samples ~4s apart, or exits 1. Usage: gpu_guard.sh [skip_csv]
set -uo pipefail
SKIP="${1:-}"
declare -A busy
for s in 1 2 3; do
  while IFS=, read -r idx mem util; do
    idx=$(echo "$idx" | tr -d ' ')
    mem=$(echo "$mem" | tr -d ' MiB')
    util=$(echo "$util" | tr -d ' %')
    if [ "$mem" -gt 200 ] || [ "$util" -gt 5 ]; then busy[$idx]=1; fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader)
  [ "$s" -lt 3 ] && sleep 4
done
for i in 0 1 2 3 4 5 6 7; do
  case ",$SKIP," in *",$i,"*) continue;; esac
  if [ -z "${busy[$i]:-}" ]; then echo "$i"; exit 0; fi
done
echo "NO_FREE_GPU" >&2
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv >&2
exit 1
