#!/usr/bin/env bash
# Scan for truly-available GPUs on this machine, per user's criteria
# (2026-05-13):
#   1. util < 1%
#   2. memory < 500 MiB
#   3. no other users' processes on the card
#
# Usage:
#   bash scratch/scan_free_gpus.sh            # prints free GPU indices, one per line
#   bash scratch/scan_free_gpus.sh count      # prints just the count
#   bash scratch/scan_free_gpus.sh wait N M   # poll every N seconds (max M polls)
#                                                until >= 4 free GPUs found

set -euo pipefail

ME=$(id -un)

scan_once() {
    # nvidia-smi gives 3 columns: index, mem_used (MiB), util (%)
    local mem_thresh_mib=500
    local util_thresh_pct=1
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
               --format=csv,noheader,nounits 2>/dev/null \
    | awk -v memT="$mem_thresh_mib" -v utilT="$util_thresh_pct" -F', *' \
          '{ if ($2 < memT && $3 < utilT) print $1 }' \
    | while read -r idx; do
        # Check that no process from ANYONE else uses this index.
        # nvidia-smi pmon -s u prints one row per process: gpu, pid, type, ...
        # If any pid on this GPU is NOT owned by $ME, treat the GPU as busy.
        # Use --query-compute-apps to map gpu_uuid -> pid -> owner.
        local uuid=$(nvidia-smi -i "$idx" --query-gpu=gpu_uuid --format=csv,noheader 2>/dev/null)
        local pids
        pids=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null \
               | awk -F', *' -v U="$uuid" '$1==U { print $2 }')
        if [ -z "$pids" ]; then
            echo "$idx"
        else
            local foreign=0
            for pid in $pids; do
                local owner
                owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ' || true)
                if [ -n "$owner" ] && [ "$owner" != "$ME" ]; then
                    foreign=1; break
                fi
            done
            [ "$foreign" -eq 0 ] && echo "$idx"
        fi
    done
}

case "${1:-list}" in
    list)
        scan_once
        ;;
    count)
        scan_once | wc -l
        ;;
    wait)
        interval="${2:-60}"
        max_polls="${3:-30}"
        need="${4:-4}"
        for ((i=1; i<=max_polls; i++)); do
            free=$(scan_once)
            n=$(echo "$free" | grep -c . || true)
            echo "[$(date +%H:%M:%S)] poll $i/$max_polls: $n free ($free)" >&2
            if [ "$n" -ge "$need" ]; then
                echo "$free" | head -n "$need" | paste -sd,
                exit 0
            fi
            sleep "$interval"
        done
        echo "FAILED: no $need-GPU window in $((interval*max_polls))s" >&2
        exit 1
        ;;
    *)
        echo "Usage: $0 [list|count|wait <interval> <max_polls> <need>]" >&2
        exit 2
        ;;
esac
