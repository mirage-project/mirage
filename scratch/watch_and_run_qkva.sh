#!/usr/bin/env bash
# Watchdog: continuously scan for 4 free GPUs (per scan_free_gpus.sh criteria),
# then auto-launch the QKV-a TP=4 EP=2 fused smoke with the pre-RMSnorm
# diagnostic dump position.
#
# Writes progress to /tmp/qkva_watchdog.log. Once 4 free GPUs are found:
#   1. Atomically tries to claim them by exporting CUDA_VISIBLE_DEVICES.
#   2. Runs the smoke (mpirun -np 4).
#   3. Writes results summary to /tmp/qkva_watchdog.result.
#
# Usage:
#   bash scratch/watch_and_run_qkva.sh &     # run in background
#
# Stop by writing "STOP" to /tmp/qkva_watchdog.stop:
#   touch /tmp/qkva_watchdog.stop

set -uo pipefail

LOG=/tmp/qkva_watchdog.log
RESULT=/tmp/qkva_watchdog.result
STOP=/tmp/qkva_watchdog.stop
INTERVAL=30           # seconds between scans
NEED=4
TAG="${WATCHDOG_TAG:-membar_gl_fix}"

mkdir -p $(dirname "$LOG")
rm -f "$RESULT" "$STOP"
echo "[$(date '+%H:%M:%S')] watchdog started; need=$NEED; interval=${INTERVAL}s" > "$LOG"

poll=0
while true; do
  poll=$((poll + 1))
  if [ -f "$STOP" ]; then
    echo "[$(date '+%H:%M:%S')] stop file detected; exiting" >> "$LOG"
    exit 0
  fi
  free_list=$(bash /home/muhengl/mirage/scratch/scan_free_gpus.sh list 2>/dev/null)
  count=$(echo "$free_list" | grep -c '.' || echo 0)
  echo "[$(date '+%H:%M:%S')] poll $poll: $count free ($(echo $free_list | tr '\n' ','))" >> "$LOG"
  if [ "$count" -ge "$NEED" ]; then
    gpus=$(echo "$free_list" | head -n "$NEED" | paste -sd,)
    echo "[$(date '+%H:%M:%S')] FOUND $NEED free GPUs: $gpus — launching smoke" >> "$LOG"

    # Run the smoke with the claimed GPUs. We can't edit the in-tree script
    # without polluting; instead inline the mpirun with explicit env.
    OUT=/home/muhengl/mirage/outputs/dpskv3_qkva_fused_${TAG}
    mkdir -p "${OUT}/build" "${OUT}/dump"

    export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1
    export PATH=$MPI_HOME/bin:$PATH
    export MPI_INC_PATH=$MPI_HOME/include
    export MPI_LIB_PATH=$MPI_HOME/lib
    export NVSHMEM_HOME=/home/muhengl/local/nvshmem-3.6.5-dev/usr
    export NVSHMEM_INC_PATH=$NVSHMEM_HOME/include/nvshmem_13
    export NVSHMEM_LIB_PATH=$NVSHMEM_HOME/lib/x86_64-linux-gnu/nvshmem/13
    export LD_LIBRARY_PATH=$NVSHMEM_LIB_PATH:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}
    export LD_PRELOAD=$NVSHMEM_LIB_PATH/libnvshmem_host.so.3.6.5
    export NVSHMEM_MAX_TEAMS=128

    export MPK_DSV3_QKV_A_FUSED=1
    export MPK_DSV3_QKV_A_FUSED_N=2176
    export MPK_DSV3_FP8_BUF_ATTACH="${MPK_DSV3_FP8_BUF_ATTACH:-0}"
    export MPK_DSV3_QKV_A_OUT_ATTACH="${MPK_DSV3_QKV_A_OUT_ATTACH:-0}"
    export MPK_DEEPSEEK_WEIGHT_CACHE_DIR=/tmp/dpskv3_v8_weight_cache_qkva_fused_2176

    cd /home/muhengl/mirage

    set +e
    CUDA_VISIBLE_DEVICES="$gpus" mpirun --allow-run-as-root -np 4 \
        -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH \
        -x MPI_INC_PATH -x MPI_LIB_PATH -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH \
        -x NVSHMEM_MAX_TEAMS \
        -x MPK_DSV3_QKV_A_FUSED -x MPK_DSV3_QKV_A_FUSED_N \
        -x MPK_DSV3_FP8_BUF_ATTACH -x MPK_DSV3_QKV_A_OUT_ATTACH \
        -x MPK_DEEPSEEK_WEIGHT_CACHE_DIR \
        /home/muhengl/mirage/.venv/bin/python demo/deepseek_v3/demo.py \
        --model-path /raid/catalyst/models/DeepSeek-V3 \
        --use-mirage --max-num-batched-tokens 128 --max-num-batched-requests 1 \
        --page-size 128 --max-num-pages 2 --max-seq-length 256 \
        --prompt-length 128 --ignore-eos --max-new-tokens 1 \
        --layers 0-3 --mtp 0 --ep-size 2 \
        --output-dir "${OUT}/build" --dump-hidden-dir "${OUT}/dump" \
        > "${OUT}/run.log" 2>&1
    rc=$?
    set -e

    echo "[$(date '+%H:%M:%S')] smoke rc=$rc; OUT=$OUT" >> "$LOG"

    # Analyze and write a short result
    /home/muhengl/mirage/.venv/bin/python <<PY >> "$RESULT" 2>&1
import torch, os
DIR = '$OUT/dump'
if not os.path.exists(f'{DIR}/layer0_q_a_out.pt'):
    print(f'FAIL: no dump. rc=$rc')
    raise SystemExit(1)
f = torch.load(f'{DIR}/layer0_q_a_out.pt', weights_only=True).float()
print(f'rc=$rc')
print(f'gpus={"$gpus"}')
print(f'shape: {tuple(f.shape)}')
for name, s in [('q_a',f[:,:1536]),('c_lat',f[:,1536:2048]),('k_pe',f[:,2048:2112])]:
    zr = (s.abs().sum(dim=1)==0).nonzero(as_tuple=True)[0]
    rng = ''
    if zr.numel() > 0:
        zl = zr.tolist(); ranges = []; i = 0
        while i < len(zl):
            j = i
            while j+1 < len(zl) and zl[j+1] == zl[j]+1: j += 1
            ranges.append((zl[i], zl[j])); i = j+1
        rng = ', '.join(f'{a}' if a==b else f'{a}..{b}' for a,b in ranges)
    print(f'{name:>6s}: zero={zr.numel():>3d}  rng={rng}')
print(f'row0_norm={f[0].norm():.3f}  row1_norm={f[1].norm():.3f}  row71_norm={f[71].norm():.3f}  row72_norm={f[72].norm():.3f}')

# Compare residual cos to baseline
base = '/home/muhengl/mirage/outputs/dpskv3_qkva_baseline_sanitybase/dump'
for L in range(4):
    a = torch.load(f'{base}/layer_{L:02d}_residual.pt', weights_only=True).float()
    b = torch.load(f'{DIR}/layer_{L:02d}_residual.pt', weights_only=True).float()
    cos = torch.dot(a.flatten(), b.flatten()) / (a.norm() * b.norm() + 1e-12)
    print(f'layer_{L:02d}_residual cos: {cos:.6f}')
PY

    echo "[$(date '+%H:%M:%S')] watchdog done; result in $RESULT" >> "$LOG"
    exit 0
  fi
  sleep "$INTERVAL"
done
