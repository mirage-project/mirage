#!/usr/bin/env bash
# Two single-layer (--layers 0-0) smokes to localise the QKV-a fused rows-1..71-zero bug:
#   (1) layer0_fused          : MPK_DSV3_QKV_A_FUSED=1, layers 0-0 (no cascade)
#   (2) layer0_unfused        : MPK_DSV3_QKV_A_FUSED=0, layers 0-0 (reference)
#
# If layer0_fused still shows rows 1..71 zero → bug is per-layer (not cascade)
# If layer0_fused now shows non-zero rows 1..71 → bug needs >1 layer to manifest
#
# Writes /tmp/qkva_layer0_only.log and /tmp/qkva_layer0_only.result.

set -uo pipefail

LOG=/tmp/qkva_layer0_only.log
RESULT=/tmp/qkva_layer0_only.result
STOP=/tmp/qkva_layer0_only.stop
INTERVAL=30
NEED=4

rm -f "$RESULT" "$STOP"
echo "[$(date '+%H:%M:%S')] watchdog (layer0_only) started" > "$LOG"

run_one() {
    local tag="$1"
    local extra_env="$2"
    local gpus="$3"

    local OUT=/home/muhengl/mirage/outputs/dpskv3_qkva_${tag}
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

    eval "$extra_env"
    export MPK_DEEPSEEK_WEIGHT_CACHE_DIR=/tmp/dpskv3_v8_weight_cache_qkva_fused_2176

    cd /home/muhengl/mirage

    echo "[$(date '+%H:%M:%S')] launching tag=$tag gpus=$gpus" >> "$LOG"

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
        --layers 0-0 --mtp 0 --ep-size 2 \
        --output-dir "${OUT}/build" --dump-hidden-dir "${OUT}/dump" \
        > "${OUT}/run.log" 2>&1
    local rc=$?
    set -e
    echo "[$(date '+%H:%M:%S')] tag=$tag rc=$rc" >> "$LOG"

    /home/muhengl/mirage/.venv/bin/python <<PY >> "$RESULT" 2>&1
import torch, os
DIR = '$OUT/dump'
print(f'\n=== tag=$tag rc=$rc ===')
qkv_path = f'{DIR}/layer0_q_a_out.pt'
if not os.path.exists(qkv_path):
    print(f'FAIL: no qkv dump.')
else:
    f = torch.load(qkv_path, weights_only=True).float()
    print(f'layer0_q_a_out shape: {tuple(f.shape)}')
    if f.shape[1] >= 2112:
        slices = [('q_a',f[:,:1536]),('c_lat',f[:,1536:2048]),('k_pe',f[:,2048:2112])]
    else:
        slices = [('q_a_full', f)]
    for name, sl in slices:
        zr = (sl.abs().sum(dim=1)==0).nonzero(as_tuple=True)[0]
        rng = ''
        if zr.numel() > 0:
            zl = zr.tolist(); ranges = []; i = 0
            while i < len(zl):
                j = i
                while j+1 < len(zl) and zl[j+1] == zl[j]+1: j += 1
                ranges.append((zl[i], zl[j])); i = j+1
            rng = ', '.join(f'{a}' if a==b else f'{a}..{b}' for a,b in ranges)
        print(f'  {name:>8s}: zero={zr.numel():>3d}  rng={rng}')
    print(f'  row 0  norm={f[0].norm():.4f}')
    print(f'  row 1  norm={f[1].norm():.4f}')
    print(f'  row 35 norm={f[35].norm():.4f}')
    print(f'  row 71 norm={f[71].norm():.4f}')
    print(f'  row 72 norm={f[72].norm():.4f}')
    print(f'  row 127 norm={f[127].norm():.4f}')
PY
}

poll=0
while true; do
  poll=$((poll + 1))
  if [ -f "$STOP" ]; then echo "stop" >> "$LOG"; exit 0; fi
  free_list=$(bash /home/muhengl/mirage/scratch/scan_free_gpus.sh list 2>/dev/null)
  count=$(echo "$free_list" | grep -c '.' || echo 0)
  echo "[$(date '+%H:%M:%S')] poll $poll: $count free ($(echo $free_list | tr '\n' ','))" >> "$LOG"
  if [ "$count" -ge "$NEED" ]; then
    gpus=$(echo "$free_list" | head -n "$NEED" | paste -sd,)

    # PRIMARY: single-layer fused (isolates from cross-layer cascade)
    run_one layer0_fused \
        'export MPK_DSV3_QKV_A_FUSED=1 MPK_DSV3_QKV_A_FUSED_N=2176 MPK_DSV3_FP8_BUF_ATTACH=0 MPK_DSV3_QKV_A_OUT_ATTACH=0' \
        "$gpus"

    # CONTROL: single-layer unfused (sanity check + reference)
    run_one layer0_unfused \
        'export MPK_DSV3_QKV_A_FUSED=0 MPK_DSV3_QKV_A_FUSED_N=2176 MPK_DSV3_FP8_BUF_ATTACH=0 MPK_DSV3_QKV_A_OUT_ATTACH=0' \
        "$gpus"

    echo "[$(date '+%H:%M:%S')] all smokes done; result in $RESULT" >> "$LOG"
    exit 0
  fi
  sleep "$INTERVAL"
done
