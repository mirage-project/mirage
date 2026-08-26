# DeepSeek V3 Demo — Mirage Persistent Kernel

## System Requirements

- **GPU**: NVIDIA B200 (SM100a) with ≥80 GB HBM per device
- **CUDA**: 12.8+
- **Python**: 3.10+ with PyTorch 2.6+ (BF16/FP8 support)
- **MPI**: OpenMPI 4.1+ (for TP > 1)
- **NVSHMEM**: 3.6.5 (for TP > 1, AllReduce via NVLS)
- **Model**: DeepSeek V3 weights (safetensors format, FP8 checkpoint)

### GPU Exclusivity

The persistent kernel requires **exclusive GPU access**. Other processes on the
same GPU cause worker/scheduler co-scheduling failure and deadlock.

Before running, verify GPUs are idle:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# All target GPUs must show memory ≈ 0 MiB and utilization = 0%
```

## Environment Setup

Set the following paths to point at your local install locations (shown as
placeholders; adjust to match your environment).

```bash
# Your Python venv with PyTorch 2.6+ and the mirage package installed editable.
source "${MIRAGE_VENV:-/path/to/mirage-venv}/bin/activate"

# OpenMPI 4.1+ (for TP > 1)
export MPI_HOME=${MPI_HOME:-/usr/mpi/gcc/openmpi-4.1.9a1}
export PATH=$MPI_HOME/bin:$PATH
export MPI_INC_PATH=$MPI_HOME/include
export MPI_LIB_PATH=$MPI_HOME/lib

# NVSHMEM 3.6.5 (for TP > 1, AllReduce via NVLS).
# Point NVSHMEM_HOME at the install prefix for your machine.
export NVSHMEM_HOME=${NVSHMEM_HOME:-/path/to/nvshmem-3.6.5}
export NVSHMEM_INC_PATH=$NVSHMEM_HOME/include/nvshmem_13
export NVSHMEM_LIB_PATH=$NVSHMEM_HOME/lib/x86_64-linux-gnu/nvshmem/13
export LD_LIBRARY_PATH=$NVSHMEM_LIB_PATH:$MPI_HOME/lib:$LD_LIBRARY_PATH
# libnvshmem_host is needed at load time — adjust if your package puts it
# under a different prefix (some distros split headers and host lib).
export LD_PRELOAD=$NVSHMEM_HOME/lib/x86_64-linux-gnu/nvshmem/13/libnvshmem_host.so.3.6.5

# Bump NVSHMEM team budget. The default (32) is exceeded once we register the
# MLA-TP, AllReduce-residual, and MoE teams together for the full 61-layer
# model at TP8 EP2. Without this the megakernel launch fails with "NCCL error
# … team_internal.cpp:690 'unhandled cuda error'" partway through
# `MPK: Creating nvshmem team N/M`. 128 is comfortable for all current configs.
export NVSHMEM_MAX_TEAMS=128
```

## Quickstart

### Full Model (TP=8, EP=2, all 61 layers)

DeepSeek V3 MPK ships **TP=8, EP=2 on B200 only** — the complete 61-layer model
does not fit below TP=8, and smaller TP is out of scope. All 8 GPUs must be idle
beforehand.

```bash
# Verify all 8 GPUs are idle first.
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# Decode (single-token): --max-num-batched-tokens 1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun --allow-run-as-root -np 8 \
    -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH \
    -x MPI_INC_PATH -x MPI_LIB_PATH -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH \
    python demo/deepseek_v3/demo.py \
    --model-path /path/to/DeepSeek-V3 \
    --use-mirage --ep-size 2 \
    --max-num-batched-tokens 1 --max-seq-length 4096
```

Chunked-prefill variant — set `--max-num-batched-tokens 128` (the prefill chunk
size; values `>=32` select the chunked-prefill MLA kernel):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun --allow-run-as-root -np 8 \
    -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH \
    -x MPI_INC_PATH -x MPI_LIB_PATH -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH \
    python demo/deepseek_v3/demo.py \
    --model-path /path/to/DeepSeek-V3 \
    --use-mirage --ep-size 2 \
    --max-num-batched-tokens 128 --max-seq-length 4096
```

Notes:
- `--ep-size 2` shards the routed MoE experts across two expert-parallel groups
  (routed experts run at TP = world_size / ep_size = 4); non-MoE layers and
  shared experts stay at TP=8.
- `--max-num-batched-tokens` is the model token budget for one MPK scheduling
  step. `1` selects the pure single-token decode path; values `>=32` select the
  chunked-prefill MLA kernel (in prefill this is the chunk size). In decode, the
  number of decoded tokens per step is controlled by the active request count,
  not by this budget.

## Weight cache

Pass `--weight-cache-dir /path` (or set `MPK_DEEPSEEK_WEIGHT_CACHE_DIR`) to cache
the converted / fused / sharded per-rank tensors after the first run. Later runs
with the same model / TP / EP / vocab-parallel settings load them directly and
skip the (slow) cold weight conversion.

On a cold TP8 start all 8 ranks convert weights at once, which can exhaust host
RAM. Set `MPK_CONVERT_SEMAPHORE=K` to cap concurrent per-rank conversion to `K`
ranks at a time and bound peak host memory.

## Profiling

Use `--profiling` and `--trace-name` to emit one Perfetto trace per rank. Keep
the trace name outside the repo if you do not want profiling artifacts in the
working tree.

Example: full model at TP=8 EP=2, batch size 1, a 1024-token synthetic context
plus 128 decode tokens:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun --allow-run-as-root -np 8 \
    -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH \
    -x MPI_INC_PATH -x MPI_LIB_PATH -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH \
    python demo/deepseek_v3/demo.py \
    --model-path /path/to/DeepSeek-V3 \
    --use-mirage --ep-size 2 --profiling \
    --trace-name /tmp/deepseek_v3_tp8_ep2_ctx1024_decode128 \
    --prompt-length 1024 --ignore-eos \
    --max-num-batched-tokens 1 \
    --max-num-batched-requests 1 \
    --max-seq-length 1152 \
    --max-new-tokens 128 \
    --save-tokens /tmp/deepseek_v3_tp8_ep2_ctx1024_decode128_tokens.json
```

This writes files such as
`/tmp/deepseek_v3_tp8_ep2_ctx1024_decode128_rank0.perfetto-trace`.

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | (required) | Path to converted DeepSeek V3 weights |
| `--use-mirage` | off | Use the Mirage persistent kernel (vs native PyTorch) |
| `--profiling` | off | Enable profiling to emit a Perfetto trace |
| `--max-num-batched-tokens` | 8 | Model token budget for one scheduling step. In prefill this is the chunk size; `>=32` selects the chunked-prefill MLA kernel; `1` = pure single-token decode. Decode width is set by active requests, not this budget. |
| `--max-num-batched-requests` | 1 | Max concurrent requests (the request batch-size limit) |
| `--page-size` | 128 | Tokens per KV cache page |
| `--max-num-pages` | 64 | Max KV cache pages |
| `--max-seq-length` | 4096 | Max sequence length (affects KV cache allocation). **Use `max_seq_length ≈ prompt_len + max_new_tokens`** — the offline driver generates to `max_seq_length`, so over-allocating costs O(max_seq²) decode time. |
| `--prompt` | (built-in text) | Input prompt text |
| `--prompts-json` | None | JSON array of prompts for batched-request testing; its length must equal `--max-num-batched-requests` |
| `--prompt-length` | 0 | If >0, replace `--prompt` with a synthetic prompt of exactly N tokens (prefill stress test) |
| `--ep-size` | 1 | Expert-parallel group count for routed MoE experts. Non-MoE layers and shared experts keep TP=world_size; routed experts use TP=world_size/ep_size. |
| `--disable-vocab-parallel-lm-head` | off | Disable the TP vocab-parallel LM-head fast path (enabled by default for TP>1) |
| `--output-dir` | None | Output files directory (compiled kernel artifacts) |
| `--trace-name` | "" | Output name for the Perfetto trace file |
| `--ignore-eos` | off | Do not stop at the EOS token |
| `--max-new-tokens` | None | Decode cap (e.g. for CI determinism) |
| `--temperature` | 0.0 | Sampling temperature (0 = greedy) |
| `--top_p` | 1.0 | Top-p sampling |
| `--do-sample` | off | Enable sampling |
| `--save-tokens` | None | Dump the first N generated token_ids, text, and latency to JSON (path optional; defaults under `outputs/deepseek_v3/`) |
| `--weight-cache-dir` | `$MPK_DEEPSEEK_WEIGHT_CACHE_DIR` | Directory for the MPK-ready per-rank weight cache; converted/fused/sharded tensors are saved after the first run and reused by later runs with matching model/TP/EP/vocab-parallel settings |
