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
```

## Quickstart

### Single GPU (TP=1)

```bash
# Full model, 40 layers + MTP
python demo/deepseek_v3/demo.py \
    --model-path /raid/catalyst/models/DeepSeek-V3 \
    --use-mirage --layers 0-39 --mtp --num-speculative-tokens 3 \
    --max-num-batched-tokens 1 --max-seq-length 4096

# Single MoE layer, correctness test
python demo/deepseek_v3/demo.py \
    --model-path /raid/catalyst/models/DeepSeek-V3 \
    --use-mirage --correctness --layers 3 \
    --max-num-batched-tokens 1 --max-seq-length 512
```

### Multi-GPU (TP=4)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun --allow-run-as-root -np 4 \
    -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH \
    -x MPI_INC_PATH -x MPI_LIB_PATH -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH \
    python demo/deepseek_v3/demo.py \
    --model-path /raid/catalyst/models/DeepSeek-V3 \
    --use-mirage --layers 0-39 \
    --max-num-batched-tokens 1 --max-seq-length 4096
```

### Full Model on 8 GPUs (TP=8, all 61 layers + MTP)

Use TP=8 when you want to hold the **complete** DeepSeek V3 weights (61
decoder layers + MTP) on a single node. All 8 GPUs must be idle beforehand.

```bash
# Verify all 8 GPUs are idle first.
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun --allow-run-as-root -np 8 \
    -x CUDA_VISIBLE_DEVICES -x LD_LIBRARY_PATH -x LD_PRELOAD -x PATH \
    -x MPI_INC_PATH -x MPI_LIB_PATH -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH \
    python demo/deepseek_v3/demo.py \
    --model-path /path/to/DeepSeek-V3 \
    --use-mirage --layers 0-60 --mtp --num-speculative-tokens 3 \
    --max-num-batched-tokens 128 --max-seq-length 16384 \
    --max-num-pages 160
```

Notes:
- `--max-num-batched-tokens 128` enables chunked prefill (see
  `MPK_USE_PREFILL` below). Drop to `1` if you want the pure decode path.
- `--max-num-pages 160` ≈ `(16384 / 128) + 32` page headroom at `--page-size 128`.
- MTP with `--num-speculative-tokens 3` keeps spec_length ≤ 7 and stays on the
  MLA TP decode kernel; larger spec lengths currently aren't validated.

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | (required) | Path to DeepSeek V3 safetensors weights |
| `--use-mirage` | off | Use Mirage persistent kernel (vs native PyTorch) |
| `--layers` | all 61 | Comma-separated layer indices (e.g. `0,3,60`) or range `0-39` |
| `--correctness` | off | Run PyTorch reference comparison (single token) |
| `--mtp` | off | Enable MTP speculative decoding |
| `--num-speculative-tokens` | 1 | Draft tokens per MTP step (1-7) |
| `--max-num-batched-tokens` | 8 | Batch size (tokens per decode step) |
| `--max-num-batched-requests` | 1 | Max concurrent requests |
| `--max-seq-length` | 4096 | Max sequence length (affects KV cache allocation) |
| `--max-num-pages` | 64 | Max KV cache pages |
| `--page-size` | 128 | Tokens per KV cache page |
| `--profiling` | off | Attach profiler tensor for Perfetto trace |
| `--trace-name` | "" | Output name for Perfetto trace file |
| `--prompt` | (default) | Input prompt text |
| `--temperature` | 0.0 | Sampling temperature (0 = greedy) |
| `--do-sample` | off | Enable sampling |
| `--ignore-eos` | off | Don't stop at EOS token |
| `--max-new-tokens` | None | Cap on generated tokens |
| `--save-tokens` | off | Dump generated tokens to JSON |
| `--output-dir` | None | Output directory for compiled kernel |

## Debug Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MPK_SKIP_ATTN` | 0 | Skip MLA attention (test MLP/MoE path only) |
| `MPK_FUSE_RESIDUAL` | 1 | Use fused residual kernel (0 = separate elementwise_add) |
| `MPK_SKIP_MLA_ONLY` | 0 | Skip post-attention MLP/MoE (test attention path only) |
| `MPK_SKIP_SHARED_EXPERT` | 0 | Skip MoE shared expert |
| `MPK_SKIP_ROUTED_EXPERTS` | 0 | Skip MoE routed experts |
| `MPK_SKIP_LM_HEAD` | 0 | Skip lm_head linear |
| `MPK_DUMP_LOGITS` | 0 | Expose lm_head output buffer for inspection |
| `MPK_DUMP_MOE_OUTPUT` | 0 | Expose MoE output buffer |
| `MPK_MLA_CHECKPOINT` | 0 | Compare per-step MLA intermediates vs reference |
| `MPK_REF_NO_QUANT` | 0 | PyTorch reference: skip FP8 requant (use raw dequant) |
| `MPK_REF_TRUE_FP8` | 0 | PyTorch reference: use per-block FP8 matmul |
| `MPK_AR_LOCAL_COPY` | 0 | Replace AllReduce with local copy (no-op comms) |
| `MPK_SKIP_ALLREDUCE` | 0 | Remove AllReduce tasks from builder entirely |
| `MPK_NO_NVSHMEM` | 0 | Compile without NVSHMEM (TP=1 only) |
| `MPK_DRY_RUN` | 0 | Stop after task graph generation (inspect offsets) |
| `MPK_SO_PATH` | auto | Override compiled .so path for cumodule init |
| `MPK_USE_PREFILL` | 1 | Dispatch `mla_prefill_sm100` when `max_num_batched_tokens >= 32`. Set to `0` to force the decode kernel for all Q_LEN (debug/bisect flag). |

## Known Limitations

- **MLA TP efficiency**: The generic MLA kernel (M=128 MMA tile) works correctly
  for all TP sizes by masking unused rows via `effective_len=0`. Dedicated TP
  kernels (tp2/tp4/tp8, from PR #663) use smaller MMA tiles for better efficiency
  but are not yet wired into the builder. This is a performance optimization, not
  a correctness issue.
- **FP8 precision drift**: Single-layer MLA cosine ~0.93 vs BF16 reference
  (inherent FP8 quantization). Compounds over layers.
