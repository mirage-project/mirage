# Segmented μGraph compilation — prototype

Compiles selected model regions with Mirage's ordinary μGraph compiler
(`KNGraph.superoptimize()` / `KNGraph.compile()`) instead of lowering them to
the MPK task/event graph. The MPK task-graph implementation is unchanged and
used only as the performance baseline.

See [`DESIGN.md`](DESIGN.md) for the design, the region boundaries, and the
fair-comparison caveats.

## Layout

| File               | Purpose                                                             |
| ------------------ | ------------------------------------------------------------------- |
| `runner.py`        | `SegmentedMuGraphRunner`, region graphs, cache keys, task-graph guard |
| `common.py`        | Deterministic inputs, FP32 oracle, metrics, CUDA-event timing        |
| `bench_mlp.py`     | Stage 1 — three-way Qwen3 MLP microbenchmark                          |
| `hybrid_mlp.py`    | `HybridQwen3MLP` + model patching                                    |
| `bench_qwen3.py`   | Stage 2 — three-mode Qwen3-0.6B experiment                            |

Tests live in `tests/experiments/test_segmented_mugraph.py`. Reports are
written to `experiments/outputs/` (git-ignored).

## Running

All commands assume the repo root and `PYTHONPATH=.`.

```bash
# Stage 1 — three-way MLP microbenchmark (8 tokens, 4096 hidden, 2048 inter, bf16)
PYTHONPATH=. python -m experiments.segmented_mugraph.bench_mlp \
    --warmups 20 --iters 100 \
    --out experiments/outputs/stage1_mlp.json

# Stage 2 — three-mode Qwen3-0.6B experiment
PYTHONPATH=. python -m experiments.segmented_mugraph.bench_qwen3 \
    --model Qwen/Qwen3-0.6B --gen-tokens 32 \
    --out experiments/outputs/stage2_qwen3.json

# Tests (static ones need no GPU)
PYTHONPATH=. pytest tests/experiments/test_segmented_mugraph.py -v
```

Useful flags: `--no-superoptimize` (skip the search, compile the high-level
graph directly), `--scopes region_a,region_b,full`, `--impls torch,mpk,mugraph`,
`--extra-buckets 2,4` (compile extra fixed token buckets for the hybrid model).

## Measured results

NVIDIA B200 (sm_100), CUDA 13.2, torch 2.11.0+cu130, bf16, single GPU.

### Stage 1 — Qwen3 dense MLP, 8×4096→2048

Each implementation runs in its own process; Region A and Region B are not
synchronized between.

| impl    | scope    | compiler              | mean ms | p5 ms  | p95 ms | tok/s  | max\|err\| | cold s |
| ------- | -------- | --------------------- | ------- | ------ | ------ | ------ | ---------- | ------ |
| torch   | region_a | eager-pytorch-bf16    | 0.0414  | 0.0391 | 0.0461 | 193292 | 0.03125    | 0.0    |
| mpk     | region_a | mpk-task-graph        | 0.1271  | 0.1245 | 0.1286 | 62921  | 0.03125    | 30.2   |
| mugraph | region_a | direct                | 0.0517  | 0.0431 | 0.0772 | 154732 | 0.03125    | 25.1   |
| torch   | region_b | eager-pytorch-bf16    | 0.0220  | 0.0203 | 0.0244 | 364400 | 0.01562    | 0.0    |
| mpk     | region_b | mpk-task-graph        | 0.1320  | 0.1305 | 0.1335 | 60619  | 0.00781    | 29.5   |
| mugraph | region_b | **superoptimized**    | 0.0306  | 0.0286 | 0.0334 | 261397 | 0.01562    | 33.0   |
| torch   | full     | eager-pytorch-bf16    | 0.0565  | 0.0535 | 0.0657 | 141650 | 0.01562    | 0.0    |
| mpk     | full     | mpk-task-graph        | 0.1524  | 0.1509 | 0.1539 | 52500  | 0.01562    | 29.3   |
| mugraph | full     | direct+superoptimized | 0.0957  | 0.0853 | 0.1272 | 83603  | 0.01562    | 76.5   |

All implementations pass the MPK test's bf16 bound (max abs error < 1.0 for the
complete MLP; the worst observed is 0.031). For the full MLP the μGraph path
reaches cosine 0.9999991, relative L2 1.3e-3.

Region A falls back to direct compilation because the Hopper/Blackwell
threadblock transpiler backends were broken. Six defects have since been fixed
for Blackwell, and a **single-tile custom threadblock op now compiles and is
numerically correct on sm_100** (see `tests/experiments/test_blackwell_codegen.py`).
Matmul and pipelined/TMA graphs still fail, so Region A's superoptimizer
candidates — all matmul — continue to fall back. See DESIGN.md §6.

### Stage 2 — Qwen3-0.6B, greedy, batch 1, 32 tokens, 1 warmup + 3 reps

| mode           | cold s | prefill ms | TTFT ms | mean ITL ms | p50 ITL | p95 ITL | decode tok/s |
| -------------- | ------ | ---------- | ------- | ----------- | ------- | ------- | ------------ |
| torch          | 5.7    | 14.394     | 27.321  | 12.699      | 12.592  | 13.581  | 78.7         |
| hybrid-mugraph | 63.6   | 14.319     | 31.567  | 17.179      | 17.080  | 17.690  | 58.2         |
| mpk            | 42.6   | —          | —       | 1.192       | —       | —       | 839.2        |

* μGraph region variants compiled: **2** (reused across all 28 layers)
* μGraph MLP calls: **3472**, PyTorch fallback calls: **112** (prefill)
* Cache hits / misses: **6944 / 2**
* First-decode logits, torch vs hybrid: **cos = 0.999991**, top-1 match — PASS
* Token agreement torch vs hybrid: **32/32 (100%)**
* Token agreement torch vs mpk: 29/32 (90.6%), first divergence at index 29

MPK does not expose prefill/TTFT/per-step percentiles: it runs the whole decode
inside one persistent megakernel launch and reports a single amortized
per-token latency.

**Read the Stage-2 table as a runtime comparison, not a kernel comparison.**
`torch` and `hybrid-mugraph` run under PyTorch/HF Python orchestration; `mpk`
runs the entire decode in one persistent megakernel with its own scheduler,
attention and sampling kernels. Stage 1 shows the μGraph MLP is *faster* than
MPK's MLP tasks at this size, yet MPK wins end-to-end by ~13× — that gap is the
runtime, not the MLP.
