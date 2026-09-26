# Sparse MLA (SM100)

This is a BF16 selected-token MLA task for GLM-5.3-style latent layouts, not a
complete model implementation. Both decode and chunked prefill use per-query
token indices. The indexer, IndexPool expansion/deduplication, Q projection,
RoPE, cache writes and final output projection remain caller responsibilities.

## Contract

All tensors are contiguous. `q` is `[T, H, 512 + R]`, cache is
`[num_pages, page_size, 512 + R]`, and output is `[T, H, 512]`. `R` is 0 or 64;
`H` is 8, 16, 32 or 64. The first 512 cache columns are also the latent values.
Q must already contain the absorbed key projection and any positional part.
The original model's attention scale must be supplied explicitly.

`token_indices` is INT32 `[T, K_capacity]`; `index_counts` is INT32 `[T]`.
Each count selects a prefix of the corresponding index row. Within it, valid
indices must be unique. Negative, out-of-range and future indices are masked.
Indices are logical token positions within the request, not physical cache
addresses or pool IDs. A count must lie between zero and capacity. CPU reference
validation rejects duplicates and invalid counts; the kernel does not deduplicate
and defensively clamps counts. Empty selections produce exact zero.

The caller supplies valid, monotonically increasing query/page indptrs and page
tables. The cache must already include all new tokens in the query chunk. Queries
are packed by request; unused trailing query slots are zeroed. The runtime derives
absolute query positions from sequence and query lengths, then enforces causality.

```python
pk.sparse_mla_layer(
    q=pk.attach_input(q, name="q"),
    kv_cache=pk.attach_input(cache, name="cache"),
    token_indices=pk.attach_input(indices, name="indices"),
    index_counts=pk.attach_input(counts, name="counts"),
    output=pk.attach_input(output, name="output"),
    softmax_scale=0.0625,  # Example only: use the target model's scale.
    num_splits=4,
)
```

Splits 1/2/4/8 are supported; 1 writes output directly. Larger values allocate
FP32 partial output and LSE and register an additional MPK reduce task. The
kernel stages selected KV tiles in shared memory without a global gather buffer.
The implementation uses standard BF16 WMMA Tensor Core operations, as distinct
from the existing SM100 TMA/tcgen05 MLA pipeline. It is a correctness-first
baseline; performance on either query regime remains unmeasured.

## Validation commands

Run these commands from this directory. CPU tests require NumPy only:

```bash
python -m unittest test_reference -v
python -m unittest test_layer_contract -v
```

On a Blackwell SM100 host, use CUDA 12.8+ and CUDA-enabled PyTorch. The standalone
launcher supports page sizes 64/128. The MPK wrapper specializes its configured
page size. GPU tests deliberately fail if CUDA is available but the extension
is missing; a CPU-only host reports explicit skips.

```bash
python setup.py build_ext --inplace
python -m unittest test_sparse_mla -v
# Also requires this modified Mirage checkout to be built and installed:
python -m unittest test_sparse_mla_mpk -v
compute-sanitizer --tool memcheck --error-exitcode 1 python -m unittest test_sparse_mla
compute-sanitizer --tool racecheck --error-exitcode 1 python -m unittest test_sparse_mla
compute-sanitizer --tool synccheck --error-exitcode 1 python -m unittest test_sparse_mla
python benchmark_sparse_mla.py --queries 1 --kv-len 16384 --topk 2048
python benchmark_sparse_mla.py --queries 32 --kv-len 16384 --topk 2048
```

Standalone tests cover mixed requests/history, shuffled pages, all head/RoPE
specializations, split reduction, invalid selections, long lists, inactive query
slots and reused workspaces. MPK tests cover actual graph registration, reduction
dependencies and a downstream consumer, using offline pure-prefill metadata.
They do not claim end-to-end GLM compatibility or historical decode integration.

GPU acceptance: finite outputs, `atol=0.02`, `rtol=0.02`, relative RMS <= 2%, and
exact zeros for empty selections. Latency includes the standalone compute/reduce
launch chain, not index generation, cache writes or MPK scheduling. No performance
result should be inferred from the CPU tests.

## Current validation status

Developed on macOS ARM64 without NVCC, CUDA PyTorch or an NVIDIA GPU. GPU build,
correctness, sanitizers and performance are pending. See the planning document
at `docs/sparse_mla_plan.zh-CN.md` for scope and design decisions.
