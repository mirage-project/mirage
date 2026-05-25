# NVFP4 and MXFP4 Linear Layer
## 1. Overview
The branch implements NVFP4 and MXFP4 Linear Layer kernel for MPK for NVIDIA B200 (SM100a). It adds the following:
- NVFP4 Dense GEMM Kernel
- NVFP4 Dense GEMM Kernel (swapAB)
- NVFP4 Quantization Kernel
- MXFP4 Dense GEMM Kernel
- MXFP4 Dense GEMM Kernel (swapAB)
- MXFP4 Quantization Kernel
- Test scripts for each kernel
- Profile scripts for each kernel

**Related Issues:**
- #601
- #506
<!--
Issues closed by this PR:
- Closes #
-->

## 2. Kernel Support
The kernel supports the following:

### Entry Points

#### NVFP4
- **`linear_nvfp4_sm100(x, w, w_sf, residual, output)`** — auto-quantizes float32 activations then runs GEMM
- **`linear_nvfp4_sm100_no_quantization(x_q, x_sf, w, w_sf, residual, output)`** — takes pre-quantized inputs
- **`quantize_nvfp4_sm100(x, mma_n)`** — standalone quantizer, outputs layout matched to the kernel path

#### MXFP4
- **`linear_mxfp4_sm100(x, w, w_sf, residual, output)`** — auto-quantizes float32 activations then runs GEMM
- **`linear_mxfp4_sm100_no_quantization(x_q, x_sf, w, w_sf, residual, output)`** — takes pre-quantized inputs
- **`quantize_mxfp4_sm100(x, mma_n)`** — standalone quantizer, outputs layout matched to the kernel path

### Supported Shapes
- N and K are multiples of 256
- **Small batch (swapAB path):** M=1–128
- **Large batch (1d2d path):** M is a multiple of 256

### Scale Factor Layouts

- **swapAB:** per-tile `[num_n_tiles, sf_k_outer, 32, 4, 4]` — TMA indexes by n-tile
- **1d2d:** interleaved `[M/128, sf_k_outer, 32, 4, 4]` — TMA indexes by 128-row block
- Layout is selected automatically by the quantizer based on `mma_n`

### Adaptive MMA_N (swapAB)

Mirrors trtllm heuristic to avoid grid explosion at large N:

| M | MMA_N |
|---|-------|
| ≤8 | 8 |
| ≤16 | 16 |
| ≤32 | 32 |
| ≤64 | 64 |
| ≤128 | 128 |

### Optional Residual

Both entry points accept an optional residual tensor (bias add fused into epilogue).

### Benchmarking

```bash
# NVFP4 GEMM only (pre-quantized), M=1..128, N=128, K=768
python profile_linear_1d2d_nvfp4.py --m-values "$(seq -s, 1 128)" --n-values 128 --k-values 768 --no-scaled-mm --no-flashinfer --warmup 200 --reps 500

# MXFP4 GEMM only (pre-quantized)
python profile/profile_linear_1d2d_mxfp4.py
```

### Correctness Tests

```bash
python test_linear_1d2d_nvfp4.py
python test_linear_1d2d_mxfp4.py
```

- **Tests 1–4:** 1d2d path (M=4096), sequential and random inputs, with and without residual, against `torch._scaled_mm`
- **Test 5:** swapAB path (M=1–128), per-tile SF layout correctness, against `torch._scaled_mm`
- **Test 6:** auto-quantize entry point matches explicit quantize + no-quant path

## Performance

All numbers are GEMM-only (pre-quantized activations), measured with 200 warmup + 500 timed reps on B200 (SM100a).

### NVFP4

#### B=1, N=128, K=768

| B | N | K | Path | Latency |
|---|---|---|------|---------|
| 1 | 128 | 768 | swapAB | 3.9 µs |

#### B sweep (N=128, K=768)

| B | MMA_N | Custom | Custom+residual | fi_cutlass | fi_trtllm |
|---|-------|--------|-----------------|------------|-----------|
| 1 | 8 | 3.9 µs | 4.3 µs | 6.2 µs | 6.2 µs |
| 2 | 8 | 4.1 µs | 4.6 µs | 6.2 µs | 6.2 µs |
| 4 | 8 | 4.3 µs | 5.0 µs | 6.2 µs | 6.2 µs |
| 8 | 8 | 4.1 µs | 5.5 µs | 6.2 µs | 6.2 µs |
| 16 | 16 | 4.2 µs | 6.8 µs | 6.2 µs | 6.2 µs |
| 32 | 32 | 4.7 µs | 9.8 µs | 6.2 µs | 6.2 µs |
| 64 | 64 | 5.6 µs | 15.8 µs | 6.2 µs | 6.2 µs |
| 128 | 128 | 7.8 µs | 27.8 µs | 6.2 µs | 6.2 µs |

#### K sweep (B=1, N=1024)

| B | N | K | Custom | Custom+residual | fi_cutlass | fi_trtllm |
|---|---|---|--------|-----------------|------------|-----------|
| 1 | 1024 | 768 | 3.9 µs | 4.3 µs | 6.2 µs | 6.2 µs |
| 1 | 1024 | 2048 | 5.2 µs | 5.6 µs | 6.2 µs | 6.2 µs |
| 1 | 1024 | 7168 | 8.8 µs | 9.2 µs | 10.3 µs | 10.3 µs |

### MXFP4

Baseline (mode=cuda-graph-replay, warmup=200, reps=100, residual=False):

| M | N | K | Path | Custom | Custom TFLOPS | fi | fi TFLOPS | Speedup |
|---|---|---|------|--------|---------------|----|-----------|---------|
| 1 | 128 | 768 | swapAB | 5.74 µs | 0.0 | 4.36 µs | 0.0 | 0.76x |
| 2 | 128 | 768 | swapAB | 6.03 µs | 0.1 | 4.36 µs | 0.1 | 0.72x |
| 4 | 128 | 768 | swapAB | 6.03 µs | 0.1 | 4.31 µs | 0.2 | 0.71x |
| 8 | 128 | 768 | swapAB | 6.09 µs | 0.3 | 4.35 µs | 0.4 | 0.72x |
| 16 | 128 | 768 | swapAB | 5.98 µs | 0.5 | 4.36 µs | 0.7 | 0.73x |
| 32 | 128 | 768 | swapAB | 5.91 µs | 1.1 | 4.37 µs | 1.4 | 0.74x |
| 64 | 128 | 768 | swapAB | 5.94 µs | 2.1 | 4.36 µs | 2.9 | 0.73x |
| 128 | 128 | 768 | swapAB | 6.02 µs | 4.2 | 4.42 µs | 5.7 | 0.73x |
