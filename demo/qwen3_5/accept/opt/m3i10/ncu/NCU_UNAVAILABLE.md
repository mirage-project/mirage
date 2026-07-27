# Nsight Compute cannot collect counters on catalyst-B200

Part C of M3-I10 asked for `ncu` detail on the top offenders. **NCU hardware-counter collection
does not work on this box at all.** This is an environment limitation, not something about our
kernels, CUDA graphs, or the vLLM engine. `ncu/roofline.csv` is the substitute.

## What was tried

1. `ncu --graph-profiling node` against the real engine, CUDA graphs on, `--profile-from-start off`
   with `cudaProfilerStart` raised inside a steady decode window (`scripts/ncu_probe.py`).
   Boot under `ncu` is slow — vLLM's own "Initial profiling/warmup run took **766.69 s**" — and it
   then failed at the first matched kernel.
2. Same, with `enforce_eager=True` (no CUDA graphs at all) and
   `max_num_batched_tokens=1024` to shrink the warmup forward. Same failure.
3. A three-line control: `torch.randn(4096,4096).cuda() @ ...` under
   `ncu --section SpeedOfLight --launch-count 2`. **Same failure on a trivial kernel**, which is
   what makes this an environment verdict rather than a kernel-specific one.
4. `/usr/local/cuda-{12.8,13.0,13.2}/bin/ncu` — all three symlink to the same install,
   Nsight Compute **2026.1.0.0 (build 37166530)**, so there is no alternative version on the box.

Every attempt produced:

```
==ERROR== Failed to prepare kernel for profiling
==ERROR== An error was reported by the counter measurement library:
==ERROR== Unknown error on device 0.
==ERROR== Failed to profile "distribution_elementwise_grid..." in process 3929334
==ERROR== The application returned an error code (9).
```

## What it is not

- **Not a permissions problem.** `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 0`,
  so profiling is not restricted to admin users. The usual `ERR_NVGPUCTRPERM` message never
  appears.
- **Not CUDA graphs.** It reproduces with `enforce_eager=True` and on a bare PyTorch matmul.
- **Not co-tenancy.** The final attempt ran on GPU 2 with 5 MiB used and 0 % utilisation.
- **Not our kernel selection.** The control kernel is `distribution_elementwise_grid_stride_kernel`
  from `torch.randn`.

Driver `595.58.03`, Nsight Compute 2026.1.0.0, B200 sm_100. Most likely a driver / Nsight version
pairing issue on this fleet; it would need a box-level fix (driver or Nsight upgrade) to resolve.

## MPK could not be NCU-profiled either, by construction

Even with a working `ncu`, it cannot decompose MPK. The megakernel is a **single persistent
kernel**: NCU would report one launch covering the whole decode step, with no per-task breakdown.
All MPK per-task numbers in this issue come from the committed perfetto/profiler-buffer tables
(M3-I1 `opt/pertask_by_bs.csv`, `opt/tables/bs*_attrib.json`).

## The substitute

`roofline.csv` / `roofline.json`, produced by `scripts/roofline.py`. It answers the question the
SOL section would have answered — how far the incumbent kernel is from the memory roof, i.e. how
much room a replacement actually has — by combining:

- **bytes moved per decode step**, derived from the exact shapes in `docs/qwen35/vllm-graph.md`
  §3.3 (the complete GEMM inventory), §4.1 (GDN state) and §4.2 (paged KV), and
- **measured median µs/step** from `tables/bs*_kernels.csv` (this issue).

Its calibration check is `lm_head`: 1.017 GB of bf16 weights in 150.7 µs = **6.75 TB/s, 84 % of
the 8 TB/s B200 HBM3e roof**, i.e. 1.2× off roofline. A memory-bound kernel that we know must be
at the roof lands at the roof, so the method is sound for the rest of the table.

What it does not give you, and NCU would have: achieved occupancy, register/shared-memory
pressure, launch configuration, warp stall reasons, and the compute-vs-memory SOL split. Where a
ferret task needs those, it should measure them on its own standalone reproduction, on a box where
`ncu` works.
