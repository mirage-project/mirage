# vLLM baseline capture — 2026-07-25 (binding for M3/M4 targets)

Protocol: `docs/qwen35/bench-protocol.md` (pinned workload input=256 / output=1024, greedy,
FP8, no spec-decode, CUDA graphs on, `language_model_only=off` per the ruling below).
Engine: vLLM 0.25.1, torch 2.11.0+cu130, flashinfer 0.6.13, checkpoint revision `9d1823d2`.
Host: catalyst-B200 GPU 5 (exclusive; co-tenants on GPUs 0/1/4 throughout — see protocol §6
escalation rule). Kernel identity asserted per run: CUTLASS block-scale dense (DeepGEMM
auto-disabled marker), FlashInfer TRT-LLM fp8 block-scale MoE.

## Binding decode-throughput table (median tok/s; MPK must beat every row, AC-4)

| bs | decode tok/s | source | dispersion | e2e (s, AC-5 ref) |
|----|-------------:|--------|-----------:|------------------:|
| 1  |  285.5 | `full/bs1.json` (3 reps, 1 boot)  | 0.06 % | 3.60 |
| 2  |  529.8 | `full/bs2.json` (3 reps, 1 boot)  | 0.50 % | 3.89 |
| 4  |  934.4 | `full/bs4.json` (3 reps, 1 boot)  | 3.48 % | 4.45 |
| 8  | 1692.5 | merged `full`+`full-hibs` (8 reps, 2 boots) | 3.00 % (boot medians Δ2.06 %) | 4.953 |
| 16 | 3018.1 | merged `full`+`full-hibs` (8 reps, 2 boots) | 3.43 % (boot medians Δ1.07 %) | 5.568 |

bs 8/16 single-boot runs exceeded the 5 % bound (5.24 % / 6.41 %); per the protocol §6
escalation rule the binding value is the merged-boot median (rule added 2026-07-25 from this
data). Raw per-rep arrays, GPU clock snapshots, co-tenant records, and output-id digests are in
the per-bs JSONs.

## `language_model_only` ruling (`ruling/`)

OFF (vLLM default) is pinned: the fused-path ON showed no speed advantage at our shapes
(probe: 183.0 vs 183.3 tok/s at bs1; 738.5 vs 745.2 at bs4 — within noise, OFF marginally
faster), OFF is the config that matched the HF reference 64/64 on p01, and both modes select
identical dense/MoE kernels (log-verified). Token divergence between modes at bs4 is the
fused-vs-unfused rounding difference; with OFF pinned it is moot for the baseline.

## Run provenance

- `full/` — sweep bs {1,2,4,8,16}, 3 reps + 1 warmup, single boot (GPU 5).
- `full-hibs/` — second independent boot, bs {8,16}, 5 reps + 1 warmup (GPU 5).
- `ruling/` — `--mode ruling` A/B at bs {1,4}, 64-token windows.
- Engine subprocess requires the venv `bin` on PATH (`ninja` lives there; the lmo=off vision
  path JIT-compiles and hard-fails without it).
- The `--log-file` passed to `bench_vllm.py` must be the SAME file the caller redirects
  stdout/stderr into (`> F 2>&1` with `--log-file F`) — the fairness assertions grep it.
