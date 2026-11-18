Selected plan files for the 132-SM H100: `<kind>_e<experts>_r<requests>.json` (`plan_path` in `gpt_oss_static.py`; kind
`sliding`, `full` or `head`). A plan holds the grids the compiler may change (the QKV GEMM's task count, the gate's
router split S) and the per-SM task lists. Configurations without a checked-in plan run round robin.

The demo makes them with MPK's standard pipeline, as the GLM-5.3 demos do:

    python gpt_oss_static.py CKPT --batch R --tune costs.json [--plans DIR] [--time 300]

For each kernel, `--tune` builds every plan of `plan_gen.collection_plans` plus the current plan (the timing build,
L2 emptied before each launch, page table at mid-sequence), appends the per-task cost records to `costs.json`, and runs
`search.search_plan` from the current plan. The solver picks the grids (e.g. S of the gate) and the placement. The result
is written here, and the spans of both plans are printed.

The checked-in plans are `--tune` results for the layer graphs then ending at an in-kernel MoE pool (since removed: its
tasks are dropped from the files, the other nodes' ids and tasks unchanged) and, in the full-attention layers, the split
attention pool. They cover the configurations with documented per-layer gains: 20B full attention at 8, 16 and 32
requests, and full and sliding attention at 32 requests for both models. `--tune` keeps whichever of the current and the
solver's plan measures faster. On 20B with 32 distinct prompts, the full layer tuned vs round robin: 8 requests 177 ->
170, 16 requests 230 -> 223, 32 requests 374 -> 345 us; at 32 requests on 120B: sliding 465 -> 425, full 528 -> 482 us.

`sliding_e128_r2.json` and `full_e128_r2.json` are `--tune` results for 120B at two requests (round robin vs tuned:
sliding 46.6 -> 45.2, full 50.3 -> 44.4 us).

Full attention decodes split-KV (`sm90_attention_split`, `sm90_attention_merge`). A split covers one request's range
of 32-position tiles for all 8 KV heads, a warp a head, so it reads whole 1 KB cache rows. The splits are about one per
SM over the step's tiles (`SM90_ATTENTION_UNITS_PER_SM`); per warp, a 3-stage cp.async ring of 32-position K and V tiles
feeds bf16 `mma.sync` (QK^T via `ldmatrix`, PV via `ldmatrix.trans`) with the online softmax in log2 units. The last
split of a request also writes the step's K (RoPE'd) and V to the cache and folds that position in. The merge combines
a head's splits with its sink. At 2 requests x 100K positions the attention pair takes 142 us in isolation (2.9 TB/s of
KV), down from about 640 us for the previous chunked pool.

On 120B, 2 requests, a 100K-token real prompt and 64 new tokens (`max_model_len` 100096), MPK matches the vLLM and
SGLang references' tokens for all 64 positions (and the two duplicate requests each other) at 6.17 ms/token decode.
vLLM 0.33 (FA3, bf16 KV) measures 7.149 and SGLang 12.164 ms/token, paired `(T64 - T1) / 63` wall time.

The layer graph ends at the gate: the routed experts run as two kernels after the layer kernel (`sm90_moe_split`). Their
grid has a row per expert the step routes to (at most requests x top-k, not every expert), and at one or two requests a
deeper 8-stage weight ring. Against the all-experts grid, 4-stage ring (decode us/step, tokens identical): 120B batch 1
4030 -> 3864, batch 2 4981 -> 4199, batch 8 7746 -> 7135; 20B batch 1 2741 -> 2645, batch 8 4231 -> 4125.

The weights the layer streams first can be prefetched into L2 at its start, with the shared `STATIC_L2_PREFETCH_MB`
(`StaticMegakernel.l2_prefetch_weights`, as the GLM demos). On these H100s, 12 MB measured best at batch one (20B
2834 -> 2791, 120B 4133 -> 4060 us/step).

`*_e128_r64_t64.json` (and `full_e128_r16_t16.json`, `sliding_e128_r8.json`, `full_e128_r8.json`) are `--tune` results
for 120B's current layer graphs (span, solved vs round robin: 64 requests sliding 102 -> 90, full 130 -> 120 us; 16
requests full 62 -> 60 us; 8 requests sliding 49 -> 47, full 56 -> 53 us). At 32 requests and the heads the solver's
plans measured within noise of the existing ones.

From 32 requests the routed experts run as triton_kernels' `matmul_ogs` (`triton_moe.py`, the kernel SGLang and vLLM
use on Hopper; `GPTOSS_MOE=triton|mpk` overrides): the gate's routing turned into its ragged metadata by one kernel
(`_route`), W13 + SwiGLU, W2 with the routing weights, then `_combine` (x1 + the token's four rows, fp32). Both GEMMs at
4 warps with the scales swizzled to match (more CTAs an SM; 120B, 64 tokens on 57 experts: W13 + W2 341 -> 286 us; the
engines run 8). The step must be one CUDA graph (`sweep_mpk.py` `GRAPH=1`): launched eagerly from Python it is
host-bound (about 37 ms/step at 32 or 64 requests); `gpt_oss_static.py` captures it (`--eager`: not).
`triton_kernels` is not a Mirage dependency: SGLang's and vLLM's installs ship it (or `pip install` it from the
Triton repository's `python/triton_kernels`); without it the default falls back to `sm90_moe_split`.

Decode ms/step, 120B, one H100 (Slurm node), distinct real prompts (`sweep_prompts.py`), MPK as one CUDA graph a step;
vLLM 0.33 (FA3) and SGLang 0.5.23 (FA3) one engine process per batch, paired `(T64 - T32) / 32` wall time:

| batch | context | MPK | vLLM | SGLang |
|---|---|---|---|---|
| 2 | 128 / 8K / 32K / 98K | 4.27 / 4.45 / 5.06 / 6.52 | 5.11 / 5.24 / 5.95 / 7.39 | 5.70 / 6.00 / 6.34 / 12.16 |
| 8 | 128 / 8K / 24K | 6.96 / 7.44 / 9.29 | 8.68 / 8.62 / 10.81 | 8.83 / 9.26 / 16.42 |
| 16 | 128 / 4K / 12K | 9.42 / 10.03 / 11.57 | 12.04 / 11.33 / 13.32 | 11.70 / 11.94 / - |
| 32 | 128 / 2K / 6K | 12.02 / 13.19 / 14.22 | 13.76 / 14.62 / 15.56 | 14.61 / 15.12 / 33.34 |
| 64 | 128 / 1K / 2K / 3K | 16.05 / 16.83 / 17.71 / 18.30 | 16.59 / 17.82 / 18.89 / 19.39 | 17.73 / 18.64 / 31.67 / 40.69 |

(64 x 128 for vLLM: the median of 9 pairs, a separate run.) Before the matmul_ogs path and the
graph, MPK took 16.4 / 17.0 / 18.1 ms at 32 requests and 30.7 / 34.0 ms at 64 (128 / 2K). Correctness: per-layer fp64
(`layer_check.py`, 20B and 120B at 32 requests: the MoE's relRMS at or below Hugging Face bf16's) and GSM8K 200
questions at 32 requests (120B 0.940 and 0.945 in two runs, 20B 0.870; the sm90_moe_split path 0.955 and 0.870).
