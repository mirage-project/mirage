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

The layer graph ends at the gate: the routed experts run as two kernels after the layer kernel (`sm90_moe_split`).

The weights the layer streams first can be prefetched into L2 at its start, with the shared `STATIC_L2_PREFETCH_MB`
(`StaticMegakernel.l2_prefetch_weights`, as the GLM demos). On these H100s, 12 MB measured best at batch one (20B
2834 -> 2791, 120B 4133 -> 4060 us/step).
