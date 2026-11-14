Plan files for the 132-SM H100, one per kernel graph: `<kind>_e<experts>_r<requests>.json` (`plan_path` in
`gpt_oss_static.py`; kind `sliding`, `full` or `head`). A plan holds the grids the compiler may change (the QKV GEMM's
task count, the gate's router split S) and the per-SM task lists. A kernel with no plan file here runs round robin.

The demo makes them with MPK's standard pipeline, as the GLM-5.3 demos do:

    python gpt_oss_static.py CKPT --batch R --tune costs.json [--plans DIR] [--time 300]

For each kernel, `--tune` builds every plan of `plan_gen.collection_plans` plus the current plan (the timing build,
L2 emptied before each launch, page table at mid-sequence), appends the per-task cost records to `costs.json`, and runs
`search.search_plan` from the current plan. The solver picks the grids (e.g. S of the gate) and the placement. The result
is written here, and the spans of both plans are printed.

The plans here are `--tune` results for the graphs with the MoE pool and, in the full-attention layers, the split
attention pool (20B at one, two, four and eight requests, 120B at one and eight; the head's is the same graph for both
models; both models at 16 and 32 requests in the 16- and 32-row buckets, `_t16`, `_t32`). `--tune` keeps whichever of
the current and the solver's plan measures faster. 20B on 32 distinct prompts, the full layer tuned vs round robin: 8
requests 177 -> 170, 16 requests 230 -> 223, 32 requests 374 -> 345 us; 120B at 32 requests: sliding 465 -> 425, full
528 -> 482 us.

From four requests the routed experts run as two kernels after the layer kernel (`sm90_moe_split`; `MOE_SPLIT=0 / 1`
forces either path), and the layer graph ends at the gate. `load_plan` drops the MoE pool's and the combine's tasks
from these plans: the other nodes keep their ids and tasks, so the same files serve both graphs.

The weights the layer streams first can be prefetched into L2 at its start, with the shared `STATIC_L2_PREFETCH_MB`
(`StaticMegakernel.l2_prefetch_weights`, as the GLM demos). On these H100s, 12 MB measured best at batch one (20B
2834 -> 2791, 120B 4133 -> 4060 us/step).
