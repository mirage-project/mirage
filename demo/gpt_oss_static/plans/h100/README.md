Plan files for the 132-SM H100, one per kernel graph: `<kind>_e<experts>_r<requests>.json` (`plan_path` in
`gpt_oss_static.py`; kind `sliding`, `full` or `head`). A plan holds the grids the compiler may change (the QKV GEMM's
task count, the gate's router split S) and the per-SM task lists. A kernel with no plan file here runs round robin.

The demo makes them with MPK's standard pipeline, as the GLM-5.3 demos do:

    python gpt_oss_static.py CKPT --batch R --tune costs.json [--plans DIR] [--time 300]

For each kernel, `--tune` builds every plan of `plan_gen.collection_plans` plus the current plan (the timing build,
L2 emptied before each launch, page table at mid-sequence), appends the per-task cost records to `costs.json`, and runs
`search.search_plan` from the current plan. The solver picks the grids (e.g. S of the gate) and the placement. The result
is written here, and the spans of both plans are printed.

The `*_r1.json` plans are the solver plans from earlier batch-one tuning, carried over unchanged (20B: S = 4, 120B: S = 8).

The weights the layer streams first can be prefetched into L2 at its start, with the shared `STATIC_L2_PREFETCH_MB`
(`StaticMegakernel.l2_prefetch_weights`, as the GLM demos). On these H100s, 12 MB measured best at batch one (20B
2834 -> 2791, 120B 4133 -> 4060 us/step).
