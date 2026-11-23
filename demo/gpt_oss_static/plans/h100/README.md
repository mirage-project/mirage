Plan files for the 132-SM H100: `<kind>_e<experts>_r<requests>[_t<rows>].json` (`plan_path` in `gpt_oss_static.py`;
kind `sliding`, `full` or `head`). A plan holds the grids the compiler may change (the QKV GEMM's task count, the gate's
router split) and the per-SM task lists. Configurations without a plan here run round robin.

They come from MPK's standard pipeline, as the GLM-5.3 demos' do:

    python gpt_oss_static.py CKPT --batch R --tune costs.json [--plans DIR] [--time 300]

`--tune` builds every plan of `plan_gen.collection_plans` plus the current one (timing build, L2 emptied before each
launch, page table at mid-sequence), appends the per-task cost records to `costs.json`, runs `search.search_plan` from
the current plan, and writes whichever of the two measures faster.

Only plans with a clear per-layer win are checked in (layer span, round robin -> tuned): 20B full attention at 32
requests 65.3 -> 60.7 us; 120B at 64 requests sliding 102 -> 90, full 130 -> 120; 120B full attention at 2 requests
50.3 -> 44.4 us. The 120B 32-request plans were tuned when the layer still held the MoE (sliding 465 -> 425, full
528 -> 482 us; its tasks since dropped from the files); re-tuning the current layers found nothing faster. Plans for 8
and 16 requests and the heads won 2-5% on a layer, under 1.5% of a step, and are not kept (the PR's benchmark table
was measured with them).
