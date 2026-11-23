Plan files for the 148-SM B200: `<kind>_e<experts>_r<requests>[_t<rows>].json` (`plan_path` in `gpt_oss_static.py`;
kind `sliding`, `full`, `full_a`, `full_b` or `head`), made by the same `--tune` pipeline as the H100 plans (see
`../h100/README.md`), with FlashInfer's trtllm-gen MXFP4 MoE outside the layer kernel. Each is the faster of round robin /
the solver's plan on the layer span (timing build, L2 emptied); e.g. full attention at 2 requests 52.5 -> 45.7 us,
sliding at 64 requests (K split floor) 57.9 -> 48.4 us.

The full layers (`full_attention` in `gpt_oss_static.py`): up to 4096 cache positions `full`, the split attention's
direct mode (no merge node); past that two kernels around FlashInfer's trtllm-gen decode, `full_a` (the norm, the QKV
GEMM, `gpt_oss_rope_append`) and `full_b` (o_proj, the gate). Their plans keep the merge-era full plans' grids for the
nodes they share (the QKV / o_proj K parts, the router split) with the attention's SMs' lists rebuilt; the
`test_gpt_oss_b200_plans.py` test checks every file fits its graph.

`fi_tactics.json`: FlashInfer MoE tactics (`[tile_n, config]`) pinned per batch, applied only on the FlashInfer build and
GPU named in the file (tactic ids are theirs; `flashinfer_moe.pinned_tactic`). FlashInfer's tuner times each tactic
once, cold, on synthetic routing, so its pick moves between runs; at 64 requests the in-step best `[16, 769]` measured
-0.15 .. -0.18 ms a step against the tuner's pick (same tokens). At 8 / 16 / 32 requests the tuner's pick tied the
in-step best, so nothing is pinned.
