"""Round-trip inventory for the Qwen3.5 weight loader (M2-I8 acceptance 1).

The property under test: **every tensor in the checkpoint is consumed exactly
once, or skipped with an explicit reason.** A weight that is never read is a
silently-zero projection; one read into two runtime slots means two slots
disagree about who owns it. Neither shows up as a crash — the model just
produces subtly wrong tokens, which is precisely what AC-3 cannot absorb.

Two phases:

  `plan`  — pure, index-only. Maps all 64 196 keys of the 14-shard index and
            asserts the counts, the destination uniqueness, and that the only
            skips are the vision tower and the MTP draft layer. No GPU, no
            tensor I/O, so it can run anywhere the index file is reachable.
  `load`  — executes the plan against the real checkpoint (GPU, ~35 GB) and
            asserts the loader's own `assert_round_trip()`: planned == read,
            each exactly once. Also checks every runtime tensor's shape/dtype
            against `vllm-graph.md` §5.2-§5.4.

Run:
    python .../test_loader_inventory.py --snapshot <hf snapshot dir> [--plan-only]
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "python"))

from mirage.mpk.models.qwen3_5.weight_loader import (  # noqa: E402
    Qwen35Config, Qwen35WeightLoader, plan_checkpoint)

# vllm-graph.md §5.4 totals, quoted so a checkpoint swap fails loudly here
# instead of somewhere numeric.
EXPECT = {
    "shards": 14,
    "total": 64196,
    "visual": 333,
    "mtp": 1560,
    "lang": 62302,
    "lm_head": 1,
}


def phase_plan(snapshot):
    config = Qwen35Config.from_json(os.path.join(snapshot, "config.json"))
    with open(os.path.join(snapshot, "model.safetensors.index.json")) as f:
        index = json.load(f)
    keys = list(index["weight_map"])
    plan = plan_checkpoint(keys, config)

    failures = []

    def check(name, got, want):
        ok = got == want
        print(f"  {'OK ' if ok else 'FAIL'} {name:34s} {got} (expected {want})")
        if not ok:
            failures.append(f"{name}: {got} != {want}")

    shards = len(set(index["weight_map"].values()))
    check("shards", shards, EXPECT["shards"])
    check("checkpoint keys", len(keys), EXPECT["total"])
    check("mapped + skipped", len(plan.consumed) + len(plan.skipped), len(keys))
    check("mapped", len(plan.consumed),
          EXPECT["lang"] + EXPECT["lm_head"])
    check("skipped", len(plan.skipped), EXPECT["visual"] + EXPECT["mtp"])
    check("skipped: vision tower",
          sum(1 for k in plan.skipped if k.startswith("model.visual.")),
          EXPECT["visual"])
    check("skipped: mtp draft",
          sum(1 for k in plan.skipped if k.startswith("mtp.")), EXPECT["mtp"])
    check("every skip has a reason",
          sum(1 for r in plan.skipped.values() if r and len(r) > 10),
          len(plan.skipped))

    dests = list(plan.consumed.values())
    check("destinations unique", len(set(dests)), len(dests))

    # the layer split the config declares must be the one the plan used
    check("GDN layers", len(config.gdn_layers), 30)
    check("full-attention layers", len(config.attn_layers), 10)
    gdn_keys = sum(1 for k in plan.consumed if ".linear_attn." in k)
    attn_keys = sum(1 for k in plan.consumed if ".self_attn." in k)
    check("GDN tensors (30 x 12)", gdn_keys, 30 * 12)
    check("full-attention tensors (10 x 10)", attn_keys, 10 * 10)
    expert_keys = sum(1 for k in plan.consumed if ".mlp.experts." in k)
    check("routed-expert tensors (40 x 256 x 6)", expert_keys, 40 * 256 * 6)

    print("\n  plan summary:")
    for k, v in sorted(plan.summary().items()):
        print(f"    {k}: {v}")
    return failures


def _shape(t):
    return tuple(t.shape)


def phase_load(snapshot, qk_dense_path):
    import torch

    loader = Qwen35WeightLoader(snapshot, device="cuda",
                                qk_dense_path=qk_dense_path)
    c = loader.config
    w = loader.load()                      # asserts round-trip internally
    report = loader.inventory_report()
    print("  inventory:", json.dumps(report, indent=2))

    H, E = c.hidden_size, c.num_experts
    inter, si = c.moe_intermediate_size, c.shared_expert_intermediate_size
    B = 128
    failures = []

    def want(name, shape, dtype):
        if name not in w:
            failures.append(f"missing runtime tensor {name}")
            return
        t = w[name]
        if _shape(t) != shape or t.dtype != dtype:
            failures.append(
                f"{name}: {_shape(t)}/{t.dtype} != {shape}/{dtype}")

    bf, f32, fp8 = torch.bfloat16, torch.float32, torch.float8_e4m3fn
    want("embed_tokens", (c.vocab_size, H), bf)
    want("lm_head", (c.vocab_size, H), bf)
    want("model_norm", (H,), bf)
    for i in range(c.num_layers):
        want(f"layer_{i}_input_layernorm", (H,), bf)
        want(f"layer_{i}_post_attention_layernorm", (H,), bf)
        want(f"layer_{i}_router", (E, H), bf)
        want(f"layer_{i}_shared_expert_gate", (1, H), bf)
        want(f"layer_{i}_shared_gate_up", (2 * si, H), fp8)
        want(f"layer_{i}_shared_gate_up_scale", (2 * si // B, H // B), f32)
        want(f"layer_{i}_shared_down", (H, si), fp8)
        want(f"layer_{i}_shared_down_scale", (H // B, si // B), f32)
        want(f"layer_{i}_w13", (E, 2 * inter, H), fp8)
        want(f"layer_{i}_w13_scale", (E, 2 * inter // B, H // B), f32)
        want(f"layer_{i}_w2", (E, H, inter), fp8)
        want(f"layer_{i}_w2_scale", (E, H // B, inter // B), f32)
        if c.layer_types[i] == "linear_attention":
            want(f"layer_{i}_gdn_in_proj_qkv", (c.conv_dim, H), fp8)
            want(f"layer_{i}_gdn_in_proj_qkv_scale", (c.conv_dim // B, H // B), f32)
            want(f"layer_{i}_gdn_in_proj_z", (c.gdn_z_dim, H), fp8)
            want(f"layer_{i}_gdn_in_proj_z_scale", (c.gdn_z_dim // B, H // B), f32)
            want(f"layer_{i}_gdn_in_proj_ba", (2 * c.linear_num_value_heads, H), bf)
            want(f"layer_{i}_gdn_conv1d", (c.conv_dim, c.linear_conv_kernel_dim), bf)
            want(f"layer_{i}_gdn_alog_dtbias", (2, c.linear_num_value_heads), f32)
            want(f"layer_{i}_gdn_norm", (c.linear_value_head_dim,), f32)
            want(f"layer_{i}_gdn_out_proj", (H, c.gdn_z_dim), fp8)
            want(f"layer_{i}_gdn_out_proj_scale", (H // B, c.gdn_z_dim // B), f32)
        else:
            want(f"layer_{i}_qkvg_proj", (c.qkvg_dim, H),
                 fp8 if qk_dense_path == "fp8" else bf)
            if qk_dense_path == "fp8":
                want(f"layer_{i}_qkvg_proj_scale", (c.qkvg_dim // B, H // B), f32)
            want(f"layer_{i}_q_norm", (c.head_dim,), bf)
            want(f"layer_{i}_k_norm", (c.head_dim,), bf)
            want(f"layer_{i}_o_proj", (H, c.num_attention_heads * c.head_dim), fp8)
            want(f"layer_{i}_o_proj_scale",
                 (H // B, c.num_attention_heads * c.head_dim // B), f32)

    print(f"  runtime tensors built: {len(w)}")
    if loader.transform_stats:
        worst = max(s["frob_rel_vs_exact_permute"]
                    for s in loader.transform_stats.values())
        frac = max(s["rescaled_fraction"] for s in loader.transform_stats.values())
        print(f"  qk permute (fp8): worst frob_rel vs exact permute {worst:.3e}, "
              f"worst rescaled fraction {frac:.3f}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--qk-dense-path", default="fp8", choices=["fp8", "bf16"])
    args = ap.parse_args()

    print("== phase plan (index only) ==")
    failures = phase_plan(args.snapshot)
    if not args.plan_only:
        print("\n== phase load (real checkpoint) ==")
        failures += phase_load(args.snapshot, args.qk_dense_path)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nLOADER ROUND-TRIP INVENTORY PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
