"""
Test: Full BF16 Qwen3 MoE block at batch=8 via PersistentKernel test_mode.

Regression for #740: validates the end-to-end MoE pipeline with batch>1,
including router weight layout (batch, topk) consumed by moe_mul_sum_add.

Pipeline (matches demo/qwen3/demo_30B_A3B.py):
  gate linear -> moe_topk_softmax -> moe_w13 -> moe_silu_mul ->
  moe_w2 -> moe_mul_sum_add

Qwen3-30B-A3B dims:
  hidden=2048, moe_intermediate=768, fused=1536, experts=128, topk=8

Run:
    python tests/runtime_python/test_mode/test_moe_pipeline_batch_testmode.py
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

# Qwen3-30B-A3B configuration (matches HF config.json / demo_30B_A3B.py)
BATCH_SIZE = 8
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768
FUSED_OUTDIM = 2 * INTERMEDIATE_SIZE  # 1536 (gate + up fused)
NUM_EXPERTS = 128
NUM_EXPERTS_PER_TOK = 8
WORLD_SIZE = 1


def _make_pk():
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=WORLD_SIZE,
        max_num_batched_tokens=BATCH_SIZE,
        max_num_batched_requests=BATCH_SIZE,
    )
    return PersistentKernel(**params)


def _pytorch_moe_block_ref(hidden, gate_weight, w13, w2, residual):
    """PyTorch reference for the full Qwen3 MoE block."""
    gate_logits = hidden.float() @ gate_weight.float().T
    probs = F.softmax(gate_logits, dim=-1)
    topk_w, topk_idx = probs.topk(NUM_EXPERTS_PER_TOK, dim=-1)
    topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).float()

    routing = torch.zeros(NUM_EXPERTS, BATCH_SIZE, dtype=torch.int32, device=hidden.device)
    for t in range(BATCH_SIZE):
        for k in range(NUM_EXPERTS_PER_TOK):
            routing[topk_idx[t, k].item(), t] = k + 1

    mlp_per_expert = torch.zeros(
        BATCH_SIZE, NUM_EXPERTS_PER_TOK, HIDDEN_SIZE,
        dtype=torch.float32, device=hidden.device,
    )
    for t in range(BATCH_SIZE):
        for s in range(NUM_EXPERTS_PER_TOK):
            e = topk_idx[t, s].item()
            w13_out = hidden[t].float() @ w13[e].float().T
            gate_part = w13_out[:INTERMEDIATE_SIZE]
            up_part = w13_out[INTERMEDIATE_SIZE:]
            silu_out = F.silu(gate_part) * up_part
            mlp_per_expert[t, s] = silu_out @ w2[e].float().T

    weighted = (mlp_per_expert * topk_w.unsqueeze(-1)).sum(dim=1)
    final = (weighted + residual.float()).to(hidden.dtype)
    return final, topk_w, routing, gate_logits


def test_moe_pipeline_batch():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    print(f"\n{'='*70}")
    print("Test: Full Qwen3 MoE block at batch=8 (BF16)")
    print(
        f"  B={BATCH_SIZE}, H={HIDDEN_SIZE}, I={INTERMEDIATE_SIZE}, "
        f"fused={FUSED_OUTDIM}, E={NUM_EXPERTS}, topk={NUM_EXPERTS_PER_TOK}"
    )

    hidden = torch.randn(BATCH_SIZE, HIDDEN_SIZE, dtype=dtype, device=device) * 0.1
    residual = torch.randn(BATCH_SIZE, HIDDEN_SIZE, dtype=dtype, device=device) * 0.1
    gate_weight = (
        torch.randn(NUM_EXPERTS, HIDDEN_SIZE, dtype=dtype, device=device)
        / math.sqrt(HIDDEN_SIZE)
    )
    w13_weight = (
        torch.randn(NUM_EXPERTS, FUSED_OUTDIM, HIDDEN_SIZE, dtype=dtype, device=device)
        / math.sqrt(HIDDEN_SIZE)
    )
    w2_weight = (
        torch.randn(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=dtype, device=device)
        / math.sqrt(INTERMEDIATE_SIZE)
    )

    gate_out = torch.zeros(BATCH_SIZE, NUM_EXPERTS, dtype=dtype, device=device)
    topk_w = torch.zeros(BATCH_SIZE, NUM_EXPERTS_PER_TOK, dtype=torch.float32, device=device)
    routing = torch.zeros(NUM_EXPERTS, BATCH_SIZE, dtype=torch.int32, device=device)
    mask = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
    mlp_mid = torch.zeros(
        BATCH_SIZE, NUM_EXPERTS_PER_TOK, FUSED_OUTDIM, dtype=dtype, device=device,
    )
    silu_out = torch.zeros(
        BATCH_SIZE, NUM_EXPERTS_PER_TOK, INTERMEDIATE_SIZE, dtype=dtype, device=device,
    )
    mlp_out = torch.zeros(
        BATCH_SIZE, NUM_EXPERTS_PER_TOK, HIDDEN_SIZE, dtype=dtype, device=device,
    )
    final_out = torch.zeros(BATCH_SIZE, HIDDEN_SIZE, dtype=dtype, device=device)

    ref_final, ref_topk_w, ref_routing, ref_gate = _pytorch_moe_block_ref(
        hidden, gate_weight, w13_weight, w2_weight, residual,
    )

    pk = _make_pk()

    hidden_dt = pk.attach_input(hidden, name="hidden")
    residual_dt = pk.attach_input(residual, name="residual")
    gate_w_dt = pk.attach_input(gate_weight, name="gate_weight")
    w13_w_dt = pk.attach_input(w13_weight, name="w13_weight")
    w2_w_dt = pk.attach_input(w2_weight, name="w2_weight")
    gate_out_dt = pk.attach_input(gate_out, name="gate_out")
    topk_w_dt = pk.attach_input(topk_w, name="topk_weight")
    routing_dt = pk.attach_input(routing, name="routing_indices")
    mask_dt = pk.attach_input(mask, name="moe_mask")
    mlp_mid_dt = pk.attach_input(mlp_mid, name="mlp_mid")
    silu_out_dt = pk.attach_input(silu_out, name="silu_out")
    mlp_out_dt = pk.attach_input(mlp_out, name="mlp_out")
    final_out_dt = pk.attach_input(final_out, name="final_out")

    # Gate linear (demo_30B_A3B.py: grid_dim=(1, 1, 1))
    pk.linear_layer(
        input=hidden_dt,
        weight=gate_w_dt,
        output=gate_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(256, 1, 1),
    )

    # Router
    pk.moe_topk_softmax_routing_layer(
        input=gate_out_dt,
        output=(topk_w_dt, routing_dt, mask_dt),
        grid_dim=(1, 1, 1),
        block_dim=(256, 1, 1),
    )

    # W13 fused gate+up linear
    pk.moe_w13_linear_layer(
        input=hidden_dt,
        weight=w13_w_dt,
        moe_routing_indices=routing_dt,
        moe_mask=mask_dt,
        output=mlp_mid_dt,
        grid_dim=(10, FUSED_OUTDIM // WORLD_SIZE // 128, 1),
        block_dim=(256, 1, 1),
    )

    pk.moe_silu_mul_layer(
        input=mlp_mid_dt,
        output=silu_out_dt,
        grid_dim=(BATCH_SIZE, NUM_EXPERTS_PER_TOK, 1),
        block_dim=(256, 1, 1),
    )

    pk.moe_w2_linear_layer(
        input=silu_out_dt,
        weight=w2_w_dt,
        moe_routing_indices=routing_dt,
        moe_mask=mask_dt,
        output=mlp_out_dt,
        grid_dim=(8, 16, 1),
        block_dim=(256, 1, 1),
    )

    pk.moe_mul_sum_add_layer(
        input=mlp_out_dt,
        weight=topk_w_dt,
        residual=residual_dt,
        output=final_out_dt,
        grid_dim=(BATCH_SIZE, HIDDEN_SIZE // 256, 1),
        block_dim=(256, 1, 1),
    )

    outdir = os.path.join(os.path.dirname(__file__), "_moe_pipeline_b8")
    os.makedirs(outdir, exist_ok=True)
    print("Compiling...")
    pk.compile(output_dir=outdir)
    print("Running...")
    pk()
    torch.cuda.synchronize()

    # --- Compare router outputs ---
    gate_err = (gate_out.float() - ref_gate).abs().max().item()
    topk_err = (topk_w.float() - ref_topk_w).abs().max().item()
    route_counts = [(routing[:, t] > 0).sum().item() for t in range(BATCH_SIZE)]
    counts_ok = all(c == NUM_EXPERTS_PER_TOK for c in route_counts)

    # Routing slot order can differ from PyTorch topk when scores tie; verify the
    # routed experts match the reference set per token.
    routing_ok = True
    for t in range(BATCH_SIZE):
        ref_experts = set(ref_routing[:, t].nonzero().flatten().tolist())
        got_experts = set(routing[:, t].nonzero().flatten().tolist())
        if ref_experts != got_experts:
            routing_ok = False
            break

    print(f"\nGate linear max err (all logits): {gate_err:.6f}")
    print(f"Router topk weight err:           {topk_err:.6f}")
    print(f"Routed expert sets match:       {routing_ok}")
    print(f"Topk counts per token:          {route_counts}")

    # --- Compare final output ---
    print(f"\nFinal[0, :8]:    {final_out[0, :8]}")
    print(f"Reference[0, :8]: {ref_final[0, :8]}")

    max_abs = (final_out.float() - ref_final.float()).abs().max().item()
    denom = ref_final.float().abs().max().item()
    max_rel = max_abs / max(denom, 1e-6)

    print(f"\nFinal max absolute diff: {max_abs:.6f}")
    print(f"Final max relative err:  {max_rel:.6f}")

    passed = (
        topk_err < 1e-3
        and routing_ok
        and counts_ok
        and max_rel < 0.05
    )

    if passed:
        print("\nPASSED: Full Qwen3 MoE block at batch=8 matches PyTorch reference")
    else:
        print("\nFAILED: MoE pipeline batch=8 test")
        if topk_err >= 1e-3:
            print(f"  router weight error {topk_err:.6f} >= 1e-3")
        if not routing_ok:
            print("  routed expert sets mismatch")
        if not counts_ok:
            print(f"  topk counts not all {NUM_EXPERTS_PER_TOK}")
        if max_rel >= 0.05:
            print(f"  final relative error {max_rel:.4f} >= 5%")

    pk.finalize()
    if not passed:
        sys.exit(1)
    return passed


if __name__ == "__main__":
    test_moe_pipeline_batch()
    print("Test completed successfully!")
