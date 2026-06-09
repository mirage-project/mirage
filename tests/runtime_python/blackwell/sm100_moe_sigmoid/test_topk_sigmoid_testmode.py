"""
Test: moe_topk_sigmoid_routing_layer (topk_sigmoid_sm100) via PersistentKernel test_mode.

Refactored to real DeepSeek-V3 routing params and swept over bs ∈ {1,2,4,8,16}.
This is a per-token op (no TP head-sharding), so the primary sweep is bs-only at
expert-parallel ep=1 (local_expert_start=0, num_local_experts=256 = full routing).
A secondary ep>1 variant is also exercised: num_local_experts = 256/ep with a
local_expert_start offset, verifying the reference slices to local experts and
1-indexes within the local range exactly as the kernel does.

Each (bs, ep) config builds a minimal PersistentKernel in test_mode, compiles it,
runs once, and compares:
  - topk_weights      (bs, 8)           float32  — rel/atol bf16 ≈ 1e-2
  - routing_indices   (EL, bs)          int32    — exact int match
  - active_expert_ids (EL+1,)           int32    — exact set + count match

Run:
    python tests/runtime_python/blackwell/sm100_moe_sigmoid/test_topk_sigmoid_testmode.py
"""

import torch
import sys
import os

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

# Use the shared per-folder PyTorch reference (lifted from the kernel test).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import moe_topk_sigmoid_routing_ref


# ============================================================================
# DeepSeek V3 routing configuration (real DSV3 params)
# ============================================================================
NUM_EXPERTS = 256          # n_routed_experts
NUM_EXPERTS_PER_TOK = 8    # num_experts_per_tok
NUM_GROUPS = 8             # n_group
TOPK_GROUP = 4             # topk_group
ROUTED_SCALING_FACTOR = 2.5  # routed_scaling_factor

WEIGHT_TOL = 1e-2  # bf16 rel/atol


def _run_case(bs, ep, seed=42):
    """Run one (bs, ep) config through test_mode; return per-config diff dict.

    ep=1  → full routing (local_expert_start=0, num_local_experts=256).
    ep>1  → num_local_experts = 256/ep; this ep rank owns the SECOND local
            slice (local_expert_start = num_local_experts) to actually exercise
            a non-zero offset and the local-index shift.
    """
    device = "cuda"
    assert NUM_EXPERTS % ep == 0
    num_local_experts = NUM_EXPERTS // ep
    # Pick a non-trivial rank for ep>1 (rank 1) so local_expert_start != 0.
    ep_rank = 0 if ep == 1 else 1
    local_expert_start = ep_rank * num_local_experts

    print(f"\n{'='*64}")
    print(f"[bs={bs}, ep={ep}] DSV3 topk_sigmoid routing")
    print(f"  num_experts={NUM_EXPERTS}  num_experts_per_tok={NUM_EXPERTS_PER_TOK}")
    print(f"  num_groups={NUM_GROUPS}  topk_group={TOPK_GROUP}  scale={ROUTED_SCALING_FACTOR}")
    print(f"  num_local_experts={num_local_experts}  local_expert_start={local_expert_start}")
    print(f"{'='*64}")

    g = torch.Generator(device=device).manual_seed(seed + bs * 100 + ep)

    # Inputs: router_logits (bs, 256) bf16, bias (256,) f32.
    gating_output = torch.randn(
        (bs, NUM_EXPERTS), device=device, dtype=torch.bfloat16, generator=g
    )
    bias = torch.randn(
        NUM_EXPERTS, device=device, dtype=torch.float32, generator=g
    ) * 0.1

    # The kernel writes zeros back into its input (split-k reset) → clone for ref.
    gating_output_ref = gating_output.clone()

    # Outputs.
    topk_weights = torch.zeros(
        bs, NUM_EXPERTS_PER_TOK, device=device, dtype=torch.float32
    )
    routing_indices = torch.zeros(
        (num_local_experts, bs), device=device, dtype=torch.int32
    )
    active_expert_ids = torch.full(
        (num_local_experts + 1,), -1, device=device, dtype=torch.int32
    )

    # Build PersistentKernel in test mode.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = bs
    params["max_num_batched_requests"] = bs
    pk = PersistentKernel(**params)

    gating_dt = pk.attach_input(gating_output, name="gating_output")
    bias_dt = pk.attach_input(bias, name="bias")
    topk_weights_dt = pk.attach_input(topk_weights, name="topk_weights")
    routing_indices_dt = pk.attach_input(routing_indices, name="routing_indices")
    active_ids_dt = pk.attach_input(active_expert_ids, name="active_expert_ids")

    # Single-CTA path (FUSE_COMPACTION=true): grid=(1,1,1), 256 threads (8 warps).
    pk.moe_topk_sigmoid_routing_layer(
        input=gating_dt,
        bias=bias_dt,
        output=(topk_weights_dt, routing_indices_dt, active_ids_dt),
        grid_dim=(1, 1, 1),
        block_dim=(256, 1, 1),
        num_groups=NUM_GROUPS,
        topk_group=TOPK_GROUP,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        local_expert_start=local_expert_start,
    )

    # Unique per-config compile dir under /tmp (avoid shared-FS writes / collisions).
    compile_dir = f"/tmp/mpk_topk_sigmoid_bs{bs}_ep{ep}"
    os.makedirs(compile_dir, exist_ok=True)
    print("Compiling...")
    pk.compile(output_dir=compile_dir)
    print("Running...")
    pk()
    torch.cuda.synchronize()

    # Reference.
    ref_weights, ref_routing, ref_expert_mask = moe_topk_sigmoid_routing_ref(
        gating_output_ref,
        bias,
        bs,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        num_groups=NUM_GROUPS,
        topk_group=TOPK_GROUP,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        local_expert_start=local_expert_start,
        num_local_experts=num_local_experts,
    )

    # --- topk_weights: bf16 rel/atol ---
    weight_diff = (topk_weights - ref_weights).abs().max().item()
    weights_ok = torch.allclose(
        topk_weights, ref_weights, rtol=WEIGHT_TOL, atol=WEIGHT_TOL
    )

    # --- routing_indices: exact int match ---
    routing_match = torch.equal(routing_indices, ref_routing)
    n_routing_diff = (routing_indices != ref_routing).sum().item()

    # --- active_expert_ids (mask): exact set + count match ---
    num_active = int(active_expert_ids[-1].item())
    if num_active > 0:
        recon_mask = torch.zeros(
            (num_local_experts,), device=device, dtype=torch.int32
        )
        ids = active_expert_ids[:num_active].to(torch.long)
        recon_mask.index_fill_(0, ids, 1)
        mask_match = torch.equal(recon_mask, ref_expert_mask)
    else:
        mask_match = ref_expert_mask.sum().item() == 0
    ref_active = int(ref_expert_mask.sum().item())

    passed = bool(weights_ok and routing_match and mask_match)

    print(f"  topk_weights max_diff = {weight_diff:.6f}  "
          f"({'PASS' if weights_ok else 'FAIL'}, tol={WEIGHT_TOL})")
    print(f"  routing_indices       = {'PASS' if routing_match else 'FAIL'}  "
          f"({n_routing_diff} mismatched)")
    print(f"  active_expert_ids     = {'PASS' if mask_match else 'FAIL'}  "
          f"(kernel={num_active} active, ref={ref_active} active)")
    print(f"  [bs={bs}, ep={ep}] {'PASS' if passed else 'FAIL'}")

    pk.finalize()

    return {
        "bs": bs,
        "ep": ep,
        "num_local_experts": num_local_experts,
        "local_expert_start": local_expert_start,
        "weight_diff": weight_diff,
        "weights_ok": weights_ok,
        "routing_match": routing_match,
        "n_routing_diff": n_routing_diff,
        "mask_match": mask_match,
        "num_active": num_active,
        "ref_active": ref_active,
        "passed": passed,
    }


# Matrix: bs sweep at ep=1 (full routing) + a secondary ep>1 (=2) variant
# across the same bs sweep to exercise the local-expert offset / index shift.
MATRIX = [(bs, 1) for bs in (1, 2, 4, 8, 16)] + \
         [(bs, 2) for bs in (1, 2, 4, 8, 16)]


def test_topk_sigmoid_testmode():
    results = []
    for bs, ep in MATRIX:
        results.append(_run_case(bs, ep))

    print(f"\n{'='*64}")
    print("MATRIX SUMMARY (moe_topk_sigmoid_routing)")
    print(f"{'='*64}")
    n_pass = 0
    for r in results:
        tag = "PASS" if r["passed"] else "FAIL"
        if r["passed"]:
            n_pass += 1
        print(f"  bs={r['bs']:2d} ep={r['ep']} EL={r['num_local_experts']:3d} "
              f"start={r['local_expert_start']:3d} | "
              f"w_diff={r['weight_diff']:.5f} routing_diff={r['n_routing_diff']} "
              f"active(k/ref)={r['num_active']}/{r['ref_active']} -> {tag}")
    print(f"\n{n_pass}/{len(results)} PASS")

    failed = [r for r in results if not r["passed"]]
    assert not failed, (
        f"{len(failed)} config(s) FAILED: "
        + ", ".join(f"(bs={r['bs']},ep={r['ep']})" for r in failed)
    )
    print("\nALL PASS")


if __name__ == "__main__":
    test_topk_sigmoid_testmode()
