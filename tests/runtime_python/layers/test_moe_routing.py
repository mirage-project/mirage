"""Numerical test: ``layers.moe.MoETopkRouting`` via PersistentKernel test_mode.

Tests both ``variant="softmax"`` and ``variant="sigmoid"``.

We compare ``moe_topk_weights`` (loose tolerance — bf16 logits) and check
that the routing_indices sets the right number of slots per token. The
``moe_mask`` semantics differ between the catalog forward() (prefix
counts) and the kernel ("active_expert_ids" list), so we don't compare
that output strictly.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.moe.routing import MoETopkRouting
from mirage.mpk.persistent_kernel import PersistentKernel


def _check_routing_sanity(routing_indices: torch.Tensor, batch_size: int, topk: int):
    """The kernel writes routing_indices as (E, B) with 1..topk per token,
    exactly ``topk`` non-zero entries per column."""
    nz_per_token = (routing_indices != 0).sum(dim=0)
    assert torch.all(nz_per_token == topk), (
        f"each token must be routed to {topk} experts; "
        f"got per-token nz counts {nz_per_token.tolist()}"
    )
    # Slot values 1..topk distinct per token.
    for t in range(batch_size):
        col = routing_indices[:, t]
        slots = col[col != 0].sort().values
        expected = torch.arange(1, topk + 1, dtype=col.dtype, device=col.device)
        assert torch.equal(slots, expected), (
            f"token {t} routing slots are not a permutation of "
            f"1..{topk}: {slots.tolist()}"
        )


def _topk_weights_close(out_w: torch.Tensor, ref_w: torch.Tensor):
    """The kernel picks top-k and the reference picks top-k, both should
    agree on the SET of values (per row) up to bf16 noise. We sort each
    row and compare."""
    out_sorted, _ = torch.sort(out_w, dim=-1)
    ref_sorted, _ = torch.sort(ref_w, dim=-1)
    torch.testing.assert_close(out_sorted, ref_sorted, atol=2e-2, rtol=2e-2)


def test_moe_topk_routing_softmax():
    device = "cuda"
    torch.manual_seed(0)

    # Kernel requires THREADS_PER_ROW (= num_experts/VPT where VPT=8) be 16 or 32:
    # num_experts must be 128 or 256. We use 128 to stay light.
    batch_size = 2
    num_experts = 128
    topk = 4

    print("\n=== softmax routing ===")
    logits_bf16 = torch.randn(batch_size, num_experts, dtype=torch.bfloat16, device=device)
    topk_weights = torch.zeros(batch_size, topk, dtype=torch.float32, device=device)
    routing_indices = torch.zeros(num_experts, batch_size, dtype=torch.int32, device=device)
    moe_mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)

    m = MoETopkRouting(
        num_experts=num_experts,
        num_experts_per_tok=topk,
        variant="softmax",
    ).to(device=device)

    # Reference uses fp32 logits internally; pass bf16 cast to fp32.
    ref_weights, ref_routing, _ = m.forward(logits_bf16.float())

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    logits_dt = pk.attach_input(logits_bf16, name="route_logits")
    topk_w_dt = pk.attach_input(topk_weights, name="route_topk_w")
    routing_dt = pk.attach_input(routing_indices, name="route_indices")
    mask_dt = pk.attach_input(moe_mask, name="route_mask")

    with pk.compile_scope():
        _ = m.compile(logits_dt, topk_w_dt, routing_dt, mask_dt)

    print("Compiling softmax routing test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"topk_weights[0]: {topk_weights[0]}")
    print(f"ref_weights[0]:  {ref_weights[0]}")

    _check_routing_sanity(routing_indices, batch_size, topk)
    _topk_weights_close(topk_weights, ref_weights)
    print("PASSED: MoETopkRouting(softmax) numerical agreement")
    pk.finalize()


def test_moe_topk_routing_sigmoid():
    device = "cuda"
    torch.manual_seed(1)

    # Kernel constraints same as softmax: num_experts in {128, 256}.
    batch_size = 2
    num_experts = 128
    topk = 4
    num_groups = 4
    topk_group = 2

    print("\n=== sigmoid routing ===")
    logits_bf16 = torch.randn(batch_size, num_experts, dtype=torch.bfloat16, device=device)
    bias = torch.randn(num_experts, dtype=torch.float32, device=device) * 0.1
    topk_weights = torch.zeros(batch_size, topk, dtype=torch.float32, device=device)
    routing_indices = torch.zeros(num_experts, batch_size, dtype=torch.int32, device=device)
    moe_mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)

    m = MoETopkRouting(
        num_experts=num_experts,
        num_experts_per_tok=topk,
        variant="sigmoid",
        num_groups=num_groups,
        topk_group=topk_group,
        routed_scaling_factor=2.5,
    ).to(device=device)
    with torch.no_grad():
        m.bias.copy_(bias)

    ref_weights, ref_routing, _ = m.forward(logits_bf16.float())

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    logits_dt = pk.attach_input(logits_bf16, name="sig_logits")
    topk_w_dt = pk.attach_input(topk_weights, name="sig_topk_w")
    routing_dt = pk.attach_input(routing_indices, name="sig_indices")
    mask_dt = pk.attach_input(moe_mask, name="sig_mask")

    with pk.compile_scope():
        _ = m.compile(logits_dt, topk_w_dt, routing_dt, mask_dt)

    print("Compiling sigmoid routing test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"topk_weights[0]: {topk_weights[0]}")
    print(f"ref_weights[0]:  {ref_weights[0]}")

    _check_routing_sanity(routing_indices, batch_size, topk)
    _topk_weights_close(topk_weights, ref_weights)
    print("PASSED: MoETopkRouting(sigmoid) numerical agreement")
    pk.finalize()


if __name__ == "__main__":
    test_moe_topk_routing_softmax()
    test_moe_topk_routing_sigmoid()
    print("All routing tests completed successfully!")
