"""Numerical test: ``layers.moe.MoEW13`` (bf16) via PersistentKernel test_mode.

The kernel expects ``moe_mask`` in "active expert list" format —
``moe_mask[0..n-1]`` are activated expert ids, ``moe_mask[NUM_EXPERTS]``
is the count. We build that explicitly; ``forward()`` does NOT consume
``moe_mask`` (it derives the expert mapping from ``routing_indices``).
"""

import math
import os
import sys

import torch

import mirage
from mirage.mpk.layers.moe.w13 import MoEW13
from mirage.mpk.persistent_kernel import PersistentKernel


def _make_routing(num_experts: int, batch_size: int, topk: int, device, seed: int = 0):
    """Build (routing_indices (E,B), moe_mask (E+1,)) for the kernel."""
    g = torch.Generator(device=device).manual_seed(seed)
    routing_indices = torch.zeros(num_experts, batch_size, dtype=torch.int32, device=device)
    for b in range(batch_size):
        # Pick `topk` distinct experts per token, assign 1..topk in routing_indices.
        perm = torch.randperm(num_experts, generator=g, device=device)[:topk]
        for slot, e in enumerate(perm.tolist()):
            routing_indices[e, b] = slot + 1
    # Build active-expert list for the kernel mask.
    activated = []
    for e in range(num_experts):
        if (routing_indices[e] != 0).any():
            activated.append(e)
    moe_mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    for idx, e in enumerate(activated):
        moe_mask[idx] = e
    moe_mask[num_experts] = len(activated)
    return routing_indices, moe_mask


def test_moe_w13_bf16():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    batch_size = 1
    hidden_size = 256
    intermediate_size = 128
    num_experts = 4
    topk = 2

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device) * 0.1
    w = torch.randn(num_experts, 2 * intermediate_size, hidden_size,
                    dtype=dtype, device=device) / math.sqrt(hidden_size)
    out_buf = torch.zeros(batch_size, topk, 2 * intermediate_size,
                          dtype=dtype, device=device)

    routing_indices, moe_mask = _make_routing(num_experts, batch_size, topk, device)

    m = MoEW13(
        num_experts=num_experts,
        num_experts_per_tok=topk,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype="bf16",
        prefix="w13_",
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        m.weight.copy_(w)

    ref = m.forward(x, routing_indices)

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

    x_dt = pk.attach_input(x, name="w13_x")
    routing_dt = pk.attach_input(routing_indices, name="w13_routing_indices")
    mask_dt = pk.attach_input(moe_mask, name="w13_mask")
    out_dt = pk.attach_input(out_buf, name="w13_out")

    with pk.compile_scope():
        _ = m.compile(x_dt, routing_dt, mask_dt, out_dt)

    print("Compiling MoEW13 (bf16) test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # Inspect a routed slot.
    print(f"out_buf[0, 0, :8]: {out_buf[0, 0, :8]}")
    print(f"ref[0, 0, :8]:     {ref[0, 0, :8]}")

    # Only compare slots that were routed (others may stay zero in either path).
    routed_b_k = [(b, k) for b in range(batch_size)
                  for k in range(topk)
                  if (routing_indices[:, b] == (k + 1)).any().item()]
    for b, k in routed_b_k:
        try:
            torch.testing.assert_close(out_buf[b, k], ref[b, k], atol=2e-2, rtol=2e-2)
        except AssertionError as e:
            print(f"FAILED at (b={b}, k={k}):\n{e}")
            pk.finalize()
            sys.exit(1)
    print("PASSED: MoEW13 (bf16) compile() matches forward() on all routed slots")
    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_moe_w13_bf16()
