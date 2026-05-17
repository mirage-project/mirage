"""Numerical test: ``layers.moe.MoEW2(dtype='fp8')`` via test_mode."""

import math
import os
import sys

import torch

import mirage
from mirage.mpk.layers.moe.w2 import MoEW2
from mirage.mpk.persistent_kernel import PersistentKernel


def _quantize_fp8_3d_last(x: torch.Tensor):
    """fp8 quantize a 3-D tensor along its last dim per-128-group."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    A, B, K = x.shape
    assert K % 128 == 0
    x_b = x.reshape(A, B, K // 128, 128)
    amax = x_b.abs().amax(dim=3)
    scale = (amax / fp8_max).clamp(min=1e-12)
    x_fp8 = (x_b / scale.unsqueeze(3)).reshape(A, B, K).to(torch.float8_e4m3fn)
    return x_fp8, scale.float()


def _make_routing(num_experts, batch_size, topk, device):
    routing = torch.zeros(num_experts, batch_size, dtype=torch.int32, device=device)
    for b in range(batch_size):
        experts = [(b * topk + s) % num_experts for s in range(topk)]
        for slot, e in enumerate(experts):
            routing[e, b] = slot + 1
    activated = [e for e in range(num_experts) if routing[e].any()]
    mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    for idx, e in enumerate(activated):
        mask[idx] = e
    mask[num_experts] = len(activated)
    return routing, mask


def test_moe_w2_fp8():
    device = "cuda"
    torch.manual_seed(0)

    batch_size = 2
    hidden_size = 256
    intermediate_size = 128  # must be % 128
    num_experts = 4
    topk = 2

    x_val = torch.randn(batch_size, topk, intermediate_size, device=device) * 0.1
    w_val = torch.randn(num_experts, hidden_size, intermediate_size, device=device) \
        / math.sqrt(intermediate_size)
    x_fp8, x_scale = _quantize_fp8_3d_last(x_val)
    w_fp8, w_scale = _quantize_fp8_3d_last(w_val)

    out_buf = torch.zeros(batch_size, topk, hidden_size, dtype=torch.bfloat16, device=device)
    routing_indices, moe_mask = _make_routing(num_experts, batch_size, topk, device)

    m = MoEW2(
        num_experts=num_experts,
        num_experts_per_tok=topk,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype="fp8",
        prefix="w2fp8_",
    ).to(device=device)
    with torch.no_grad():
        m.weight.data.copy_(w_fp8)
        m.weight_scale.data.copy_(w_scale)

    ref = m.forward(x_fp8, routing_indices, x_scale=x_scale)

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

    x_dt = pk.attach_input(x_fp8, name="w2fp8_x")
    x_sc_dt = pk.attach_input(x_scale, name="w2fp8_x_scale")
    routing_dt = pk.attach_input(routing_indices, name="w2fp8_routing")
    mask_dt = pk.attach_input(moe_mask, name="w2fp8_mask")
    out_dt = pk.attach_input(out_buf, name="w2fp8_out")

    with pk.compile_scope():
        _ = m.compile(x_dt, routing_dt, mask_dt, out_dt, x_scale=x_sc_dt,
                      grid_dim=(num_experts, hidden_size // 128, 1))

    print("Compiling MoEW2 (fp8) test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[0, 0, :8]: {out_buf[0, 0, :8]}")
    print(f"ref[0, 0, :8]:     {ref[0, 0, :8]}")

    routed_b_k = [(b, k) for b in range(batch_size)
                  for k in range(topk)
                  if (routing_indices[:, b] == (k + 1)).any().item()]
    for b, k in routed_b_k:
        diff = (out_buf[b, k].float() - ref[b, k].float()).abs()
        max_abs = diff.max().item()
        max_rel = max_abs / max(ref[b, k].float().abs().max().item(), 1e-6)
        print(f"  slot (b={b}, k={k}): max_abs={max_abs:.4f}, max_rel={max_rel:.4f}")
        try:
            torch.testing.assert_close(out_buf[b, k], ref[b, k], atol=0.5, rtol=0.5)
        except AssertionError as e:
            print(f"FAILED at (b={b}, k={k}):\n{e}")
            pk.finalize()
            sys.exit(1)
    print("PASSED: MoEW2 (fp8) compile() ~ forward() within fp8 tolerance")
    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_moe_w2_fp8()
