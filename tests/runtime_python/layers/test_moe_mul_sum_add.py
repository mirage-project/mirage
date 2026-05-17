"""Numerical test: ``layers.moe.MoeMulSumAdd`` via PersistentKernel test_mode."""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.moe.mul_sum_add import MoeMulSumAdd
from mirage.mpk.persistent_kernel import PersistentKernel


def test_moe_mul_sum_add():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    batch_size = 2
    topk = 4
    hidden_size = 256  # divisible by 256/16 etc.

    x = torch.randn(batch_size, topk, hidden_size, dtype=dtype, device=device) * 0.1
    topk_weights = torch.rand(batch_size, topk, dtype=torch.float32, device=device)
    residual = torch.randn(batch_size, hidden_size, dtype=dtype, device=device) * 0.01
    out_buf = torch.zeros(batch_size, hidden_size, dtype=dtype, device=device)

    m = MoeMulSumAdd(
        hidden_size=hidden_size,
        num_experts_per_tok=topk,
    ).to(device=device, dtype=dtype)
    ref = m.forward(x, topk_weights, residual)

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

    x_dt = pk.attach_input(x, name="msa_x")
    w_dt = pk.attach_input(topk_weights, name="msa_topk_w")
    res_dt = pk.attach_input(residual, name="msa_residual")
    out_dt = pk.attach_input(out_buf, name="msa_out")

    with pk.compile_scope():
        _ = m.compile(x_dt, w_dt, res_dt, out_dt)

    print("Compiling MoeMulSumAdd test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[0, :8]: {out_buf[0, :8]}")
    print(f"ref[0, :8]:     {ref[0, :8]}")
    max_diff = (out_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute diff: {max_diff}")
    try:
        torch.testing.assert_close(out_buf, ref, atol=2e-2, rtol=2e-2)
        print("PASSED: MoeMulSumAdd compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: MoeMulSumAdd disagreement\n{e}")
        pk.finalize()
        sys.exit(1)
    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_moe_mul_sum_add()
