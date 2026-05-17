"""Numerical test: ``layers.moe.MoESiluMul`` via PersistentKernel test_mode.

Mirrors ``test_silu_mul.py`` but for the MoE 3-D layout.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.moe.silu_mul import MoESiluMul
from mirage.mpk.persistent_kernel import PersistentKernel


def test_moe_silu_mul_3d():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    batch_size = 2
    num_experts_per_tok = 2
    intermediate_size = 256
    fused = 2 * intermediate_size

    gateup = torch.randn(batch_size, num_experts_per_tok, fused, dtype=dtype, device=device)
    out_buf = torch.zeros(batch_size, num_experts_per_tok, intermediate_size,
                          dtype=dtype, device=device)

    m = MoESiluMul(intermediate_size=intermediate_size).to(device=device, dtype=dtype)
    ref = m.forward(gateup)

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

    gateup_dt = pk.attach_input(gateup, name="moe_silu_gateup")
    out_dt = pk.attach_input(out_buf, name="moe_silu_out")

    with pk.compile_scope():
        _ = m.compile(gateup_dt, out_dt)

    print("Compiling MoESiluMul test kernel...")
    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[0, 0, :8]: {out_buf[0, 0, :8]}")
    print(f"ref[0, 0, :8]:     {ref[0, 0, :8]}")
    max_diff = (out_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute diff: {max_diff}")
    try:
        torch.testing.assert_close(out_buf, ref, atol=1e-2, rtol=1e-2)
        print("PASSED: MoESiluMul (3-D) compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: MoESiluMul (3-D) disagreement\n{e}")
        pk.finalize()
        sys.exit(1)
    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_moe_silu_mul_3d()
