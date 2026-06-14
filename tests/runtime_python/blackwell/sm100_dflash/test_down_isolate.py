"""Isolate the down linear: random input [S,I] @ down_w[H,I].T -> [S,H].

Scans a few (S, grid) configs to localize the MLP down failure.
Run: CUDA_VISIBLE_DEVICES=2 python test_down_isolate.py
"""
import os
import torch
import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def run(S, I, H, grid, use_resid):
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    x = torch.randn(S, I, dtype=dtype, device=device) * 0.05
    w = torch.randn(H, I, dtype=dtype, device=device) * 0.05
    ref = (x.float() @ w.float().T)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = S
    params["max_num_batched_requests"] = 1
    params["use_cutlass_kernel"] = True
    pk = PersistentKernel(**params)

    out = torch.zeros(S, H, dtype=dtype, device=device)
    x_dt = pk.attach_input(x, name="x")
    w_dt = pk.attach_input(w, name="w")
    o_dt = pk.attach_input(out, name="o")
    bd = (256, 1, 1)
    if use_resid:
        z = torch.zeros(S, H, dtype=dtype, device=device)
        z_dt = pk.attach_input(z, name="z")
        pk.linear_with_residual_layer(input=x_dt, weight=w_dt, residual=z_dt,
                                      output=o_dt, grid_dim=(grid, 1, 1), block_dim=bd)
    else:
        pk.linear_layer(input=x_dt, weight=w_dt, output=o_dt,
                        grid_dim=(grid, 1, 1), block_dim=bd)
    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()
    err = (out.float() - ref).abs().max().item()
    pk.finalize()
    print(f"S={S} I={I} H={H} grid={grid} resid={use_resid} -> maxerr {err:.4f} "
          f"(refmax {ref.abs().max().item():.2f})")
    return err


if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "a"
    # one config per process (test-mode is one-shot per compile)
    if cfg == "a":
        run(8, 18432, 7168, 64, False)
    elif cfg == "b":
        run(16, 18432, 7168, 64, False)
    elif cfg == "c":
        run(8, 18432, 7168, 112, False)
    elif cfg == "d":
        run(8, 7168, 7168, 64, False)
