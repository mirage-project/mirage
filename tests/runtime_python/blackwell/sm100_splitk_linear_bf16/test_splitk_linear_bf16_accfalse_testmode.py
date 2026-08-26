"""Regression repro: BF16 splitk_linear_layer + accumulate=False deadlocks.

Single-op test_mode invocation of `splitk_linear_layer` with
`accumulate=False` (which prepends a `tensor_init`) reproducibly hangs the
MPK persistent-kernel runtime on a dedicated GPU. Verbose worker traces
(MPK_ENABLE_VERBOSE=1) show the kernel itself hangs after launch:
  - all 4 prepended `tensor_init` (262) tasks reach `[DONE]`
  - all 4 `splitk_linear_sm100` (251) tasks print `EXECUTE_TASK 251`
  - none of the splitk tasks reach `[DONE]`
  - scheduler is idle waiting for END_OF_TASK_GRAPH triggers

The same `splitk_linear_sm100` kernel works fine when invoked with
`accumulate=True` from the qwen3 o_proj path end-to-end. The same
`accumulate=False` flow on `linear_splitk_swapAB_fp8_layer` (FP8) works
(see `tests/runtime_python/blackwell/sm100_linear_splitk_fp8_swapAB/`).
So the bug is specific to **BF16 splitk + prepended-tensor_init**.

Shape mirrors the DeepSeek V3 MoE gate that surfaced this deadlock:
  input=(1, 7168)  weight=(256, 7168)  output=(1, 256)
  grid=(2, 2, 1)   block=(256, 1, 1)   accumulate=False

This test is expected to TIME OUT until the underlying issue is fixed. The
DSv3 builder gates the gate-splitk replacement behind
`_BF16_GATE_SPLITK_ENABLED = False` until then.

Tried but did NOT fix the hang:
  - changing output from `attach_input` -> `new_tensor`
  - registering an unrelated upstream `tensor_init` op before the splitk
  - using a larger qwen3-style shape (4096x4096, grid 32x4) — also hangs
    in standalone (unlike the qwen3 demo, which works end-to-end)

Run:
  CUDA_VISIBLE_DEVICES=<free-gpu> python tests/runtime_python/blackwell/sm100_splitk_linear_bf16/test_splitk_linear_bf16_accfalse_testmode.py
"""
import os

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def main():
    device = "cuda"
    torch.manual_seed(0)

    batch = 1
    K = 7168
    N = 256

    inp = torch.randn(batch, K, dtype=torch.bfloat16, device=device)
    weight = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    output = torch.zeros(batch, N, dtype=torch.bfloat16, device=device)

    nw, ns = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=nw,
        num_local_schedulers=ns,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=batch,
        max_num_batched_requests=batch,
    )
    pk = PersistentKernel(**params)

    in_dt = pk.attach_input(inp, name="input")
    w_dt = pk.attach_input(weight, name="weight")
    out_dt = pk.attach_input(output, name="output")

    pk.splitk_linear_layer(
        input=in_dt,
        weight=w_dt,
        output=out_dt,
        grid_dim=(N // 128, 2, 1),
        block_dim=(256, 1, 1),
        accumulate=False,
    )

    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    pk()
    torch.cuda.synchronize()

    ref = inp.float() @ weight.float().t()
    err = (output.float() - ref).abs().max().item()
    print(f"max-abs-error = {err:.4f}  (tol ~ 2.0)")
    print("RAN_TO_COMPLETION")
    pk.finalize()


if __name__ == "__main__":
    main()
