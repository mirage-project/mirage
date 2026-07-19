"""Parametric test_mode test for ``splitk_linear_layer`` (BF16, sm100).

Runs ONE configuration (batch, N, K, grid_x, grid_y, accumulate) per
invocation, prints PASS / FAIL on the last line, and exits non-zero on
failure.

Usage:
  CUDA_VISIBLE_DEVICES=<gpu> python test_splitk_linear_bf16_testmode.py \\
      --batch 1 --N 256 --K 7168 --grid-x 2 --grid-y 2 --accumulate=False
"""
import argparse
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import splitk_linear_ref


# Per-config artifacts go to /tmp so the source tree stays clean and so
# concurrent subprocess runs (one per config) don't fight over filenames.
def _artifact_dir(label: str) -> str:
    base = os.environ.get("MPK_TEST_OUTPUT_DIR",
                          "/tmp/mpk_test_splitk_linear_bf16")
    p = os.path.join(base, label)
    os.makedirs(p, exist_ok=True)
    return p


def _make_pk(batch: int, requests: int = None) -> PersistentKernel:
    nw, ns = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=nw,
        num_local_schedulers=ns,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=batch,
        max_num_batched_requests=requests if requests is not None else batch,
    )
    return PersistentKernel(**params)


def run_case(batch: int, N: int, K: int, grid_x: int, grid_y: int,
             accumulate: bool, label: str) -> bool:
    device = "cuda"
    torch.manual_seed(42)

    if N % grid_x != 0 or (N // grid_x) % 128 != 0:
        print(f"SKIP per-task N={N // grid_x} not multiple of 128")
        return True
    if K % grid_y != 0:
        print(f"SKIP K={K} not divisible by grid_y={grid_y}")
        return True

    inp = (torch.randn(batch, K, dtype=torch.bfloat16, device=device) * 0.05
           ).contiguous()
    weight = (torch.randn(N, K, dtype=torch.bfloat16, device=device)
              / (K ** 0.5)).contiguous()

    if accumulate:
        # Kernel reduce-adds onto pre_output. Use a small non-zero seed so
        # the test detects accidental zeroing.
        pre_output = (torch.randn(batch, N, dtype=torch.bfloat16,
                                  device=device) * 0.1).contiguous()
    else:
        # Kernel will land a pure result; pre-fill with garbage to ensure
        # the prepended tensor_init really zeroes the buffer.
        pre_output = torch.randn(batch, N, dtype=torch.bfloat16,
                                 device=device).contiguous()
    output = pre_output.clone()
    pre_output_snapshot = pre_output.clone()

    ref = splitk_linear_ref(inp, weight,
                            pre_output=pre_output_snapshot,
                            accumulate=accumulate)

    pk = _make_pk(batch, requests=int(os.environ.get("MPK_TEST_REQUESTS", "0")) or batch)
    in_dt = pk.attach_input(inp, name="input")
    w_dt = pk.attach_input(weight, name="weight")
    out_dt = pk.attach_input(output, name="output")
    pk.splitk_linear_layer(
        input=in_dt,
        weight=w_dt,
        output=out_dt,
        grid_dim=(grid_x, grid_y, 1),
        block_dim=(256, 1, 1),
        accumulate=accumulate,
    )

    print(f"Compiling {label} ...", flush=True)
    pk.compile(output_dir=_artifact_dir(label))
    print(f"Running {label} ...", flush=True)
    pk()
    torch.cuda.synchronize()

    err = (output.float() - ref.float()).abs().max().item()
    print(f"  output[0,:8] = {output[0, :8].tolist()}")
    print(f"  ref[0,:8]    = {ref[0, :8].tolist()}")
    print(f"  pre[0,:8]    = {pre_output_snapshot[0, :8].tolist()}")
    print(f"  max-abs-error = {err:.4f}")
    pk.finalize()

    # bf16 matmul + accumulate has substantial rounding; pick a generous tol.
    tol = 0.5 + 0.05 * (K ** 0.5)
    ok = err < tol
    print(f"{'PASS' if ok else 'FAIL'} {label}  err={err:.4f}  tol={tol:.4f}")
    return ok


def parse_bool(s: str) -> bool:
    if s.lower() in ("true", "1", "yes"):
        return True
    if s.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--grid-x", type=int, required=True)
    ap.add_argument("--grid-y", type=int, required=True)
    ap.add_argument("--accumulate", type=parse_bool, required=True)
    ap.add_argument("--label", type=str, default="case")
    args = ap.parse_args()

    ok = run_case(args.batch, args.N, args.K, args.grid_x, args.grid_y,
                  args.accumulate, args.label)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
