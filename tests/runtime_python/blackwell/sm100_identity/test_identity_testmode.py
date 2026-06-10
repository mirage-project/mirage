"""
Test ``PersistentKernel.identity_layer`` end-to-end through the full MPK
compilation pipeline (test_mode).

The identity layer is a BF16 copy: output == input.  It is used in the
DeepSeek V3 builder as a phantom bridge to preserve task-graph dependency
edges (e.g. before an FP8 GEMM or across a chunked-prefill barrier).

Layer constraint (from persistent_kernel.py):
    input.num_dims == output.num_dims
    input.dim(i) == output.dim(i) for all i
    last_dim index must be 1 (2-D tensor) or 2 (3-D tensor)

We use the DSv3 hidden size 7168 and a 2-D [bs, hidden] layout, with the
identity split across grid.x = 56 CTAs (7168 = 56 * 128, same as the
production builder).

Sweep: bs ∈ {1, 2, 4, 8, 16}.

Run:
    CUDA_VISIBLE_DEVICES=<gpu> python \\
        tests/runtime_python/blackwell/sm100_identity/test_identity_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

HIDDEN = 7168
GRID_X = 56           # 7168 = 56 * 128; identity_layer splits last dim over grid.x
BS_LIST = [1, 2, 4, 8, 16]


def _run_case(bs: int):
    device = "cuda"
    torch.manual_seed(42 + bs)

    x = torch.randn(bs, HIDDEN, dtype=torch.bfloat16, device=device)
    y = torch.zeros(bs, HIDDEN, dtype=torch.bfloat16, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = bs
    params["max_num_batched_requests"] = bs
    params["meta_tensors"] = {
        "prompt_lengths": torch.ones(bs, dtype=torch.int32, device=device),
    }
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x, name="input")
    y_dt = pk.attach_input(y, name="output")

    pk.identity_layer(
        input=x_dt,
        output=y_dt,
        grid_dim=(GRID_X, 1, 1),
        block_dim=(128, 1, 1),
    )

    folder_path = os.path.dirname(os.path.abspath(__file__))
    print(f"\n{'='*60}")
    print(f"Test: identity_layer  bs={bs}  hidden={HIDDEN}")
    print("Compiling...")
    pk.compile(output_dir=folder_path)
    print("Running...")
    pk()
    torch.cuda.synchronize()

    # Reference: output should be bit-exact copy of input.
    max_diff = (y.float() - x.float()).abs().max().item()
    print(f"  max abs diff: {max_diff:.6f}", end="")

    assert max_diff == 0.0, (
        f"bs={bs}: identity output mismatch (max_diff={max_diff})")
    print("  PASS")

    pk.finalize()
    return max_diff


def test_identity_testmode():
    for bs in BS_LIST:
        _run_case(bs)


if __name__ == "__main__":
    results = []
    for bs in BS_LIST:
        diff = _run_case(bs)
        results.append((bs, diff))

    print(f"\n{'='*60}")
    print(f"IDENTITY SUMMARY  hidden={HIDDEN}")
    for bs, diff in results:
        print(f"  bs={bs:2d}: max_diff={diff:.6f}  PASS")
    print(f"ALL PASS ({len(results)}/{len(results)})")
    sys.exit(0)
