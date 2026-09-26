"""Standalone task-chain latency, including reduce, excluding allocations.

No speedup claim: reports measurements for the selected-index implementation.
"""

import argparse
import torch
import runtime_kernel_sparse_mla
from reference import make_case, torch_reference
from test_sparse_mla import allocate_workspace, assert_close, to_cuda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rope-dim", type=int, choices=(0, 64), default=64)
    parser.add_argument("--heads", type=int, choices=(8, 16, 32, 64), default=64)
    parser.add_argument("--kv-len", type=int, default=16384)
    parser.add_argument("--queries", type=int, default=1)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not 0 < args.queries <= args.kv_len or args.topk <= 0 or args.iterations <= 0:
        parser.error("require 0 < queries <= kv-len, topk > 0 and iterations > 0")
    if torch.cuda.get_device_capability() != (10, 0):
        parser.error("requires an SM100 GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    case = to_cuda(make_case(args.rope_dim, args.heads, (args.queries,),
                             (args.kv_len,), capacity=args.topk))
    expected = torch_reference(*case, 0.0625)
    print("GPU:", torch.cuda.get_device_name())
    print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("Measured: sparse MLA plus optional reduce; no indexer or KV writes.")
    for splits in (1, 2, 4, 8):
        workspace = allocate_workspace(case[0], splits)
        for _ in range(10):
            runtime_kernel_sparse_mla.run(*case, *workspace, 0.0625, splits)
        torch.cuda.synchronize()
        assert_close(workspace[0], expected)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            runtime_kernel_sparse_mla.run(*case, *workspace, 0.0625, splits)
        end.record()
        end.synchronize()
        print(f"splits={splits}: {start.elapsed_time(end) * 1000 / args.iterations:.2f} us")


if __name__ == "__main__":
    main()
