import os
import sys

import torch
import runtime_kernel_blackwell_sinkhorn as runtime_kernel_blackwell

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(THIS_DIR, "profile")
if PROFILE_DIR not in sys.path:
    sys.path.insert(0, PROFILE_DIR)

from utils import sinkhorn_knopp_torch


def time_ms(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_case(num_tokens, hidden_size, repeat=20, token_block_size=4):
    comb_res_mix = torch.randn(
        (num_tokens, hidden_size, hidden_size),
        device="cuda",
        dtype=torch.float32,
    )
    comb_res_mix_out = torch.empty_like(comb_res_mix)

    kernel_ms = time_ms(
        lambda: runtime_kernel_blackwell.sinkhorn_sm100(
            comb_res_mix,
            comb_res_mix_out,
            repeat=repeat,
            eps=1e-9,
            token_block_size=token_block_size,
        )
    )
    torch_ms = time_ms(
        lambda: sinkhorn_knopp_torch(comb_res_mix, repeat=repeat, eps=1e-9),
        warmup=5,
        iters=20,
    )
    matrices_per_ms = num_tokens / kernel_ms
    print(
        f"tokens={num_tokens:5d} hidden={hidden_size:2d} repeat={repeat:2d} "
        f"kernel={kernel_ms:.4f} ms torch={torch_ms:.4f} ms "
        f"matrices/ms={matrices_per_ms:.1f}"
    )


if __name__ == "__main__":
    torch.cuda.init()
    for shape in [(1024, 4), (4096, 4), (4096, 8), (4096, 16)]:
        bench_case(*shape)
