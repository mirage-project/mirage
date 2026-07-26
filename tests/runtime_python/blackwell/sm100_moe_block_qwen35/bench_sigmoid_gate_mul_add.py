"""Latency benchmark for `sigmoid_gate_mul_add_sm100` (task 238).

One thread block, exactly as a megakernel worker runs it, across the decode and
prefill-chunk batch sizes the Qwen3.5 build uses (mbt = 16,
docs/qwen35/v1-architecture.md 9.1) plus a small shape for reference.

The number that matters is not the absolute latency but whether folding the gate
GEMV, the sigmoid, the broadcast multiply and the residual add into ONE task is
cheaper than the alternatives it replaced: a degenerate N=1 `linear_layer` plus a
separate elementwise pass over `hidden`. Both are reported.

Run:  python tests/runtime_python/blackwell/sm100_moe_block_qwen35/bench_sigmoid_gate_mul_add.py
"""

import torch

import runtime_kernel_blackwell_moe_block_qwen35 as blk

DEVICE = "cuda"
REPS = 200
WARMUP = 20


def time_call(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / REPS


def main():
    torch.manual_seed(20260726)
    print(f"{'batch':>6} {'hidden':>7} {'task_us':>9} {'torch_us':>9} {'bytes':>10} {'GB/s':>8}")
    for batch, hidden in ((1, 2048), (2, 2048), (8, 2048), (16, 2048), (4, 256)):
        x = torch.randn(batch, hidden, dtype=torch.bfloat16, device=DEVICE)
        w = torch.randn(1, hidden, dtype=torch.bfloat16, device=DEVICE)
        shared = torch.randn(batch, hidden, dtype=torch.bfloat16, device=DEVICE)
        resid = torch.randn(batch, hidden, dtype=torch.bfloat16, device=DEVICE)
        out = torch.zeros_like(resid)

        t_task = time_call(
            lambda: blk.sigmoid_gate_mul_add_sm100(x, w, shared, resid, out))
        # what it replaces: an N=1 GEMV, a sigmoid, a broadcast multiply and an add
        t_torch = time_call(
            lambda: torch.add(
                resid,
                torch.sigmoid((x.float() @ w.float().t()).to(torch.bfloat16).float())
                .to(torch.bfloat16) * shared))
        # 4 bf16 tensors of batch*hidden (x, shared, residual in; out) + w
        nbytes = 2 * (4 * batch * hidden + hidden)
        print(f"{batch:6d} {hidden:7d} {t_task * 1e3:9.2f} {t_torch * 1e3:9.2f} "
              f"{nbytes:10d} {nbytes / (t_task * 1e-3) / 1e9:8.1f}")


if __name__ == "__main__":
    main()
