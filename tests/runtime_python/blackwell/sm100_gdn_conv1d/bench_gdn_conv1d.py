"""Latency benchmark for `gdn_conv1d_sm100`.

One task per request slot, so the launch config sweeps the batch sizes v1
targets (docs/qwen35/v1-architecture.md 9: batch 1..16) at both decode
(Q_LEN = 1) and chunked-prefill (Q_LEN > 1) chunk lengths.

The op is memory-bound by construction: per token it reads/writes
2 * CONV_DIM * 2 B of activations plus 2 * (K-1) * CONV_DIM * 2 B of state per
slot, against only 2*K FLOPs per channel. The reported effective bandwidth is
the number to watch, not the FLOP rate.

Run:  python bench_gdn_conv1d.py
"""

import argparse

import torch

import runtime_kernel_blackwell_gdn_conv1d as conv_kernel

DEV = "cuda"
BF16 = torch.bfloat16
CONV_DIM = 8192
KERNEL_SIZE = 4

# (num_slots, q_len_per_slot) - the shapes v1 actually schedules.
CONFIGS = [
    (1, 1),     # batch 1 decode
    (4, 1),
    (8, 1),
    (16, 1),    # batch 16 decode (the AC-4 target point)
    (1, 256),   # single-request prefill chunk (workload pin 256)
    (4, 256),
    (1, 1024),  # long prefill chunk (workload pin 1024)
]

# grid.y: how many channel blocks the CONV_DIM is split into. 1 = one task per
# request (a whole prefill chunk on one SM); 32 = vLLM's Triton split
# (BLOCK_N = 256 over 8192 channels).
CHANNEL_BLOCKS = [1, 8, 32]


def bench(num_slots, q_len, num_channel_blocks=1, iters=200, warmup=20):
    total = num_slots * q_len
    gen = torch.Generator(device=DEV).manual_seed(0)
    x = torch.randn(total, CONV_DIM, generator=gen, device=DEV,
                    dtype=torch.float32).to(BF16)
    w = (torch.randn(CONV_DIM, KERNEL_SIZE, generator=gen, device=DEV,
                     dtype=torch.float32) * 0.05).to(BF16)
    state = torch.randn(num_slots, KERNEL_SIZE - 1, CONV_DIM, generator=gen,
                        device=DEV, dtype=torch.float32).to(BF16)
    out = torch.zeros(total, CONV_DIM, dtype=BF16, device=DEV)
    qo = torch.tensor([i * q_len for i in range(num_slots + 1)],
                      dtype=torch.int32, device=DEV)
    zs = torch.zeros(num_slots, dtype=torch.uint8, device=DEV)

    for _ in range(warmup):
        conv_kernel.gdn_conv1d_sm100(x, w, state, out, qo, zs,
                                     num_channel_blocks)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        conv_kernel.gdn_conv1d_sm100(x, w, state, out, qo, zs,
                                     num_channel_blocks)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters

    act_bytes = 2 * total * CONV_DIM * 2
    state_bytes = 2 * num_slots * (KERNEL_SIZE - 1) * CONV_DIM * 2
    w_bytes = CONV_DIM * KERNEL_SIZE * 2
    gbs = (act_bytes + state_bytes + w_bytes) / (ms * 1e-3) / 1e9
    return ms, gbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"CONV_DIM={CONV_DIM} K={KERNEL_SIZE}")
    print("NOTE: a standalone launch has a ~16 us floor on this box, so the "
          "decode rows are launch-bound, not kernel-bound; inside the "
          "megakernel there is no per-task launch.")
    header = f"{'slots':>6} {'q_len':>6} {'tokens':>7}"
    for ncb in CHANNEL_BLOCKS:
        header += f" {'ms@y=' + str(ncb):>10} {'GB/s':>8}"
    print(header)
    for num_slots, q_len in CONFIGS:
        line = f"{num_slots:>6} {q_len:>6} {num_slots * q_len:>7}"
        for ncb in CHANNEL_BLOCKS:
            ms, gbs = bench(num_slots, q_len, ncb, iters=args.iters)
            line += f" {ms:>10.5f} {gbs:>8.1f}"
        print(line)


if __name__ == "__main__":
    main()
