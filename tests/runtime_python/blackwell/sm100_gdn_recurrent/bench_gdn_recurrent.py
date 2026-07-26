"""Latency benchmark for `gdn_recurrent_sm100` (kernel-wrapper path).

INFORMATIONAL ONLY. These numbers are a standalone-kernel sanity check, not a
performance claim: the task runs inside the megakernel alongside 30 GDN layers'
worth of other work, so wall-clock there is set by scheduling and by the S
traffic, not by this launch.

What the shapes are for:

  decode  bs 1..16   the production decode step - one token per request. The
                     recurrence reads and writes 64 KiB of fp32 state per
                     (v-head, slot), so 32 heads x bs tasks move
                     bs * 4 MiB per layer. Arithmetic intensity is ~0.9
                     FLOP/byte (vllm-graph.md 2.1.7), i.e. this is the most
                     memory-bound op in the model.
  prefill 64..256    a chunked-prefill chunk. Here the state is loaded ONCE and
                     the token loop runs in shared memory, so cost is
                     compute-bound and grows linearly in Q_LEN - the recurrence
                     is sequential in t and cannot be split across tasks.

Run:
    python bench_gdn_recurrent.py
"""

import torch

import runtime_kernel_blackwell_gdn_recurrent as gdn

DEV = "cuda"
BF16 = torch.bfloat16
NUM_V_HEADS = 32
NUM_K_HEADS = 16
HEAD_K_DIM = 128
HEAD_V_DIM = 128
VAL_DIM = NUM_V_HEADS * HEAD_V_DIM
QKV_STRIDE = 8192
BA_STRIDE = 64
WARMUP = 20
ITERS = 200

NUM_GDN_LAYERS = 30      # Qwen3.5-35B-A3B: 30 of 40 layers are GDN


def bench(num_slots, q_len):
    total = num_slots * q_len
    g = torch.Generator(device=DEV).manual_seed(0)
    qkv = (torch.randn(total, QKV_STRIDE, generator=g, device=DEV) * 0.5).to(BF16)
    ba = (torch.randn(total, BA_STRIDE, generator=g, device=DEV) * 0.5).to(BF16)
    z = (torch.randn(total, VAL_DIM, generator=g, device=DEV) * 0.5).to(BF16)
    ad = torch.stack([torch.randn(NUM_V_HEADS, generator=g, device=DEV) * 0.5,
                      torch.randn(NUM_V_HEADS, generator=g, device=DEV)]).contiguous()
    nw = torch.ones(HEAD_V_DIM, dtype=torch.float32, device=DEV)
    state = (torch.randn(num_slots, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM,
                         generator=g, device=DEV) * 0.1).contiguous()
    out = torch.zeros(total, VAL_DIM, dtype=BF16, device=DEV)
    qo = torch.tensor([i * q_len for i in range(num_slots + 1)],
                      dtype=torch.int32, device=DEV)
    zs = torch.zeros(num_slots, dtype=torch.uint8, device=DEV)

    def once():
        gdn.gdn_recurrent_sm100(qkv, ba, ad, state, z, nw, out, qo, zs,
                                NUM_K_HEADS, None)

    for _ in range(WARMUP):
        once()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        once()
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1000.0 / ITERS

    # Compulsory state traffic: 64 KiB read + 64 KiB write per (head, slot).
    state_bytes = num_slots * NUM_V_HEADS * HEAD_V_DIM * HEAD_K_DIM * 4 * 2
    gbps = state_bytes / (us * 1e-6) / 1e9
    return us, gbps


def main():
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"shape: {NUM_V_HEADS} v-heads x {HEAD_V_DIM}x{HEAD_K_DIM} fp32 state "
          f"({NUM_V_HEADS * HEAD_V_DIM * HEAD_K_DIM * 4 / 2**20:.0f} MiB/slot)")

    print("\ndecode (Q_LEN = 1) - one task per (v-head, slot)")
    print(f"{'batch':>6} {'tasks':>6} {'us/layer':>10} {'state GB/s':>11} "
          f"{'ms/step, 30 layers':>19}")
    for bs in (1, 2, 4, 8, 16):
        us, gbps = bench(bs, 1)
        print(f"{bs:6d} {bs * NUM_V_HEADS:6d} {us:10.2f} {gbps:11.1f} "
              f"{us * NUM_GDN_LAYERS / 1000:19.3f}")

    print("\nchunked prefill (1 request) - state loaded once, token loop in smem")
    print(f"{'Q_LEN':>6} {'tasks':>6} {'us/layer':>10} {'us/token':>10}")
    for q_len in (8, 32, 64, 128, 256):
        us, _ = bench(1, q_len)
        print(f"{q_len:6d} {NUM_V_HEADS:6d} {us:10.2f} {us / q_len:10.3f}")

    print("\nchunked prefill (4 requests x Q_LEN) - fills more SMs")
    print(f"{'Q_LEN':>6} {'tasks':>6} {'us/layer':>10} {'us/token':>10}")
    for q_len in (32, 64, 128, 256):
        us, _ = bench(4, q_len)
        print(f"{q_len:6d} {4 * NUM_V_HEADS:6d} {us:10.2f} "
              f"{us / (4 * q_len):10.3f}")


if __name__ == "__main__":
    main()
