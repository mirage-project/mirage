#!/usr/bin/env python3
"""Confirm whether reusing the SAME input tensor object (vs a fresh one each
call) is what triggers the row-0-only degeneration."""
import torch
import runtime_kernel_blackwell_moe_block_qwen35 as mk

NUM_EXPERTS = 256
TOPK = 8


def alloc(rows):
    return (
        torch.empty((rows, TOPK), device="cuda", dtype=torch.float32),
        torch.empty((NUM_EXPERTS, rows), device="cuda", dtype=torch.int32),
        torch.empty((NUM_EXPERTS + 1,), device="cuda", dtype=torch.int32),
    )


def run(logits, vpt=0, round_weights=False):
    rows = logits.size(0)
    w, r, a = alloc(rows)
    mk.topk_softmax_sm100(logits, w, r, a, vpt, round_weights)
    torch.cuda.synchronize()
    return w, r, a


torch.manual_seed(20260727)
dev = torch.device("cuda")
rows = 16
scale = 0.05

print("=== FRESH tensor every call (new torch.randn each time) ===")
for i in range(5):
    logits_i = (torch.randn(rows, NUM_EXPERTS, device=dev) * scale).to(torch.bfloat16)
    w, r, a = run(logits_i)
    n = int(a[NUM_EXPERTS].item())
    print(f"call {i}: data_ptr={logits_i.data_ptr()} n={n}")

print()
print("=== SAME tensor, but call .clone() each time (new address, same values) ===")
base = (torch.randn(rows, NUM_EXPERTS, device=dev) * scale).to(torch.bfloat16)
for i in range(5):
    logits_i = base.clone()
    w, r, a = run(logits_i)
    n = int(a[NUM_EXPERTS].item())
    print(f"call {i}: data_ptr={logits_i.data_ptr()} n={n}")

print()
print("=== SAME tensor object, same address, repeated (the original repro) ===")
base2 = (torch.randn(rows, NUM_EXPERTS, device=dev) * scale).to(torch.bfloat16)
for i in range(5):
    w, r, a = run(base2)
    n = int(a[NUM_EXPERTS].item())
    print(f"call {i}: data_ptr={base2.data_ptr()} n={n}")
