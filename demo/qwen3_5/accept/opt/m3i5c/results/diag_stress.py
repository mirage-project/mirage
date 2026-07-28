#!/usr/bin/env python3
"""Root-cause diagnostic for stress_compaction.py's C2/C4 100% failure rate.

Hypothesis A (C4): alloc() uses torch.empty for the count/id buffer `a`
(size NUM_EXPERTS+1); only a[:n] and a[NUM_EXPERTS] are ever written by the
kernel. torch.equal(a, a2) in the original script compares the FULL buffer,
including the uninitialized tail a[n:NUM_EXPERTS], which is never written and
therefore holds whatever garbage torch.empty happened to return -- almost
certainly different between two independent allocations. This would make C4
fail near-100% even when the kernel is byte-for-byte deterministic on its
actual output.

Hypothesis B (C2): NUM_EXPERTS=256, bf16 logits (8-bit mantissa) -> the top-8
boundary is frequently EXACTLY tied in bf16, and the kernel's tie-break order
need not match torch.topk's. This is the exact phenomenon test_gate_topk.py
was reworked to be "tie-aware" for. The naive set-vs-oracle comparison in
stress_compaction.py has no such tie tolerance, so ties at the boundary
produce a "mismatch" that is not evidence of a real defect.
"""
import sys
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
scale = 0.05  # it % 4 == 0 case, the FIRST iteration's scale in the real script

logits = (torch.randn(rows, NUM_EXPERTS, device=dev) * scale).to(torch.bfloat16)

w, r, a = run(logits)
n = int(a[NUM_EXPERTS].item())
ids = a[:n].to(torch.long)

w2, r2, a2 = run(logits)
n2 = int(a2[NUM_EXPERTS].item())
ids2 = a2[:n2].to(torch.long)

print("=== HYPOTHESIS A: C4 buffer-tail check ===")
print("n, n2:", n, n2)
print("prefix a[:n] equal:", torch.equal(a[:n], a2[:n2]))
print("full-buffer a equal (what the script actually checks):", torch.equal(a, a2))
print("tail (a[n:NUM_EXPERTS]) equal:", torch.equal(a[n:NUM_EXPERTS], a2[n2:NUM_EXPERTS]))
print("weights w equal:", torch.equal(w, w2))
print("routing r equal:", torch.equal(r, r2))
diff_tail = (a[n:NUM_EXPERTS] != a2[n2:NUM_EXPERTS]).sum().item() if n == n2 else -1
print("num differing tail entries:", diff_tail, "of", NUM_EXPERTS - n)
print("sample tail a[n:n+8]:", a[n:n+8].tolist())
print("sample tail a2[n2:n2+8]:", a2[n2:n2+8].tolist())

print()
print("=== HYPOTHESIS B: C2 tie-margin check ===")
ref_vals, ref_idx = torch.topk(logits.float(), TOPK, dim=1)
want = torch.zeros(NUM_EXPERTS, device=dev, dtype=torch.int32)
want[ref_idx.reshape(-1)] = 1
got = torch.zeros(NUM_EXPERTS, device=dev, dtype=torch.int32)
if n > 0:
    got.index_fill_(0, ids, 1)
mismatch_positions = torch.nonzero(got != want).flatten().tolist()
print("total mismatching positions:", len(mismatch_positions), "of", NUM_EXPERTS)
print("n active (kernel):", n, " n active (oracle, should be <= rows*TOPK):",
      int(want.sum().item()))

# For each row, is the 8th vs 9th largest value (in float32, matching the
# bf16-quantized logits actually fed to the kernel) EXACTLY tied?
logits_f32 = logits.float()  # bf16-quantized values, upcast losslessly
sorted_vals, _ = torch.sort(logits_f32, dim=1, descending=True)
boundary_tied = (sorted_vals[:, TOPK - 1] == sorted_vals[:, TOPK]).sum().item()
print(f"rows with an EXACT tie at the top-{TOPK} boundary: {boundary_tied} / {rows}")

# How many distinct bf16 values collide overall (evidence ties are structural,
# not incidental, at this scale)
uniq_per_row = [torch.unique(logits_f32[i]).numel() for i in range(rows)]
print("distinct bf16 values per row (of 256):", uniq_per_row)
