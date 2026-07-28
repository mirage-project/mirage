#!/usr/bin/env python3
"""Follow-up diagnostic: trace WHY the second same-input call returns n=8
instead of n=99. Print actual vs expected for the failing case, and find
where the "expected" value (99) actually comes from.
"""
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
logits = (torch.randn(rows, NUM_EXPERTS, device=dev) * scale).to(torch.bfloat16)

print("=== calling run() 5x in a row on the IDENTICAL logits tensor ===")
results = []
for i in range(5):
    w, r, a = run(logits)
    n = int(a[NUM_EXPERTS].item())
    ids = a[:n].to(torch.long).tolist()
    results.append((n, ids))
    print(f"call {i}: n={n} ids[:12]={ids[:12]}")

print()
print("=== is call 0 special (cold start), or does it alternate? ===")
print("all n values:", [r[0] for r in results])

print()
print("=== warm-up with DUMMY logits first, then compare two REAL calls ===")
dummy = (torch.randn(rows, NUM_EXPERTS, device=dev) * scale).to(torch.bfloat16)
_ = run(dummy)  # throwaway warm-up call, different data
wA, rA, aA = run(logits)
nA = int(aA[NUM_EXPERTS].item())
wB, rB, aB = run(logits)
nB = int(aB[NUM_EXPERTS].item())
print(f"post-warmup call A: n={nA}  call B: n={nB}  equal={nA==nB}")
if nA == nB:
    print("  ids equal:", torch.equal(aA[:nA].to(torch.long), aB[:nB].to(torch.long)))
    print("  full buffer a equal:", torch.equal(aA, aB))
    print("  weights equal:", torch.equal(wA, wB))
    print("  routing equal:", torch.equal(rA, rB))

print()
print("=== does the oracle (99) match torch.topk directly? ===")
ref_vals, ref_idx = torch.topk(logits.float(), TOPK, dim=1)
want = torch.zeros(NUM_EXPERTS, dtype=torch.int32)
want[ref_idx.reshape(-1).cpu()] = 1
print("oracle distinct active count (torch.topk-based):", int(want.sum().item()))

print()
print("=== reference: is 'n=8' consistent with only-1-row worth of activity? ===")
n0, ids0 = results[0]
n1, ids1 = results[1]
print("call0 ids (full, n=%d):" % n0, ids0)
print("call1 ids (full, n=%d):" % n1, ids1)
print("is call1's id-set a subset of call0's?",
      set(ids1).issubset(set(ids0)) if n1 else None)
# what would row 0 alone contribute under torch.topk?
row0_top8 = torch.topk(logits[0].float(), TOPK).indices.sort().values.tolist()
print("row 0's own top-8 (oracle):", row0_top8)
