"""PR696 multi-CTA topk_sigmoid: correctness (vs single-CTA, which is already
validated vs the pytorch reference) + timing (single vs multi-CTA).

Single-CTA = today's MPK path (1 CTA loops all row-chunks).
Multi-CTA   = PR696 prefill path (init markers -> N CTAs, one per chunk ->
              separate compaction kernel).
"""
import torch
import runtime_kernel_moe_sigmoid as K

NUM_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
NUM_GROUPS = 8
TOPK_GROUP = 4
RSF = 2.5
SEED = 42


def run(fn, gating, bias):
    batch = gating.size(0)
    w = torch.empty(batch, NUM_EXPERTS_PER_TOK, device="cuda", dtype=torch.float)
    ri = torch.zeros((NUM_EXPERTS, batch), device="cuda", dtype=torch.int32)
    aid = torch.full((NUM_EXPERTS + 1,), -7, device="cuda", dtype=torch.int32)
    fn(gating.clone(), bias, w, ri, aid, RSF, NUM_GROUPS, TOPK_GROUP)
    torch.cuda.synchronize()
    return w, ri, aid


print("=" * 72)
print("CORRECTNESS — multi-CTA vs single-CTA (single already == pytorch ref)")
print("=" * 72)
g = torch.Generator(device="cuda").manual_seed(SEED)
for batch in [1, 8, 16, 32, 64, 128, 256]:
    gating = torch.randn((batch, NUM_EXPERTS), device="cuda",
                         dtype=torch.bfloat16, generator=g)
    bias = torch.randn(NUM_EXPERTS, device="cuda",
                       dtype=torch.float32, generator=g) * 0.1

    w_s, ri_s, aid_s = run(K.topk_sigmoid_sm100, gating, bias)
    w_m, ri_m, aid_m = run(K.topk_sigmoid_sm100_multicta, gating, bias)

    n_s = int(aid_s[-1].item())
    n_m = int(aid_m[-1].item())
    set_s = set(aid_s[:n_s].tolist())
    set_m = set(aid_m[:n_m].tolist())

    ok_w = torch.allclose(w_s, w_m, rtol=1e-3, atol=1e-3)
    ok_ri = torch.equal(ri_s, ri_m)
    ok_set = (set_s == set_m)
    status = "PASS" if (ok_w and ok_ri and ok_set) else "FAIL"
    print(f"  batch={batch:>4}  weights:{ok_w}  routing:{ok_ri}  "
          f"active_set:{ok_set} ({n_s} vs {n_m})  -> {status}")
    if status == "FAIL":
        if not ok_w:
            print("    weights max diff:", (w_s - w_m).abs().max().item())
        if not ok_ri:
            print("    routing mismatch count:", (ri_s != ri_m).sum().item())
        if not ok_set:
            print("    set diff:", set_s ^ set_m)

print("\n" + "=" * 72)
print("TIMING — single-CTA vs multi-CTA")
print("=" * 72)
WARMUP, REPS = 50, 2000
print(f"{'batch':>6} {'single us':>11} {'multi us':>11} {'speedup':>9}")
for batch in [1, 8, 32, 64, 128, 256]:
    gating = torch.randn((batch, NUM_EXPERTS), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(NUM_EXPERTS, device="cuda", dtype=torch.float32) * 0.1
    w = torch.empty(batch, NUM_EXPERTS_PER_TOK, device="cuda", dtype=torch.float)
    ri = torch.zeros((NUM_EXPERTS, batch), device="cuda", dtype=torch.int32)
    aid = torch.empty((NUM_EXPERTS + 1,), device="cuda", dtype=torch.int32)

    def bench(fn):
        for _ in range(WARMUP):
            fn(gating, bias, w, ri, aid, RSF, NUM_GROUPS, TOPK_GROUP)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(REPS):
            fn(gating, bias, w, ri, aid, RSF, NUM_GROUPS, TOPK_GROUP)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / REPS * 1000

    su = bench(K.topk_sigmoid_sm100)
    mu = bench(K.topk_sigmoid_sm100_multicta)
    print(f"{batch:>6} {su:>11.3f} {mu:>11.3f} {su/mu:>8.2f}x")
