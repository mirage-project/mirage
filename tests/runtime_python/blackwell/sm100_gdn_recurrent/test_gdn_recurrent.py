"""Kernel-wrapper unit tests for `gdn_recurrent_sm100`.

The oracle test (`test_gdn_recurrent_oracle.py`) owns the numerics against the
real checkpoint. This file owns the STRUCTURAL properties, which the oracle's
single-request dumps cannot reach:

  1. shape coverage - every instantiated (heads, dims, strides) combination,
     including the GVA ratios 1/2/4 and a strided qkv row
  2. GVA head mapping - v-head hv must read q/k from head hv // (Hv/Hk);
     a wrong mapping is invisible at ratio 1 and at Hv == Hk
  3. per-slot isolation - interleaved requests with different Q_LENs must not
     see each other's state, and a slot's result must not depend on what its
     neighbours are doing
  4. `zero_state` (the step == 0 predicate) - a fresh slot must ignore whatever
     is in the pool, and must write back the same state a zeroed pool would
  5. parked slot (Q_LEN == 0) - state must survive completely untouched
  6. carried chain - N sequential single-token calls must equal one N-token
     chunk, which is the property the decode loop depends on
  7. state layout - the pool slice a task writes must be exactly its own
     (slot, head); neighbouring heads and slots stay byte-identical

Run:
    python test_gdn_recurrent.py
"""

import itertools

import torch
import torch.nn.functional as F

import runtime_kernel_blackwell_gdn_recurrent as gdn

DEV = "cuda"
BF16 = torch.bfloat16
EPS = 1e-6

# (num_v_heads, num_k_heads, head_k_dim, head_v_dim, qkv_stride, ba_stride,
#  z_stride, out_stride) - must match GDN_RECURRENT_CASES in the wrapper .cu
SHAPES = [
    (32, 16, 128, 128, 8192, 64, 4096, 4096),
    (32, 16, 128, 128, 12288, 64, 4096, 4096),
    (8, 4, 128, 128, 2048, 16, 1024, 1024),
    (4, 2, 32, 32, 256, 8, 128, 128),
    (4, 4, 32, 32, 384, 8, 128, 128),
    (4, 1, 32, 32, 192, 8, 128, 128),
    (2, 1, 64, 64, 256, 4, 128, 128),
]

FAILURES = []


def ok(cond, msg):
    if cond:
        print(f"    PASS  {msg}")
    else:
        print(f"    FAIL  {msg}")
        FAILURES.append(msg)


def make_inputs(shape, num_tokens, num_slots, seed):
    hv, hk, dk, dv, qs, bs, zs, os_ = shape
    g = torch.Generator(device=DEV).manual_seed(seed)
    qkv = (torch.randn(num_tokens, qs, generator=g, device=DEV) * 0.5).to(BF16)
    ba = (torch.randn(num_tokens, bs, generator=g, device=DEV) * 0.5).to(BF16)
    z = (torch.randn(num_tokens, zs, generator=g, device=DEV) * 0.5).to(BF16)
    ad = torch.stack([
        torch.randn(hv, generator=g, device=DEV) * 0.5,
        torch.randn(hv, generator=g, device=DEV),
    ]).contiguous()
    norm_w = (torch.randn(dv, generator=g, device=DEV) * 0.2 + 1.0).contiguous()
    state = (torch.randn(num_slots, hv, dv, dk, generator=g, device=DEV) * 0.1
             ).contiguous()
    return qkv, ba, z, ad, norm_w, state


def run(shape, qkv, ba, ad, state, z, norm_w, qo, zero_state, want_o=False):
    hv, hk, dk, dv, qs, bs, zs, os_ = shape
    T = qkv.shape[0]
    out = torch.zeros(T, os_, dtype=BF16, device=DEV)
    o_dbg = torch.zeros(T, os_, dtype=BF16, device=DEV) if want_o else None
    st = state.clone().contiguous()
    gdn.gdn_recurrent_sm100(
        qkv.contiguous(), ba.contiguous(), ad.contiguous(), st,
        z.contiguous(), norm_w.contiguous(), out,
        torch.tensor(qo, dtype=torch.int32, device=DEV),
        torch.tensor(zero_state, dtype=torch.uint8, device=DEV),
        hk, o_dbg)
    torch.cuda.synchronize()
    return (out, st, o_dbg) if want_o else (out, st)


def ref_chain(shape, qkv, ba, z, ad, norm_w, S0, zero_state):
    """Inline torch reference in HF's rounding order (see the oracle test)."""
    hv_n, hk_n, dk, dv, qs, bs, zs, os_ = shape
    key_dim = hk_n * dk
    T = qkv.shape[0]
    S = (torch.zeros_like(S0) if zero_state else S0.clone()).float()
    out = torch.zeros(T, os_, dtype=BF16, device=DEV)
    o_all = torch.zeros(T, os_, dtype=BF16, device=DEV)
    beta = ba[:, :hv_n].sigmoid().float()
    g = -ad[0].exp() * F.softplus(ba[:, hv_n:2 * hv_n].float() + ad[1])
    scale = 1.0 / (dk ** 0.5)
    for t in range(T):
        o_t = torch.zeros(hv_n, dv, dtype=torch.float32, device=DEV)
        for h in range(hv_n):
            ih = h // (hv_n // hk_n)
            q = qkv[t, ih * dk:(ih + 1) * dk]
            k = qkv[t, key_dim + ih * dk:key_dim + (ih + 1) * dk]
            v = qkv[t, 2 * key_dim + h * dv:2 * key_dim + (h + 1) * dv].float()
            qn = (q * torch.rsqrt((q * q).sum(-1, keepdim=True)
                                  + EPS)).float() * scale
            kn = (k * torch.rsqrt((k * k).sum(-1, keepdim=True) + EPS)).float()
            Sh = S[h] * g[t, h].exp()
            delta = (v - (Sh * kn.unsqueeze(0)).sum(-1)) * beta[t, h]
            Sh = Sh + delta.unsqueeze(-1) * kn.unsqueeze(0)
            o_t[h] = (Sh * qn.unsqueeze(0)).sum(-1)
            S[h] = Sh
        ob = o_t.to(BF16)
        o_all[t, :hv_n * dv] = ob.reshape(-1)
        obf = ob.float()
        xh = (obf * torch.rsqrt(obf.pow(2).mean(-1, keepdim=True)
                                + EPS)).to(BF16).float()
        y = norm_w.float() * xh
        out[t, :hv_n * dv] = (
            y * F.silu(z[t, :hv_n * dv].reshape(hv_n, dv).float())
        ).to(BF16).reshape(-1)
    return out, o_all, S


# --------------------------------------------------------------- 1 & 2: shapes
def test_shapes_and_gva():
    print("\n[1/2] shape coverage + GVA head mapping vs a torch reference")
    for shape in SHAPES:
        hv_n, hk_n, dk, dv = shape[0], shape[1], shape[2], shape[3]
        T = 4
        qkv, ba, z, ad, norm_w, state = make_inputs(shape, T, 1, 7 + hv_n + dk)
        out, st, o_dbg = run(shape, qkv, ba, ad, state, z, norm_w, [0, T],
                             [0], want_o=True)
        r_out, r_o, r_S = ref_chain(shape, qkv, ba, z, ad, norm_w, state[0],
                                    False)
        eo = (o_dbg.float() - r_o.float()).abs().max().item()
        ey = (out.float() - r_out.float()).abs().max().item()
        es = (st[0] - r_S).abs().max().item()
        srms = r_S.pow(2).mean().sqrt().item()
        tag = (f"Hv={hv_n:2d} Hk={hk_n:2d} Dk={dk:3d} Dv={dv:3d} "
               f"qkv_stride={shape[4]:5d}")
        ok(eo < 5e-2 and ey < 5e-2 and es / max(srms, 1e-9) < 1e-4,
           f"{tag}  o={eo:.2e} y={ey:.2e} S={es:.2e} (|S|rms={srms:.2e})")

        # GVA: at ratio > 1, deliberately breaking the mapping must change the
        # answer - proves the kernel is really reading head hv//(Hv/Hk).
        if hv_n // hk_n > 1:
            key_dim = hk_n * dk
            swapped = qkv.clone()
            # swap q/k head 0 with head 1: heads 0,1 (which share q/k head 0)
            # must change, and they only can if the mapping is live
            a0 = swapped[:, 0:dk].clone()
            swapped[:, 0:dk] = swapped[:, dk:2 * dk]
            swapped[:, dk:2 * dk] = a0
            out2, _ = run(shape, swapped, ba, ad, state, z, norm_w, [0, T], [0])
            ok(not torch.equal(out2[:, :2 * dv], out[:, :2 * dv]),
               f"{tag}  GVA mapping is live (q-head swap changes v-heads 0,1)")


# ---------------------------------------------------------- 3 & 7: slot isolate
def test_slot_isolation():
    print("\n[3/7] per-slot isolation with interleaved, different-length "
          "requests")
    shape = (8, 4, 128, 128, 2048, 16, 1024, 1024)
    hv_n, hk_n, dk, dv = shape[0], shape[1], shape[2], shape[3]
    lens = [3, 1, 5, 2]
    qo = [0]
    for n in lens:
        qo.append(qo[-1] + n)
    T = qo[-1]
    qkv, ba, z, ad, norm_w, state = make_inputs(shape, T, len(lens), 4242)
    zs = [1, 0, 0, 1]
    out, st = run(shape, qkv, ba, ad, state, z, norm_w, qo, zs)

    # Each slot alone must reproduce exactly what it produced in the batch.
    for s, n in enumerate(lens):
        lo, hi = qo[s], qo[s + 1]
        solo_out, solo_st = run(
            shape, qkv[lo:hi], ba[lo:hi], ad, state[s:s + 1], z[lo:hi],
            norm_w, [0, n], [zs[s]])
        ok(torch.equal(out[lo:hi], solo_out),
           f"slot {s} (Q_LEN={n}, zero_state={zs[s]}) output is batch-invariant")
        ok(torch.equal(st[s], solo_st[0]),
           f"slot {s} state is batch-invariant")

    # And it must match the reference.
    for s, n in enumerate(lens):
        lo, hi = qo[s], qo[s + 1]
        r_out, _, r_S = ref_chain(shape, qkv[lo:hi], ba[lo:hi], z[lo:hi], ad,
                                  norm_w, state[s], zs[s] == 1)
        ok((out[lo:hi].float() - r_out.float()).abs().max().item() < 5e-2,
           f"slot {s} matches the torch reference")


# --------------------------------------------------------------- 4: step == 0
def test_zero_state_reset():
    print("\n[4] step == 0 predicate (fresh slot ignores the stale pool)")
    shape = (8, 4, 128, 128, 2048, 16, 1024, 1024)
    T = 3
    qkv, ba, z, ad, norm_w, state = make_inputs(shape, T, 1, 99)

    fresh_out, fresh_st = run(shape, qkv, ba, ad, state, z, norm_w, [0, T], [1])
    zeroed = torch.zeros_like(state)
    zero_out, zero_st = run(shape, qkv, ba, ad, zeroed, z, norm_w, [0, T], [0])
    ok(torch.equal(fresh_out, zero_out),
       "zero_state=1 on a dirty pool == zero_state=0 on a zeroed pool (output)")
    ok(torch.equal(fresh_st, zero_st),
       "... and the written-back state is identical too")

    carried_out, _ = run(shape, qkv, ba, ad, state, z, norm_w, [0, T], [0])
    ok(not torch.equal(fresh_out, carried_out),
       "the predicate is LIVE (carrying the stale state gives a different "
       "answer)")


# -------------------------------------------------------------- 5: parked slot
def test_parked_slot():
    print("\n[5] parked slot (Q_LEN == 0) leaves its state byte-identical")
    shape = (8, 4, 128, 128, 2048, 16, 1024, 1024)
    lens = [2, 0, 3]
    qo = [0]
    for n in lens:
        qo.append(qo[-1] + n)
    T = qo[-1]
    qkv, ba, z, ad, norm_w, state = make_inputs(shape, T, len(lens), 31337)
    out, st = run(shape, qkv, ba, ad, state, z, norm_w, qo,
                  [0, 0, 0])
    ok(torch.equal(st[1], state[1]),
       "parked slot 1 state untouched (bit-identical)")
    # ... even when the parked slot is flagged zero_state, which would otherwise
    # zero the pool
    out2, st2 = run(shape, qkv, ba, ad, state, z, norm_w, qo, [0, 1, 0])
    ok(torch.equal(st2[1], state[1]),
       "parked slot ignores zero_state too (early return precedes it)")
    ok(torch.equal(st[0], st2[0]) and torch.equal(st[2], st2[2]),
       "active neighbours of a parked slot are unaffected")


# ------------------------------------------------------------- 6: carried chain
def test_carried_chain():
    print("\n[6] carried chain: N single-token decodes == one N-token chunk")
    shape = (8, 4, 128, 128, 2048, 16, 1024, 1024)
    hv_n, dv = shape[0], shape[3]
    T = 8
    qkv, ba, z, ad, norm_w, state = make_inputs(shape, T, 1, 777)

    chunk_out, chunk_st = run(shape, qkv, ba, ad, state, z, norm_w, [0, T], [0])

    st = state.clone()
    step_out = torch.zeros(T, shape[7], dtype=BF16, device=DEV)
    for t in range(T):
        o1, st = run(shape, qkv[t:t + 1], ba[t:t + 1], ad, st, z[t:t + 1],
                     norm_w, [0, 1], [0])
        step_out[t] = o1[0]
    ok(torch.equal(step_out, chunk_out),
       "per-token decode output == chunked output (bit-identical)")
    ok(torch.equal(st, chunk_st),
       "per-token carried state == chunked state (bit-identical)")

    # ... and the whole chain still tracks the torch reference
    r_out, _, r_S = ref_chain(shape, qkv, ba, z, ad, norm_w, state[0], False)
    ey = (chunk_out.float() - r_out.float()).abs().max().item()
    es = (chunk_st[0] - r_S).abs().max().item()
    srms = r_S.pow(2).mean().sqrt().item()
    ok(ey < 5e-2 and es / srms < 1e-4,
       f"8-step chain vs torch reference: y={ey:.2e} S={es:.2e} "
       f"(S rel {es / srms:.2e})")


# ----------------------------------------------------- 7b: state slice hygiene
def test_state_slice_hygiene():
    print("\n[7b] a task writes ONLY its own (slot, head) state slice")
    shape = (8, 4, 128, 128, 2048, 16, 1024, 1024)
    hv_n = shape[0]
    lens = [2, 2]
    qkv, ba, z, ad, norm_w, state = make_inputs(shape, 4, 2, 5150)
    # Only slot 0 is active; slot 1 is parked.
    _, st = run(shape, qkv, ba, ad, state, z, norm_w, [0, 4, 4], [0, 0])
    ok(torch.equal(st[1], state[1]), "inactive slot 1 fully untouched")
    changed = [h for h in range(hv_n) if not torch.equal(st[0, h], state[0, h])]
    ok(changed == list(range(hv_n)),
       f"every head of the active slot was updated ({len(changed)}/{hv_n})")


# ------------------------------------------- 8: decode split path, BIT-EXACT
# (split, depth) pairs the wrapper instantiates; must match
# GDN_DECODE_SPLIT_CASES in runtime_kernel_wrapper_sm100.cu.
SPLIT_CASES = {
    (32, 16, 128, 128, 8192, 64, 4096, 4096): [
        (1, 2), (2, 2), (4, 2), (8, 2), (16, 2), (32, 2),
        (1, 3), (1, 4), (2, 4), (4, 4),
    ],
    (32, 16, 128, 128, 12288, 64, 4096, 4096): [(1, 2), (2, 2), (4, 2)],
    (8, 4, 128, 128, 2048, 16, 1024, 1024): [(1, 2), (2, 2), (4, 2)],
    (4, 2, 32, 32, 256, 8, 128, 128): [(1, 2), (2, 2), (4, 2)],
    (4, 4, 32, 32, 384, 8, 128, 128): [(1, 2), (2, 2), (4, 2)],
    (4, 1, 32, 32, 192, 8, 128, 128): [(1, 2), (2, 2), (4, 2)],
    (2, 1, 64, 64, 256, 4, 128, 128): [(1, 2), (2, 2), (4, 2)],
}


def run_split(shape, qkv, ba, ad, state, z, norm_w, qo, split, depth,
              scratch=None):
    hv_n, hk_n, dk, dv, qs, bs, zs, os_ = shape
    num_slots = state.shape[0]
    out = torch.zeros(qkv.shape[0], os_, dtype=BF16, device=DEV)
    st = state.clone().contiguous()
    if scratch is None:
        scratch = torch.zeros(num_slots, hv_n, dv + 8, dtype=torch.float32,
                              device=DEV)
    gdn.gdn_recurrent_decode_split_sm100(
        qkv.contiguous(), ba.contiguous(), ad.contiguous(), st,
        z.contiguous(), norm_w.contiguous(), out, scratch,
        torch.tensor(qo, dtype=torch.int32, device=DEV),
        hk_n, split, depth)
    torch.cuda.synchronize()
    return out, st, scratch


def test_decode_split_bit_exact():
    """The decode fast path must be BYTE-identical to the golden task impl.

    This is the gate the ferret loop that produced this kernel ran on every
    iteration (integer memcmp of `out` AND the updated fp32 `state`); re-run
    here against the in-tree golden so the MPK port carries the same guarantee
    at every instantiated shape and split.
    """
    print("\n[8] decode split path is bit-exact vs the golden task impl")
    for shape, cases in SPLIT_CASES.items():
        dv = shape[3]
        for num_slots in (1, 3):
            qo = list(range(num_slots + 1))          # one token per slot
            qkv, ba, z, ad, norm_w, state = make_inputs(
                shape, num_slots, num_slots, 9100 + num_slots)
            g_out, g_st = run(shape, qkv, ba, ad, state, z, norm_w, qo,
                              [0] * num_slots)
            for split, depth in cases:
                c_out, c_st, scratch = run_split(
                    shape, qkv, ba, ad, state, z, norm_w, qo, split, depth)
                same_out = torch.equal(c_out.view(torch.int16),
                                       g_out.view(torch.int16))
                same_st = torch.equal(c_st.view(torch.int32),
                                      g_st.view(torch.int32))
                ok(same_out and same_st,
                   f"{shape} slots={num_slots} split={split} depth={depth}: "
                   f"out bit-exact={same_out} state bit-exact={same_st}")
                ctr = scratch[..., dv].view(torch.int32)
                ok(bool((ctr == 0).all()),
                   f"{shape} slots={num_slots} split={split} depth={depth}: "
                   "arrival counters self-reset")


def test_decode_split_counter_reuse():
    """Back-to-back launches on ONE scratch buffer must stay bit-exact.

    The self-reset is what lets all 30 GDN layers and every decode step share
    a single scratch buffer; a leaked counter would elect the wrong epilogue
    task (or none) on the next use.
    """
    print("\n[8b] repeated launches reuse one scratch buffer correctly")
    shape = (32, 16, 128, 128, 8192, 64, 4096, 4096)
    hv_n, dv, os_ = shape[0], shape[3], shape[7]
    num_slots = 2
    qo = list(range(num_slots + 1))
    qkv, ba, z, ad, norm_w, state = make_inputs(shape, num_slots, num_slots,
                                                9313)
    g_out, g_st = run(shape, qkv, ba, ad, state, z, norm_w, qo,
                      [0] * num_slots)
    for split in (2, 4, 16):
        scratch = torch.zeros(num_slots, hv_n, dv + 8, dtype=torch.float32,
                              device=DEV)
        good = True
        for _ in range(4):
            c_out, c_st, _ = run_split(shape, qkv, ba, ad, state, z, norm_w,
                                       qo, split, 2, scratch=scratch)
            good &= torch.equal(c_out.view(torch.int16),
                                g_out.view(torch.int16))
            good &= torch.equal(c_st.view(torch.int32), g_st.view(torch.int32))
        ok(good, f"split={split}: 4 back-to-back launches on one scratch, "
                 "all bit-exact")


def main():
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    test_shapes_and_gva()
    test_slot_isolation()
    test_zero_state_reset()
    test_parked_slot()
    test_carried_chain()
    test_state_slice_hygiene()
    test_decode_split_bit_exact()
    test_decode_split_counter_reuse()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nALL GDN_RECURRENT UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
