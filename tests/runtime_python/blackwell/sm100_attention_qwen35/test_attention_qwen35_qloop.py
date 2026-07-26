#!/usr/bin/env python3
"""M2-I6 -- the `max_tokens_per_pass` Q-loop, tested as an EQUIVALENCE.

The claim of v1-architecture.md §4.3 is that splitting a request's queries into
ceil(Q_LEN / max_tokens_per_pass) passes over one smem arena computes the same
thing as a single pass -- so the smem arena can be sized by the pass instead of
by max-batched-tokens. That is only believable if a split run reproduces an
unsplit run BIT-FOR-BIT.

At the Qwen3.5 shape (GQA 8:1, head_dim 256) a single 8-row pass does not exist
-- probe P3 shows MAX_TOKENS=8 blows the 201 KiB budget, which is the whole
reason the Q-loop is needed. The equivalence is therefore proved at a smaller
shape where BOTH sides fit (4 Q / 1 KV head, head_dim 128, MAX_TOKENS up to 8),
and the Qwen3.5 shape is then checked for pass-size INVARIANCE (4x1, 2x2 and
1x4 pass splits must agree with each other) plus the oracle comparison in
`test_attention_qwen35_oracle.py`.

Also covers:
  * the causal mask under splitting -- a query in pass 1 must still attend to
    keys produced by queries in pass 0 (the failure mode a naive per-pass
    causal bound would introduce);
  * KV-write idempotence across passes;
  * the gated variant under splitting.
"""

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "demo", "qwen3_5"))

import runtime_kernel_blackwell_attention_qwen35 as K  # noqa: E402

PAGE_SIZE = 64
MAX_PAGES = 8
DEV = "cuda"
BF = torch.bfloat16


def meta(num_tokens, seq_len):
    npages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    t = lambda x: torch.tensor(x, dtype=torch.int32, device=DEV)
    return (t([0, num_tokens]), t([0, npages]), t(list(range(npages))),
            t([seq_len - (npages - 1) * PAGE_SIZE]))


def run(cfg, qkv, ctx_k, ctx_v, num_tokens, seq_len):
    qo_per_kv, kv_heads, head_dim, max_tokens, gate, q_pass = cfg
    kc = torch.zeros(MAX_PAGES, PAGE_SIZE, kv_heads, head_dim, dtype=BF, device=DEV)
    vc = torch.zeros_like(kc)
    for pos in range(seq_len - num_tokens):
        kc[pos // PAGE_SIZE, pos % PAGE_SIZE] = ctx_k[:, pos, :]
        vc[pos // PAGE_SIZE, pos % PAGE_SIZE] = ctx_v[:, pos, :]
    out = torch.zeros(num_tokens, qo_per_kv * kv_heads * head_dim, dtype=BF, device=DEV)
    qn = torch.ones(head_dim, dtype=BF, device=DEV)
    kn = torch.ones(head_dim, dtype=BF, device=DEV)
    cos = torch.ones(2048, head_dim, dtype=BF, device=DEV)
    sin = torch.zeros(2048, head_dim, dtype=BF, device=DEV)
    # real (non-identity) rotation so the RoPE path is actually exercised
    ang = torch.arange(2048, device=DEV).float()[:, None] * torch.logspace(
        0, -6, head_dim // 2, device=DEV)[None, :]
    cos[:, : head_dim // 2] = torch.cos(ang).to(BF)
    cos[:, head_dim // 2:] = torch.cos(ang).to(BF)
    sin[:, : head_dim // 2] = torch.sin(ang).to(BF)
    sin[:, head_dim // 2:] = torch.sin(ang).to(BF)
    qo, kvp, kvi, last = meta(num_tokens, seq_len)
    K.attention_qwen35(qkv, kc, vc, out, qo, kvp, kvi, last, qn, kn, cos, sin,
                       1, qo_per_kv, kv_heads, head_dim, max_tokens, gate,
                       q_pass, True, True)
    k_out = torch.stack([kc[p // PAGE_SIZE, p % PAGE_SIZE] for p in range(seq_len)], 1)
    v_out = torch.stack([vc[p // PAGE_SIZE, p % PAGE_SIZE] for p in range(seq_len)], 1)
    return out, k_out, v_out


def make_input(qo_per_kv, kv_heads, head_dim, T, gate, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    stride = 2 * head_dim if gate else head_dim
    width = (qo_per_kv * stride + 2 * head_dim) * kv_heads
    return (torch.randn(T, width, generator=g).to(BF).to(DEV) * 0.5).contiguous()


def make_ctx(kv_heads, head_dim, n, seed):
    g = torch.Generator(device="cpu").manual_seed(seed + 991)
    k = (torch.randn(kv_heads, max(n, 1), head_dim, generator=g) * 0.5).to(BF).to(DEV)
    return k, torch.zeros_like(k).copy_(
        (torch.randn(kv_heads, max(n, 1), head_dim, generator=g) * 0.5).to(BF).to(DEV))


def check(rows, name, a, b, must_match=True):
    eq = torch.equal(a.float(), b.float())
    d = (a.float() - b.float()).abs()
    rows.append({"case": name, "bit_identical": bool(eq),
                 "num_diff": int((a.float() != b.float()).sum()),
                 "numel": int(a.numel()), "max_abs": float(d.max()),
                 "expected_match": must_match})
    return eq == must_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "qloop_result.json"))
    args = ap.parse_args()
    rows, failures = [], []

    # ---------------- A. small shape: split vs UNSPLIT, bit-for-bit ---------
    # 4 Q / 1 KV head, head_dim 128 -- MAX_TOKENS=8 fits here, so the reference
    # "one pass of 8" instantiation exists.
    for gate in (0, 1):
        for T, seq_len in ((8, 8), (8, 40), (5, 37), (1, 33)):
            qkv = make_input(4, 1, 128, T, gate, seed=1000 + T + 7 * gate)
            ck, cv = make_ctx(1, 128, seq_len - T, seed=T + gate)
            ref = run((4, 1, 128, 8, gate, 0), qkv, ck, cv, T, seq_len)
            for mt, qp, label in ((8, 4, "same arena, 2x4"),
                                  (4, 4, "arena 4, 2x4 (production form)"),
                                  (2, 2, "arena 2, 4x2")):
                if gate == 1 and (mt, qp) != (4, 4):
                    continue  # only the production pair is instantiated gated
                got = run((4, 1, 128, mt, gate, qp), qkv, ck, cv, T, seq_len)
                base = f"small gate={gate} T={T} seq={seq_len} [{label}]"
                for i, what in enumerate(("out", "kv_k", "kv_v")):
                    if not check(rows, f"{base} {what}", got[i], ref[i]):
                        failures.append(f"{base} {what} differs from the unsplit run")

    # ---------------- B. Qwen3.5 shape: pass-size invariance ----------------
    # No unsplit reference exists (P3: MAX_TOKENS=8 exceeds smem), so compare
    # the admissible pass splits against each other.
    for gate in (0, 1):
        for T, seq_len in ((4, 4), (4, 70), (1, 70)):
            qkv = make_input(8, 2, 256, T, gate, seed=2000 + T + 13 * gate)
            ck, cv = make_ctx(2, 256, seq_len - T, seed=100 + T + gate)
            ref = run((8, 2, 256, 4, gate, 0), qkv, ck, cv, T, seq_len)
            for mt, qp in ((4, 4), (2, 2), (1, 1)):
                if gate == 0 and (mt, qp) != (4, 4):
                    continue  # ungated instantiations: only 4/4 is built
                got = run((8, 2, 256, mt, gate, qp), qkv, ck, cv, T, seq_len)
                base = f"qwen35 gate={gate} T={T} seq={seq_len} pass={qp} arena={mt}"
                for i, what in enumerate(("out", "kv_k", "kv_v")):
                    if not check(rows, f"{base} {what}", got[i], ref[i]):
                        failures.append(f"{base} {what} differs from the MAX_TOKENS=4 single pass")

    # ---------------- C. causal coupling ACROSS passes ---------------------
    # A pure prefill of T tokens with an empty cache: query t must attend keys
    # 0..t, including keys contributed by queries in an EARLIER pass. If the
    # causal bound were computed per-pass (forgetting q_base) the second pass
    # would mask away the first pass's keys and the output would change.
    T, seq_len = 8, 8
    qkv = make_input(4, 1, 128, T, 0, seed=4242)
    ck, cv = make_ctx(1, 128, 0, seed=7)
    ref = run((4, 1, 128, 8, 0, 0), qkv, ck, cv, T, seq_len)
    got = run((4, 1, 128, 4, 0, 4), qkv, ck, cv, T, seq_len)
    # the LAST query row is the discriminating one: it must see all 8 keys
    last_ref, last_got = ref[0][-1], got[0][-1]
    if not check(rows, "cross-pass causal coupling (last query row)", last_got, last_ref):
        failures.append("cross-pass causal coupling broken: the second pass's queries "
                        "do not see the first pass's keys")
    # and it must genuinely differ from an attention truncated to the 2nd pass
    trunc = run((4, 1, 128, 8, 0, 0), qkv[4:].contiguous(),
                torch.zeros(1, 1, 128, dtype=BF, device=DEV),
                torch.zeros(1, 1, 128, dtype=BF, device=DEV), 4, 4)
    if not check(rows, "COUNTERFACTUAL truncated-to-pass-1 attention",
                 got[0][-1], trunc[0][-1], must_match=False):
        failures.append("the truncated counterfactual matched -- the test cannot "
                        "discriminate a per-pass causal bug")

    print(f"\n{'case':<62}{'bit-identical':<15}{'expected':<10}{'max_abs':>12}")
    print("-" * 100)
    for r in rows:
        print(f"{r['case']:<62}{str(r['bit_identical']):<15}"
              f"{str(r['expected_match']):<10}{r['max_abs']:>12.3e}")
    with open(args.out, "w") as f:
        json.dump({"test": "M2-I6 Q-loop equivalence", "rows": rows,
                   "failures": failures, "torch": torch.__version__}, f, indent=2)
    print("\nwrote", args.out)
    if failures:
        print("\nFAILURES:")
        for x in failures:
            print(" -", x)
        return 1
    print("\nall Q-loop equivalence assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
