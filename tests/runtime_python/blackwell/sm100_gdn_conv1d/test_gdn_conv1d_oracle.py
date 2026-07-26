"""`gdn_conv1d_sm100` vs the M2-I3 HF oracle dumps (probe P6).

The oracle is `transformers` running the real Qwen/Qwen3.5-35B-A3B-FP8
checkpoint (workspace/demo/qwen3_5/oracle/README.md). It dumps, for GDN layer 0:

    gdn.conv_in            [B, 8192, T]   pre-conv fused [q|k|v]
    gdn.conv_out           [B, 8192, T]   post-conv, post-SiLU
    gdn.conv_state_before  [B, 8192, 4]   decode only
    gdn.conv_state_after   [B, 8192, 4]
    gdn.__weight.conv1d_weight  [8192, 1, 4]

Two layout facts are ASSERTED from the dumps rather than assumed:

  1. HF's conv cache is channel-last with state_len == kernel_size == 4 and
     holds the last four inputs, whereas vLLM / MPK keep kernel_size-1 == 3
     (vllm-graph.md 2.1.5). So MPK's pool is HF's `[..., 1:4]` transposed. The
     test checks that HF's window really is the trailing inputs before using it.
  2. The prefill dump came from HF's zero-left-padded whole-chunk conv, i.e.
     exactly MPK's step == 0 predicate.

Run (on the B200 box, from this directory):
    python test_gdn_conv1d_oracle.py
    python test_gdn_conv1d_oracle.py --dump-dir /path/to/dumps
"""

import argparse
import os

import torch

import runtime_kernel_blackwell_gdn_conv1d as conv_kernel

DEV = "cuda"
BF16 = torch.bfloat16
DEFAULT_DUMPS = os.path.expanduser("~/mpk-qwen35/oracle-work/dumps")

# Test-mode bf16 convention (test-mode SKILL.md). Reported separately from the
# actual observed error, which for a 4-tap FIR should be at the bf16 rounding
# floor rather than anywhere near this budget.
TESTMODE_ATOL = 1e-2
TESTMODE_RTOL = 1e-2


def load(dump_dir, mode, name):
    path = os.path.join(dump_dir, mode, "tensors", f"{name}.pt")
    if not os.path.exists(path):
        raise SystemExit(
            f"missing oracle tensor {path}\n"
            "The full dumps stay on B200 (see workspace/demo/qwen3_5/oracle/"
            "README.md, 'What's committed here vs what stays on B200'); "
            "regenerate them with ref_dump.py or pass --dump-dir."
        )
    return torch.load(path, map_location=DEV, weights_only=True)


def run_kernel(x, w, state, zero_state, num_channel_blocks=32):
    """x [L, D] bf16, w [D, K] bf16, state [K-1, D] bf16 -> (out, new_state).

    Defaults to the 32-channel-block grid (8192/256, the same split vLLM's
    Triton kernel uses) so the oracle check exercises the production shape,
    not a degenerate single-block one."""
    L, D = x.shape
    out = torch.zeros(L, D, dtype=BF16, device=DEV)
    st = state.clone().unsqueeze(0).contiguous()
    qo = torch.tensor([0, L], dtype=torch.int32, device=DEV)
    zs = torch.tensor([1 if zero_state else 0], dtype=torch.uint8, device=DEV)
    conv_kernel.gdn_conv1d_sm100(x.contiguous(), w.contiguous(), st, out, qo,
                                 zs, num_channel_blocks)
    torch.cuda.synchronize()
    return out, st[0]


def ref_fir(x, w, state, zero_state, round_acc):
    """4-tap causal FIR with an fp32 accumulator.

    round_acc=False -> SiLU on the fp32 accumulator, ONE rounding on the store.
                       This is vLLM's Triton kernel (causal_conv1d.py:943).
    round_acc=True  -> the accumulator is rounded to bf16 BEFORE the SiLU.
                       This is HF's `torch_causal_conv1d_update`, whose
                       `F.conv1d` runs in the weight's dtype and hands a bf16
                       tensor to `F.silu`.

    v1-architecture.md 1 pins HF as the numeric target wherever vLLM and HF
    differ in low-order bits, so `round_acc=True` is what the kernel must do.
    """
    L, D = x.shape
    K = w.shape[1]
    s = (torch.zeros(K - 1, D, dtype=torch.float32, device=x.device)
         if zero_state else state.float().clone())
    seq = torch.cat([s, x.float()], dim=0)
    wf = w.float()
    out = torch.zeros(L, D, dtype=torch.float32, device=x.device)
    for t in range(L):
        acc = torch.zeros(D, dtype=torch.float32, device=x.device)
        for j in range(K):
            acc = acc + wf[:, j] * seq[t + j]
        if round_acc:
            acc = acc.to(BF16).float()
        out[t] = acc * torch.sigmoid(acc)
    return out.to(BF16), seq[-(K - 1):].to(BF16)


def ref_hf(x, w, state, zero_state):
    """`transformers.torch_causal_conv1d_update` verbatim (channel-last cache,
    conv + activation in the weight dtype)."""
    L, D = x.shape
    K = w.shape[1]
    st = (torch.zeros(1, D, K - 1, dtype=BF16, device=x.device) if zero_state
          else state.t().contiguous().unsqueeze(0).clone())
    new = torch.cat([st, x.t().contiguous().unsqueeze(0)], dim=-1).to(w.dtype)
    out = torch.nn.functional.conv1d(new, w.unsqueeze(1), None, padding=0,
                                     groups=D)
    out = torch.nn.functional.silu(out[:, :, -L:]).to(x.dtype)
    return out[0].t().contiguous(), new[0, :, -(K - 1):].t().contiguous()


def stats(tag, got, ref):
    a, b = got.float(), ref.float()
    max_abs = (a - b).abs().max().item()
    frob = (a - b).norm().item() / max(b.norm().item(), 1e-30)
    # bf16 has 8 mantissa bits; one ULP at |v| is 2^(floor(log2|v|)-7).
    ulp = torch.where(b == 0, torch.full_like(b, 2.0 ** -133),
                      2.0 ** (torch.floor(torch.log2(b.abs().clamp(min=1e-38)))
                              - 7))
    max_ulp = ((a - b).abs() / ulp).max().item()
    n_diff = int((a != b).sum().item())
    print(f"  {tag:<30s} max_abs={max_abs:.3e}  frob_rel={frob:.3e}  "
          f"max_bf16_ulp={max_ulp:.2f}  differing={n_diff}/{b.numel()}")
    return max_abs


def assert_bit_exact(got, ref, what):
    """The kernel implements HF's exact op order (fp32 FIR -> round to bf16 ->
    SiLU -> store), so it must reproduce the oracle bit for bit, not merely
    within the bf16 tolerance.

    A non-zero count here means the fp32 FIR itself diverged (e.g. a toolchain
    change to FMA contraction), which is worth investigating rather than
    absorbing into a tolerance - the tolerance check runs too, right after.
    """
    n = int((got != ref).sum().item())
    assert n == 0, (
        f"{what}: {n}/{ref.numel()} elements differ from the HF oracle. The "
        "kernel is supposed to be bit-exact here; check the fp32 accumulation "
        "order before relaxing this."
    )
    print(f"  {what}: BIT-EXACT vs the HF oracle ({ref.numel()} elements)")


def assert_within_budget(got, ref):
    """torch.testing.assert_close's budget is atol + rtol*|ref|; report the
    worst RATIO to that budget rather than a bare max_abs, which is meaningless
    on a tensor whose values span two orders of magnitude."""
    a, b = got.float(), ref.float()
    budget = TESTMODE_ATOL + TESTMODE_RTOL * b.abs()
    ratio = ((a - b).abs() / budget).max().item()
    print(f"  {'worst |diff| / (atol+rtol*|ref|)':<30s} = {ratio:.3f}  "
          f"(pass iff <= 1)")
    torch.testing.assert_close(got, ref, atol=TESTMODE_ATOL, rtol=TESTMODE_RTOL)
    return ratio


def check_layout(dump_dir):
    """Assert the HF conv-cache layout facts this test relies on."""
    print("[layout] verifying HF conv-state layout from the dumps themselves")
    pre_in = load(dump_dir, "prefill", "gdn.conv_in")[0]        # [D, T]
    pre_after = load(dump_dir, "prefill", "gdn.conv_state_after")[0]  # [D, 4]
    dec_before = load(dump_dir, "decode", "gdn.conv_state_before")[0]
    dec_in = load(dump_dir, "decode", "gdn.conv_in")[0]
    dec_after = load(dump_dir, "decode", "gdn.conv_state_after")[0]
    K = pre_after.shape[-1]

    assert torch.equal(pre_after, pre_in[:, -K:]), (
        "prefill conv_state_after is not the last kernel_size inputs")
    print(f"  prefill state_after == conv_in[:, -{K}:]                    OK")
    assert torch.equal(dec_before, pre_after), (
        "decode conv_state_before != prefill conv_state_after")
    print("  decode state_before  == prefill state_after                OK")
    expect = torch.cat([pre_in, dec_in], dim=-1)[:, -K:]
    assert torch.equal(dec_after, expect), (
        "decode conv_state_after is not the last kernel_size inputs")
    print(f"  decode  state_after  == [prefill|decode] inputs[:, -{K}:]    OK")
    print("  => HF keeps a width-4 rolling window; MPK's 3-wide pool is "
          "its [..., 1:] slice")


def case_prefill(dump_dir):
    print("[oracle prefill] 8-token chunk, step == 0 (zero state)")
    w = load(dump_dir, "prefill", "gdn.__weight.conv1d_weight").squeeze(1)
    x = load(dump_dir, "prefill", "gdn.conv_in")[0].t().contiguous()
    y = load(dump_dir, "prefill", "gdn.conv_out")[0].t().contiguous()
    st_after = load(dump_dir, "prefill", "gdn.conv_state_after")[0]
    K = w.shape[1]
    D = w.shape[0]
    print(f"  shapes: x={tuple(x.shape)} w={tuple(w.shape)} tokens={x.shape[0]}")

    # Garbage in the pool: the step==0 predicate must ignore it.
    garbage = (torch.randn(K - 1, D, device=DEV) * 50).to(BF16)
    got, got_st = run_kernel(x, w, garbage, zero_state=True)
    r_hf, r_hf_st = ref_fir(x, w, garbage, True, round_acc=True)
    r_v, _ = ref_fir(x, w, garbage, True, round_acc=False)
    r_lit, _ = ref_hf(x, w, garbage, True)

    e_or = stats("kernel vs HF oracle", got, y)
    stats("ref_hf-semantics vs oracle", r_hf, y)
    stats("ref_vllm-semantics vs oracle", r_v, y)
    stats("literal HF fn vs oracle", r_lit, y)
    stats("kernel vs ref_hf-semantics", got, r_hf)
    assert_within_budget(got, y)
    assert_bit_exact(got, y, "prefill conv_out")

    want_st = st_after[:, 1:].t().contiguous()
    assert torch.equal(got_st, want_st), "final conv state != oracle's"
    assert torch.equal(got_st, r_hf_st)
    print("  final state bit-identical to the oracle's trailing 3 inputs")
    return e_or


def case_decode(dump_dir):
    print("[oracle decode] 1 token, carried state")
    w = load(dump_dir, "decode", "gdn.__weight.conv1d_weight").squeeze(1)
    x = load(dump_dir, "decode", "gdn.conv_in")[0].t().contiguous()
    y = load(dump_dir, "decode", "gdn.conv_out")[0].t().contiguous()
    st_before = load(dump_dir, "decode", "gdn.conv_state_before")[0]
    st_after = load(dump_dir, "decode", "gdn.conv_state_after")[0]
    print(f"  shapes: x={tuple(x.shape)} w={tuple(w.shape)}")

    state = st_before[:, 1:].t().contiguous()  # HF's 4-wide -> MPK's 3-wide
    got, got_st = run_kernel(x, w, state, zero_state=False)
    r_hf, r_hf_st = ref_fir(x, w, state, False, round_acc=True)
    r_v, _ = ref_fir(x, w, state, False, round_acc=False)
    r_lit, _ = ref_hf(x, w, state, False)

    e_or = stats("kernel vs HF oracle", got, y)
    stats("ref_hf-semantics vs oracle", r_hf, y)
    stats("ref_vllm-semantics vs oracle", r_v, y)
    stats("literal HF fn vs oracle", r_lit, y)
    stats("kernel vs ref_hf-semantics", got, r_hf)
    assert_within_budget(got, y)
    assert_bit_exact(got, y, "decode conv_out")

    want_st = st_after[:, 1:].t().contiguous()
    assert torch.equal(got_st, want_st), "updated conv state != oracle's"
    assert torch.equal(got_st, r_hf_st)
    print("  updated state bit-identical to the oracle's")
    return e_or


def case_prefill_then_decode(dump_dir):
    """End-to-end: replay the oracle's prefill and its decode step through the
    SAME state pool, exactly as the runtime would across two iterations."""
    print("[oracle prefill->decode] one pool carried across two iterations")
    w = load(dump_dir, "prefill", "gdn.__weight.conv1d_weight").squeeze(1)
    xp = load(dump_dir, "prefill", "gdn.conv_in")[0].t().contiguous()
    xd = load(dump_dir, "decode", "gdn.conv_in")[0].t().contiguous()
    yd = load(dump_dir, "decode", "gdn.conv_out")[0].t().contiguous()
    K, D = w.shape[1], w.shape[0]

    _, st = run_kernel(xp, w, torch.zeros(K - 1, D, dtype=BF16, device=DEV),
                       zero_state=True)
    got, _ = run_kernel(xd, w, st, zero_state=False)
    e = stats("decode out vs HF oracle", got, yd)
    assert_within_budget(got, yd)
    assert_bit_exact(got, yd, "prefill->decode conv_out")
    print("  the state the kernel produced in iteration 0 feeds iteration 1 "
          "with no host round trip")
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=DEFAULT_DUMPS)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "this test needs a Blackwell GPU"
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"dumps:  {args.dump_dir}\n")

    check_layout(args.dump_dir)
    print()
    worst = max(
        case_prefill(args.dump_dir),
        case_decode(args.dump_dir),
        case_prefill_then_decode(args.dump_dir),
    )
    print(f"\nORACLE PASS - worst max|kernel - HF oracle| = {worst:.3e} "
          f"(absolute; see the per-case ratio-to-budget lines for the "
          f"atol/rtol verdict)")


if __name__ == "__main__":
    main()
