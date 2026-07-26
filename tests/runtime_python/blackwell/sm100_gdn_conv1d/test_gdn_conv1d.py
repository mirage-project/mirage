"""Kernel-wrapper correctness test for `gdn_conv1d_sm100` (TaskType 234).

Covers the four behaviours the task contract names, plus the layout variants the
dispatch table instantiates:

  * decode           - one new token per slot, FIR over [state | x]
  * chunked prefill  - Q_LEN > 1, state carried across chunk boundaries
  * slot isolation   - interleaved requests keep separate conv states
  * step == 0 reset  - a slot whose request just started ignores stored state
  * inactive slot    - Q_LEN == 0 touches neither the state nor the output

Two PyTorch references are computed (inline, per the test-mode SOP - there is
no shared pytorch_reference.py in this repo):

  ref_hf_semantics : fp32 FIR accumulator, ROUNDED TO BF16, then SiLU, then
                     stored. This is what the kernel implements: HF's
                     `torch_causal_conv1d_update` runs `F.conv1d` in the weight
                     dtype, so its accumulator is rounded before the
                     activation, and v1-architecture.md 1 pins HF as the
                     numeric target wherever it differs from vLLM in low-order
                     bits. `test_gdn_conv1d_oracle.py` shows this order
                     reproduces the real checkpoint's `gdn.conv_out` exactly.
  ref_hf           : the literal `torch_causal_conv1d_update` (channel-last
                     cache, cuDNN conv) - an independent check that the inline
                     reference really is HF's formula.

`ref_vllm_semantics` (SiLU on the raw fp32 accumulator, vLLM's Triton order) is
also available and is what the kernel deliberately does NOT do.

Run:  python test_gdn_conv1d.py
"""

import torch
import torch.nn.functional as F

import runtime_kernel_blackwell_gdn_conv1d as conv_kernel

DEV = "cuda"
BF16 = torch.bfloat16

# bf16 has 8 mantissa bits: the relative rounding unit is 2^-8 = 3.9e-3. The
# test-mode convention is atol/rtol 1e-2; conv is a pure 4-tap FIR so the
# kernel should sit at the bf16 output-rounding floor, not merely inside 1e-2.
TESTMODE_ATOL = 1e-2
TESTMODE_RTOL = 1e-2


# --------------------------------------------------------------------------
# references
# --------------------------------------------------------------------------
def ref_fir(x, w, state, zero_state, round_acc=True):
    """fp32-accumulator FIR + SiLU.

    round_acc=True  (default, and what the kernel does): the accumulator is
                    rounded to bf16 before SiLU - HF's order, the pinned
                    numeric target.
    round_acc=False: SiLU on the raw fp32 accumulator - vLLM's Triton order,
                    kept as the diagnostic the tests report against.

    x     [L, D] bf16 (already sliced to this slot's chunk)
    w     [D, K] bf16
    state [K-1, D] bf16 (ignored when zero_state)
    returns (out [L, D] bf16, new_state [K-1, D] bf16)
    """
    L, D = x.shape
    K = w.shape[1]
    s = torch.zeros(K - 1, D, dtype=torch.float32, device=x.device)
    if not zero_state:
        s = state.float().clone()
    seq = torch.cat([s, x.float()], dim=0)  # [(K-1)+L, D]
    wf = w.float()
    out = torch.zeros(L, D, dtype=torch.float32, device=x.device)
    for t in range(L):
        acc = torch.zeros(D, dtype=torch.float32, device=x.device)
        for j in range(K):
            acc = acc + wf[:, j] * seq[t + j]
        if round_acc:
            acc = acc.to(BF16).float()
        out[t] = acc * torch.sigmoid(acc)
    new_state = seq[-(K - 1):]
    return out.to(BF16), new_state.to(BF16)


def ref_hf_semantics(x, w, state, zero_state):
    return ref_fir(x, w, state, zero_state, round_acc=True)


def ref_vllm_semantics(x, w, state, zero_state):
    return ref_fir(x, w, state, zero_state, round_acc=False)


def ref_hf(x, w, state, zero_state):
    """Verbatim `transformers.torch_causal_conv1d_update` semantics.

    HF keeps its conv cache CHANNEL-LAST ([B, D, state_len]) and runs the conv
    in the weight's dtype, so the accumulator is rounded to bf16 before SiLU.
    """
    L, D = x.shape
    K = w.shape[1]
    st = torch.zeros(1, D, K - 1, dtype=BF16, device=x.device)
    if not zero_state:
        st = state.t().contiguous().unsqueeze(0).clone()  # [1, D, K-1]
    hs = x.t().contiguous().unsqueeze(0)  # [1, D, L]
    new = torch.cat([st, hs], dim=-1).to(w.dtype)
    out = F.conv1d(new, w.unsqueeze(1), None, padding=0, groups=D)
    out = F.silu(out[:, :, -L:]).to(x.dtype)
    new_state = new[:, :, -(K - 1):]
    return out[0].t().contiguous(), new_state[0].t().contiguous()


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def run_kernel(x, w, state, qo_indptr, zero_state, input_stride=None,
               output_stride=None, num_channel_blocks=1):
    """Drive the wrapper. `x` is [num_tokens, D]; a wider input_stride pads the
    row so the strided-input dispatch case is exercised, and
    num_channel_blocks selects the grid.y channel split."""
    num_tokens, D = x.shape
    input_stride = input_stride or D
    output_stride = output_stride or D
    xin = torch.zeros(num_tokens, input_stride, dtype=BF16, device=DEV)
    xin[:, :D] = x
    # Fill the padding with garbage so an out-of-lane read would show up.
    if input_stride > D:
        xin[:, D:] = torch.randn(
            num_tokens, input_stride - D, dtype=BF16, device=DEV
        )
    out = torch.zeros(num_tokens, output_stride, dtype=BF16, device=DEV)
    st = state.clone()
    conv_kernel.gdn_conv1d_sm100(xin, w, st, out, qo_indptr, zero_state,
                                 num_channel_blocks)
    torch.cuda.synchronize()
    return out[:, :D].contiguous(), st


def report(tag, got, ref_a, ref_b=None):
    """ref_a is the pinned HF-order reference; ref_b, when given, is the
    literal `torch_causal_conv1d_update` (a second, independent check that the
    inline reference really is HF's formula)."""
    err = (got.float() - ref_a.float()).abs().max().item()
    n = int((got != ref_a).sum().item())
    line = f"  {tag:<34s} max|kernel-ref_hf| = {err:.3e} ({n} elems differ)"
    if ref_b is not None:
        e2 = (got.float() - ref_b.float()).abs().max().item()
        n2 = int((got != ref_b).sum().item())
        line += f" | vs literal HF fn = {e2:.3e} ({n2} differ)"
    print(line)
    return err


def indptr(lens):
    v = [0]
    for n in lens:
        v.append(v[-1] + n)
    return torch.tensor(v, dtype=torch.int32, device=DEV)


def flags(vals):
    return torch.tensor([1 if v else 0 for v in vals], dtype=torch.uint8,
                        device=DEV)


def make_weight(D, K, gen):
    # Real conv1d weights are small (checkpoint std ~0.034, |max| ~0.41);
    # match that scale so the SiLU operates in a realistic range.
    return (torch.randn(D, K, generator=gen, device=DEV, dtype=torch.float32)
            * 0.05).to(BF16)


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------
def case_decode(D=8192, K=4, seed=0):
    print(f"[decode] D={D} K={K}, 1 slot, Q_LEN=1, carried state")
    gen = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(1, D, generator=gen, device=DEV, dtype=torch.float32).to(BF16)
    w = make_weight(D, K, gen)
    state = torch.randn(1, K - 1, D, generator=gen, device=DEV,
                        dtype=torch.float32).to(BF16)
    got, st = run_kernel(x, w, state, indptr([1]), flags([False]))
    r32, s32 = ref_hf_semantics(x, w, state[0], False)
    rhf, shf = ref_hf(x, w, state[0], False)
    e = report("out", got, r32, rhf)
    es = report("state", st[0], s32, shf)
    torch.testing.assert_close(got, r32, atol=TESTMODE_ATOL, rtol=TESTMODE_RTOL)
    assert torch.equal(st[0], s32), "state must be the exact last K-1 inputs"
    torch.testing.assert_close(got, rhf, atol=TESTMODE_ATOL, rtol=TESTMODE_RTOL)
    return e, es


def case_prefill_zero_state(D=8192, K=4, L=8, seed=1):
    print(f"[prefill] D={D} K={K}, 1 slot, Q_LEN={L}, step==0 (zero state)")
    gen = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(L, D, generator=gen, device=DEV, dtype=torch.float32).to(BF16)
    w = make_weight(D, K, gen)
    # Garbage in the state pool: step==0 must ignore it entirely.
    state = (torch.randn(1, K - 1, D, generator=gen, device=DEV,
                         dtype=torch.float32) * 50).to(BF16)
    got, st = run_kernel(x, w, state, indptr([L]), flags([True]))
    r32, s32 = ref_hf_semantics(x, w, state[0], True)
    rhf, shf = ref_hf(x, w, state[0], True)
    e = report("out", got, r32, rhf)
    es = report("state", st[0], s32, shf)
    torch.testing.assert_close(got, r32, atol=TESTMODE_ATOL, rtol=TESTMODE_RTOL)
    assert torch.equal(st[0], s32)
    torch.testing.assert_close(got, rhf, atol=TESTMODE_ATOL, rtol=TESTMODE_RTOL)

    # A zero-state prefill must equal HF's left-zero-padded whole-chunk conv,
    # which is a structurally different formula (F.conv1d with padding=K-1).
    padded = F.silu(
        F.conv1d(x.t().contiguous().unsqueeze(0), w.unsqueeze(1), None,
                 padding=K - 1, groups=D)[:, :, :L]
    )[0].t().contiguous()
    e_pad = (got.float() - padded.float()).abs().max().item()
    print(f"  {'out vs HF padded-prefill conv':<34s} max = {e_pad:.3e}")
    torch.testing.assert_close(got, padded, atol=TESTMODE_ATOL,
                               rtol=TESTMODE_RTOL)
    return e, es


def case_chunked_prefill(D=4096, K=4, chunks=(5, 4, 1), seed=2):
    print(f"[chunked] D={D} K={K}, one request split into chunks {chunks}")
    gen = torch.Generator(device=DEV).manual_seed(seed)
    total = sum(chunks)
    x = torch.randn(total, D, generator=gen, device=DEV,
                    dtype=torch.float32).to(BF16)
    w = make_weight(D, K, gen)
    state = torch.zeros(1, K - 1, D, dtype=BF16, device=DEV)

    # Feed the chunks one iteration at a time, carrying the state pool.
    outs = []
    st = state.clone()
    zero_first = True
    off = 0
    for n in chunks:
        o, st = run_kernel(x[off:off + n], w, st, indptr([n]),
                           flags([zero_first]))
        outs.append(o)
        zero_first = False
        off += n
    got = torch.cat(outs, dim=0)

    # Ground truth: the same tokens in ONE pass with a zero initial state.
    r32, s32 = ref_hf_semantics(x, w, state[0], True)
    e = report("out (chunked vs single pass)", got, r32)
    torch.testing.assert_close(got, r32, atol=TESTMODE_ATOL, rtol=TESTMODE_RTOL)
    assert torch.equal(st[0], s32), "final state must match the single-pass one"

    # Chunking must be EXACT, not merely close: the FIR is causal, so replaying
    # the same tokens in different chunk sizes may not change a single bit.
    one_shot, one_state = run_kernel(x, w, state, indptr([total]), flags([True]))
    assert torch.equal(got, one_shot), "chunked output differs bit-wise"
    assert torch.equal(st, one_state), "chunked state differs bit-wise"
    print("  chunked == single-pass, bit-identical")
    return e, 0.0


def case_slot_isolation(D=4096, K=4, seed=3):
    print(f"[slots] D={D} K={K}, 2 interleaved requests, distinct states")
    gen = torch.Generator(device=DEV).manual_seed(seed)
    lens = [3, 1]  # slot 0 is prefilling a 3-token chunk, slot 1 is decoding
    total = sum(lens)
    x = torch.randn(total, D, generator=gen, device=DEV,
                    dtype=torch.float32).to(BF16)
    w = make_weight(D, K, gen)
    state = torch.randn(2, K - 1, D, generator=gen, device=DEV,
                        dtype=torch.float32).to(BF16)
    zs = [False, False]
    got, st = run_kernel(x, w, state, indptr(lens), flags(zs))

    max_e = 0.0
    off = 0
    for slot, n in enumerate(lens):
        r32, s32 = ref_hf_semantics(x[off:off + n], w, state[slot], zs[slot])
        e = report(f"slot {slot} out", got[off:off + n], r32)
        max_e = max(max_e, e)
        torch.testing.assert_close(got[off:off + n], r32, atol=TESTMODE_ATOL,
                                   rtol=TESTMODE_RTOL)
        assert torch.equal(st[slot], s32), f"slot {slot} state wrong"
        off += n

    # Cross-check: running each slot ALONE must give the same bits, i.e. the
    # slots never read or write each other's state.
    off = 0
    for slot, n in enumerate(lens):
        solo, solo_st = run_kernel(x[off:off + n], w, state[slot:slot + 1],
                                   indptr([n]), flags([zs[slot]]))
        assert torch.equal(solo, got[off:off + n]), f"slot {slot} not isolated"
        assert torch.equal(solo_st[0], st[slot]), f"slot {slot} state leaked"
        off += n
    print("  per-slot results bit-identical to solo runs")
    return max_e, 0.0


def case_step0_reset(D=512, K=4, seed=4):
    print(f"[reset] D={D} K={K}, slot reuse after request completion")
    gen = torch.Generator(device=DEV).manual_seed(seed)
    w = make_weight(D, K, gen)

    # Request A runs to completion in the slot and leaves state behind.
    xa = torch.randn(6, D, generator=gen, device=DEV,
                     dtype=torch.float32).to(BF16)
    state = torch.zeros(1, K - 1, D, dtype=BF16, device=DEV)
    _, state_after_a = run_kernel(xa, w, state, indptr([6]), flags([True]))
    assert state_after_a.abs().sum().item() > 0, "request A left no state"

    # Request B takes the same slot. Its step restarts at 0, so the kernel must
    # behave exactly as if the pool had been zeroed.
    xb = torch.randn(4, D, generator=gen, device=DEV,
                     dtype=torch.float32).to(BF16)
    reused, st_reused = run_kernel(xb, w, state_after_a, indptr([4]),
                                   flags([True]))
    zeroed, st_zeroed = run_kernel(xb, w, torch.zeros_like(state_after_a),
                                   indptr([4]), flags([True]))
    assert torch.equal(reused, zeroed), "step==0 did not ignore stale state"
    assert torch.equal(st_reused, st_zeroed)

    # ... and it must NOT equal the run that carries A's state (otherwise the
    # test would pass for a kernel that ignores zero_state altogether).
    carried, _ = run_kernel(xb, w, state_after_a, indptr([4]), flags([False]))
    assert not torch.equal(carried, reused), (
        "carried-state and reset runs are identical - the predicate is dead")
    diff = (carried.float() - reused.float()).abs().max().item()
    print(f"  reset == zero-pool (bit-identical); carried differs by {diff:.3e}")
    return 0.0, 0.0


def case_inactive_slot(D=512, K=4, seed=5):
    print(f"[inactive] D={D} K={K}, Q_LEN==0 slot must be untouched")
    gen = torch.Generator(device=DEV).manual_seed(seed)
    w = make_weight(D, K, gen)
    lens = [2, 0, 1]  # slot 1 is parked
    total = sum(lens)
    x = torch.randn(total, D, generator=gen, device=DEV,
                    dtype=torch.float32).to(BF16)
    state = torch.randn(3, K - 1, D, generator=gen, device=DEV,
                        dtype=torch.float32).to(BF16)
    got, st = run_kernel(x, w, state, indptr(lens), flags([False] * 3))
    assert torch.equal(st[1], state[1]), "parked slot's state was modified"

    off = 0
    for slot, n in enumerate(lens):
        if n == 0:
            continue
        r32, _ = ref_hf_semantics(x[off:off + n], w, state[slot], False)
        torch.testing.assert_close(got[off:off + n], r32, atol=TESTMODE_ATOL,
                                   rtol=TESTMODE_RTOL)
        off += n
    print("  parked slot state preserved bit-for-bit; active slots correct")
    return 0.0, 0.0


def case_strided_input(D=8192, K=4, seed=6):
    print(f"[strided] D={D} K={K}, input row stride 12288 (fused qkvz layout)")
    gen = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(3, D, generator=gen, device=DEV,
                    dtype=torch.float32).to(BF16)
    w = make_weight(D, K, gen)
    state = torch.randn(1, K - 1, D, generator=gen, device=DEV,
                        dtype=torch.float32).to(BF16)
    got, st = run_kernel(x, w, state, indptr([3]), flags([False]),
                         input_stride=12288)
    r32, s32 = ref_hf_semantics(x, w, state[0], False)
    e = report("out", got, r32)
    torch.testing.assert_close(got, r32, atol=TESTMODE_ATOL, rtol=TESTMODE_RTOL)
    assert torch.equal(st[0], s32)
    return e, 0.0


def case_kernel_widths(D=128, seed=7):
    print(f"[widths] D={D}, KERNEL_SIZE in (2, 4, 8)")
    max_e = 0.0
    for K in (2, 4, 8):
        gen = torch.Generator(device=DEV).manual_seed(seed + K)
        x = torch.randn(5, D, generator=gen, device=DEV,
                        dtype=torch.float32).to(BF16)
        w = make_weight(D, K, gen)
        state = torch.randn(1, K - 1, D, generator=gen, device=DEV,
                            dtype=torch.float32).to(BF16)
        got, st = run_kernel(x, w, state, indptr([5]), flags([False]))
        r32, s32 = ref_hf_semantics(x, w, state[0], False)
        rhf, _ = ref_hf(x, w, state[0], False)
        e = report(f"K={K} out", got, r32, rhf)
        max_e = max(max_e, e)
        torch.testing.assert_close(got, r32, atol=TESTMODE_ATOL,
                                   rtol=TESTMODE_RTOL)
        assert torch.equal(st[0], s32)
    return max_e, 0.0


def case_channel_blocks(D=8192, K=4, seed=8):
    """The grid.y channel split must be a pure parallelisation: every block
    count has to produce bit-identical output AND state."""
    print(f"[channels] D={D} K={K}, grid.y in (1, 8, 32), 2 slots")
    gen = torch.Generator(device=DEV).manual_seed(seed)
    lens = [7, 2]
    total = sum(lens)
    x = torch.randn(total, D, generator=gen, device=DEV,
                    dtype=torch.float32).to(BF16)
    w = make_weight(D, K, gen)
    state = torch.randn(2, K - 1, D, generator=gen, device=DEV,
                        dtype=torch.float32).to(BF16)
    zs = flags([True, False])

    base_out, base_st = run_kernel(x, w, state, indptr(lens), zs,
                                   num_channel_blocks=1)
    for ncb in (8, 32):
        got, st = run_kernel(x, w, state, indptr(lens), zs,
                             num_channel_blocks=ncb)
        assert torch.equal(got, base_out), (
            f"grid.y={ncb} output differs from the single-block run")
        assert torch.equal(st, base_st), (
            f"grid.y={ncb} state differs from the single-block run")
        print(f"  grid.y={ncb:<3d} bit-identical to grid.y=1")

    # And still correct against the reference, per slot.
    off = 0
    for slot, n in enumerate(lens):
        ref, ref_st = ref_hf_semantics(x[off:off + n], w, state[slot],
                                       slot == 0)
        torch.testing.assert_close(base_out[off:off + n], ref,
                                   atol=TESTMODE_ATOL, rtol=TESTMODE_RTOL)
        assert torch.equal(base_st[slot], ref_st)
        off += n
    return 0.0, 0.0


def main():
    assert torch.cuda.is_available(), "this test needs a Blackwell GPU"
    print(f"device: {torch.cuda.get_device_name(0)}")
    cases = [
        case_decode,
        case_prefill_zero_state,
        case_chunked_prefill,
        case_slot_isolation,
        case_step0_reset,
        case_inactive_slot,
        case_strided_input,
        case_kernel_widths,
        case_channel_blocks,
    ]
    worst = 0.0
    for fn in cases:
        e, _ = fn()
        worst = max(worst, e)
        print()
    print(f"ALL PASS - worst max|kernel - ref_hf_semantics| across cases: "
          f"{worst:.3e} (test-mode budget {TESTMODE_ATOL:.0e})")


if __name__ == "__main__":
    main()
