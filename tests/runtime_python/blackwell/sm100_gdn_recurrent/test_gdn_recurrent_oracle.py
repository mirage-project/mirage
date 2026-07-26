"""`gdn_recurrent_sm100` vs the M2-I3 HF oracle dumps (probe P6).

The oracle is `transformers` running the real Qwen/Qwen3.5-35B-A3B-FP8
checkpoint (workspace/demo/qwen3_5/oracle/README.md). It dumps, for GDN layer 0:

    gdn.conv_out           [B, 8192, T]      post-conv [q|k|v], the task's input
    gdn.b_proj_out         [B, T, 32]        \\ the task packs these as `ba`
    gdn.a_proj_out         [B, T, 32]        /
    gdn.z_proj_out         [B, T, 4096]      the output gate
    gdn.beta               [B, T, 32]        sigmoid(b)          bf16
    gdn.decay_g            [B, T, 32]        -exp(A_log)*sp(...) fp32
    gdn.core_state_before  [B, 32, 128, 128] fp32 (decode only)
    gdn.core_state_after   [B, 32, 128, 128] fp32
    gdn.core_attn_out      [B, T, 32, 128]   bf16  (o, pre-epilogue)
    gdn.gated_norm_out     [B*T*32, 128]     bf16  (y, the task's output)
    gdn.__weight.{A_log,dt_bias,norm_weight}

THE NUMERIC TARGET IS HF's ROUNDING ORDER, not vLLM's and not the architecture
doc's. Three decisions are genuine discriminators; each is checked here against
its counterfactual so a future edit cannot silently regress one:

  D1  q/k L2 norm runs in BF16 (HF applies `l2norm` to the bf16 tensors before
      the fp32 upcast). vLLM's kernel normalizes in fp32.
  D2  `o` is rounded to bf16 before the gated norm sees it.
  D3  inside the gated norm the NORMALIZED value is rounded to bf16 before it is
      multiplied by the fp32 norm weight. The architecture doc (3.2) writes the
      whole epilogue as one fp32 expression.

STATE LAYOUT: HF stores the recurrent state `[k_head_dim, v_head_dim]`; MPK
stores it `[head_v_dim, head_k_dim]` (see the kernel header for why). The
transpose is applied here, and asserted to be the right one by the fact that the
decode check reproduces `core_attn_out` bit-exactly from `core_state_before`.

Run (on the B200 box, from this directory):
    python test_gdn_recurrent_oracle.py
    python test_gdn_recurrent_oracle.py --dump-dir /path/to/dumps
"""

import argparse
import os

import torch
import torch.nn.functional as F

import runtime_kernel_blackwell_gdn_recurrent as gdn

DEV = "cuda"
BF16 = torch.bfloat16
DEFAULT_DUMPS = os.path.expanduser("~/mpk-qwen35/oracle-work/dumps")

NUM_V_HEADS = 32
NUM_K_HEADS = 16
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM      # 2048
VAL_DIM = NUM_V_HEADS * HEAD_V_DIM      # 4096
QKV_STRIDE = 2 * KEY_DIM + VAL_DIM      # 8192
EPS = 1e-6

FAILURES = []
TABLE = []


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


def check(tag, got, ref, expect_exact=True, note=""):
    """Compare and record. Bit-exactness is the assertion where it is the
    target; elsewhere the achieved delta is reported, never silently widened."""
    assert got.shape == ref.shape, f"{tag}: shape {got.shape} vs {ref.shape}"
    ne = int((got != ref).sum()) if got.dtype == ref.dtype else -1
    d = (got.float() - ref.float()).abs()
    maxabs = d.max().item()
    rms = ref.float().pow(2).mean().sqrt().item()
    relrms = maxabs / rms if rms > 0 else 0.0
    ulp = ""
    if got.dtype == BF16:
        u = (got.view(torch.int16).int() - ref.view(torch.int16).int()).abs()
        ulp = f" maxulp={u.max().item()}"
    status = "BIT-EXACT" if ne == 0 else ("ok" if not expect_exact else "FAIL")
    if expect_exact and ne != 0:
        FAILURES.append(tag)
    line = (f"  {status:9s} {tag:46s} bitdiff={ne}/{ref.numel()} "
            f"maxabs={maxabs:.3e} maxabs/rms={relrms:.2e}{ulp} {note}")
    print(line)
    TABLE.append((tag, status, ne, ref.numel(), maxabs, relrms, note))
    return ne


# ------------------------------------------------------------------ references
# Inline torch references (add-mpk-task convention: no shared
# pytorch_reference.py). Each is a transcription of the HF formula it emulates,
# NOT a copy of the kernel, so agreement is evidence rather than tautology.

def l2norm_bf16(x, eps=EPS):
    """transformers' `l2norm` on a bf16 tensor: every op round-trips bf16."""
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def l2norm_fp32(x, eps=EPS):
    """Counterfactual D1: vLLM's Triton kernel normalizes in fp32."""
    xf = x.float()
    return xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + eps)


def gated_rmsnorm(o, z, w, eps=EPS, round_xhat=True):
    """`Qwen3_5MoeRMSNormGated.forward`. round_xhat=False is counterfactual D3
    (the architecture doc's single fp32 expression)."""
    xf = o.float()
    xh = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if round_xhat:
        xh = xh.to(BF16).float()
    y = w.float() * xh
    return (y * F.silu(z.float())).to(BF16)


def dot_torch(mat, vec):
    """Contract S[v][k] with vec[k] using torch's own reduction association -
    the one HF's `.sum(dim=-2)` used when it produced the dumps."""
    return (mat * vec.unsqueeze(0)).sum(dim=-1)


def fma32(a, b, c):
    """Single-rounding fp32 `a*b + c`, i.e. what nvcc emits for `c += a*b`.

    torch has no fp32 FMA, so the product is formed in float64 - where a 24x24
    bit product is EXACT - and the sum is rounded back to fp32 once. Verified
    against a `-fmad=false` build of the same kernel: with FMA disabled the
    kernel matches a plain-multiply-then-add reference bit-for-bit, and with FMA
    enabled it matches this emulation bit-for-bit."""
    return (a.double() * b.double() + c.double()).float()


def dot_kernel_order(mat, vec, fma=True):
    """The kernel's fp32 association, exactly: one warp per v-row, lane l
    accumulates k = l, l+32, l+64, l+96 with FMAs in that order, then a shuffle
    tree combines the 32 partials (acc[j] += acc[j^m], m = 16, 8, 4, 2, 1).

    This exists to make the fp32 residual FALSIFIABLE. The kernel's fp32 state
    is NOT bit-exact against HF, and there are exactly two reasons: FMA
    contraction and this association order. A reference carrying BOTH must
    therefore agree bit-for-bit, state included - and it does, which is what
    proves there is no third, unexplained source of deviation."""
    V, K = mat.shape
    m3 = mat.reshape(V, K // 32, 32)             # [v, u, lane], k = lane + 32u
    v3 = vec.reshape(K // 32, 32)
    if fma:
        acc = torch.zeros(V, 32, dtype=torch.float32, device=mat.device)
        for u in range(K // 32):
            acc = fma32(m3[:, u], v3[u].unsqueeze(0).expand(V, 32), acc)
    else:
        prod = mat * vec.unsqueeze(0)
        part = prod.reshape(V, K // 32, 32)
        acc = part[:, 0]
        for u in range(1, part.shape[1]):
            acc = acc + part[:, u]
    n = 32
    while n > 1:
        n //= 2
        acc = acc[:, :n] + acc[:, n:2 * n]
    return acc[:, 0]


def ref_sequential(qkv, ba, z, A_log, dt_bias, norm_w, S0, l2fn=l2norm_bf16,
                   round_xhat=True, dot=dot_torch, fma=False):
    """The task's full chain, token-sequential, in plain torch.

    S0 / the returned state are in MPK's [head_v_dim, head_k_dim] layout.
    Returns (o_bf16 [T, VAL_DIM], y_bf16 [T, VAL_DIM], S [Hv, Dv, Dk]).
    """
    T = qkv.shape[0]
    S = S0.clone().float()
    o_all = torch.zeros(T, VAL_DIM, dtype=BF16, device=DEV)
    y_all = torch.zeros(T, VAL_DIM, dtype=BF16, device=DEV)
    b = ba[:, :NUM_V_HEADS]
    a = ba[:, NUM_V_HEADS:2 * NUM_V_HEADS]
    beta_all = b.sigmoid().float()
    g_all = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    scale = 1.0 / (HEAD_K_DIM ** 0.5)
    for t in range(T):
        o_t = torch.zeros(NUM_V_HEADS, HEAD_V_DIM, dtype=torch.float32,
                          device=DEV)
        for hv in range(NUM_V_HEADS):
            ih = hv // (NUM_V_HEADS // NUM_K_HEADS)
            q = qkv[t, ih * HEAD_K_DIM:(ih + 1) * HEAD_K_DIM]
            k = qkv[t, KEY_DIM + ih * HEAD_K_DIM:
                       KEY_DIM + (ih + 1) * HEAD_K_DIM]
            v = qkv[t, 2 * KEY_DIM + hv * HEAD_V_DIM:
                       2 * KEY_DIM + (hv + 1) * HEAD_V_DIM].float()
            qn = l2fn(q).float() * scale
            kn = l2fn(k).float()
            # S is [v][k] here; HF's kv_mem/readout contract over k.
            Sh = S[hv] * g_all[t, hv].exp()
            kv = dot(Sh, kn)
            delta = (v - kv) * beta_all[t, hv]
            # The rank-1 update is `row[c] + s_k[c]*delta` - also an FMA.
            if fma:
                Sh = fma32(kn.unsqueeze(0).expand_as(Sh),
                           delta.unsqueeze(-1).expand_as(Sh), Sh)
            else:
                Sh = Sh + delta.unsqueeze(-1) * kn.unsqueeze(0)
            o_t[hv] = dot(Sh, qn)
            S[hv] = Sh
        # D2: `o` is rounded to bf16 before the epilogue sees it.
        o_bf = o_t.to(BF16)
        o_all[t] = o_bf.reshape(-1)
        y_all[t] = gated_rmsnorm(o_bf,
                                 z[t].reshape(NUM_V_HEADS, HEAD_V_DIM),
                                 norm_w, round_xhat=round_xhat).reshape(-1)
    return o_all, y_all, S


def run_kernel(qkv, ba, alog_dtbias, S0, z, norm_w, zero_state,
               num_slots=1, qo=None):
    """qkv [T,QKV_STRIDE], ba [T,64], z [T,VAL_DIM], S0 [slots,Hv,Dv,Dk]."""
    T = qkv.shape[0]
    out = torch.zeros(T, VAL_DIM, dtype=BF16, device=DEV)
    o_dbg = torch.zeros(T, VAL_DIM, dtype=BF16, device=DEV)
    st = S0.clone().contiguous()
    if qo is None:
        qo = torch.tensor([0, T], dtype=torch.int32, device=DEV)
    zs = torch.tensor(zero_state, dtype=torch.uint8, device=DEV)
    gdn.gdn_recurrent_sm100(qkv.contiguous(), ba.contiguous(),
                            alog_dtbias.contiguous(), st, z.contiguous(),
                            norm_w.contiguous(), out, qo, zs,
                            NUM_K_HEADS, o_dbg)
    torch.cuda.synchronize()
    return out, o_dbg, st


def build_inputs(dump_dir, mode):
    conv_out = load(dump_dir, mode, "gdn.conv_out")          # [1, 8192, T]
    qkv = conv_out[0].transpose(0, 1).contiguous()           # [T, 8192]
    b = load(dump_dir, mode, "gdn.b_proj_out")[0]            # [T, 32]
    a = load(dump_dir, mode, "gdn.a_proj_out")[0]
    ba = torch.cat([b, a], dim=-1).contiguous()              # [T, 64]
    z = load(dump_dir, mode, "gdn.z_proj_out")[0].contiguous()   # [T, 4096]
    A_log = load(dump_dir, mode, "gdn.__weight.A_log")       # [32] f32
    dt_bias = load(dump_dir, mode, "gdn.__weight.dt_bias")   # [32] bf16
    norm_w = load(dump_dir, mode, "gdn.__weight.norm_weight")  # [128] f32
    ad = torch.stack([A_log.float(), dt_bias.float()]).contiguous()  # [2,32]
    return qkv, ba, z, A_log, dt_bias, norm_w, ad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default=DEFAULT_DUMPS)
    args = ap.parse_args()
    dd = args.dump_dir

    print(f"oracle dumps: {dd}")
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    # ============================================================ 1. gating
    print("\n[1] gating scalars (CUDA probe vs gdn.beta / gdn.decay_g)")
    for mode in ("prefill", "decode"):
        qkv, ba, z, A_log, dt_bias, norm_w, ad = build_inputs(dd, mode)
        T = ba.shape[0]
        beta_out = torch.zeros(T, NUM_V_HEADS, dtype=BF16, device=DEV)
        g_out = torch.zeros(T, NUM_V_HEADS, dtype=torch.float32, device=DEV)
        gdn.gdn_gating_probe(ba.contiguous(), ad.contiguous(), beta_out, g_out)
        torch.cuda.synchronize()
        check(f"{mode} beta = sigmoid(b) [bf16-native]", beta_out,
              load(dd, mode, "gdn.beta")[0])
        check(f"{mode} g = -exp(A_log)*softplus(a+dt)", g_out,
              load(dd, mode, "gdn.decay_g")[0])

    # ============================================================ 2. decode
    print("\n[2] decode step (1 token, carried state from gdn.core_state_before)")
    qkv, ba, z, A_log, dt_bias, norm_w, ad = build_inputs(dd, "decode")
    S_hf0 = load(dd, "decode", "gdn.core_state_before")[0]     # [32,128,128] kv
    S0 = S_hf0.transpose(-1, -2).contiguous().unsqueeze(0)     # [1,32,Dv,Dk]
    out, o_dbg, st = run_kernel(qkv, ba, ad, S0, z, norm_w, [0])

    o_ref = load(dd, "decode", "gdn.core_attn_out")[0].reshape(-1, VAL_DIM)
    y_ref = load(dd, "decode", "gdn.gated_norm_out").reshape(-1, VAL_DIM)
    S_ref = load(dd, "decode", "gdn.core_state_after")[0].transpose(-1, -2)
    check("decode o  vs gdn.core_attn_out  (bf16)", o_dbg, o_ref)
    check("decode y  vs gdn.gated_norm_out (bf16)", out, y_ref)
    check("decode S  vs gdn.core_state_after (fp32)", st[0], S_ref.contiguous(),
          expect_exact=False,
          note="fp32 dot association order, see kernel header")

    # ---- counterfactuals: prove each rounding decision is load-bearing
    print("\n[2b] counterfactual rounding orders (each MUST miss)")
    o_c, y_c, _ = ref_sequential(qkv, ba, z, A_log, dt_bias, norm_w,
                                 S0[0], l2fn=l2norm_fp32)
    check("D1 fp32 l2norm (vLLM order) -> o", o_c, o_ref, expect_exact=False,
          note="counterfactual: must NOT be bit-exact")
    if int((o_c != o_ref).sum()) == 0:
        FAILURES.append("D1 counterfactual did not discriminate")
    o_c, y_c, _ = ref_sequential(qkv, ba, z, A_log, dt_bias, norm_w,
                                 S0[0], round_xhat=False)
    check("D3 all-fp32 epilogue (arch doc) -> y", y_c, y_ref,
          expect_exact=False, note="counterfactual: must NOT be bit-exact")
    if int((y_c != y_ref).sum()) == 0:
        FAILURES.append("D3 counterfactual did not discriminate")
    # D2: feed a NON-rounded o into the epilogue
    o_fp32_ref = load(dd, "decode", "gdn.core_attn_out")[0].reshape(
        -1, NUM_V_HEADS, HEAD_V_DIM)
    y_hf = gated_rmsnorm(o_fp32_ref[0], z[0].reshape(NUM_V_HEADS, HEAD_V_DIM),
                         norm_w).reshape(1, -1)
    check("epilogue applied to oracle o -> y", y_hf, y_ref)

    # ---- torch reference in HF order must agree with the kernel
    print("\n[2c] torch reference (HF order) vs kernel")
    o_r, y_r, S_r = ref_sequential(qkv, ba, z, A_log, dt_bias, norm_w, S0[0])
    check("decode o  kernel vs torch-ref", o_dbg, o_r)
    check("decode y  kernel vs torch-ref", out, y_r)
    check("decode S  kernel vs torch-ref", st[0], S_r, expect_exact=False,
          note="both fp32, different reduction association")
    # The falsifiable version: same association order => must be bit-exact,
    # fp32 state included. This is the strict kernel-correctness gate.
    o_e, y_e, S_e = ref_sequential(qkv, ba, z, A_log, dt_bias, norm_w, S0[0],
                                   dot=dot_kernel_order, fma=True)
    check("decode o  kernel vs EXACT-ORDER ref", o_dbg, o_e)
    check("decode y  kernel vs EXACT-ORDER ref", out, y_e)
    check("decode S  kernel vs EXACT-ORDER ref (fp32)", st[0], S_e)

    # ============================================================ 3. prefill
    print("\n[3] chunked prefill (T=8, zero initial state)")
    qkv, ba, z, A_log, dt_bias, norm_w, ad = build_inputs(dd, "prefill")
    S0 = torch.zeros(1, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM,
                     dtype=torch.float32, device=DEV)
    out, o_dbg, st = run_kernel(qkv, ba, ad, S0, z, norm_w, [1])

    print("  (a) STRICT GATE: vs a torch reference carrying the kernel's own")
    print("      fp32 association order - must be bit-exact, state included")
    o_e, y_e, S_e = ref_sequential(qkv, ba, z, A_log, dt_bias, norm_w, S0[0],
                                   dot=dot_kernel_order, fma=True)
    check("prefill o  kernel vs EXACT-ORDER ref", o_dbg, o_e)
    check("prefill y  kernel vs EXACT-ORDER ref", out, y_e)
    check("prefill S  kernel vs EXACT-ORDER ref (fp32)", st[0], S_e)

    print("  (b) vs the SAME algorithm under torch's association order")
    o_r, y_r, S_r = ref_sequential(qkv, ba, z, A_log, dt_bias, norm_w, S0[0])
    check("prefill o  kernel vs torch-order ref", o_dbg, o_r,
          expect_exact=False, note="fp32 association compounds over 8 tokens")
    check("prefill y  kernel vs torch-order ref", out, y_r,
          expect_exact=False, note="downstream of the o flips")
    check("prefill S  kernel vs torch-order ref", st[0], S_r,
          expect_exact=False, note="fp32 reduction association")

    print("  (c) vs the HF dump, which used torch_chunk_gated_delta_rule")
    print("      (chunk_size=64 WY/UT transform - a DIFFERENT algorithm, so")
    print("       agreement is bounded by the two algorithms' fp32 paths)")
    o_ref = load(dd, "prefill", "gdn.core_attn_out")[0].reshape(-1, VAL_DIM)
    y_ref = load(dd, "prefill", "gdn.gated_norm_out").reshape(-1, VAL_DIM)
    S_ref = load(dd, "prefill", "gdn.core_state_after")[0].transpose(-1, -2)
    check("prefill o  vs gdn.core_attn_out (chunked)", o_dbg, o_ref,
          expect_exact=False, note="sequential vs chunked algorithm")
    check("prefill y  vs gdn.gated_norm_out (chunked)", out, y_ref,
          expect_exact=False, note="sequential vs chunked algorithm")
    check("prefill S  vs gdn.core_state_after (chunked)", st[0],
          S_ref.contiguous(), expect_exact=False,
          note="sequential vs chunked algorithm")

    # ---- the same gap measured for the torch sequential reference, to show it
    # ---- is the ALGORITHM and not the kernel
    check("prefill o  torch-seq vs chunked dump", o_r, o_ref,
          expect_exact=False, note="same gap without any CUDA involved")

    # ============================================================ summary
    print("\n" + "=" * 78)
    print(f"{'check':48s} {'status':10s} bitdiff")
    for tag, status, ne, n, maxabs, relrms, note in TABLE:
        print(f"{tag:48s} {status:10s} {ne}/{n}  maxabs={maxabs:.3e}")
    if FAILURES:
        print(f"\nFAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        raise SystemExit(1)
    print("\nALL BIT-EXACTNESS TARGETS MET")


if __name__ == "__main__":
    main()
