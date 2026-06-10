"""Test mode: NEW DeepSeek-V3 decode MoE grouped FP8 GEMM, COMPACT dispatch.

Targets the largem + COMPACT active-mask kernel
`fp8_group_gemm_largem_compact_sm100.cuh` (the non-compact
`fp8_group_gemm_largem_sm100_task_impl` is a thin pass-through wrapper that
delegates EVERY arg to the compact impl — verified in-tree). Driven at the
exact DSv3 TP=4 EP=2 decode config (E=128 local experts, M_total=E*128,
K=7168 W13 hidden, N=2*routed_intermediate=2048). bs=1 decode routes one token
to only a few experts, so the compact kernel should iterate ONLY
num_active*nn tiles — this is the gate for the suspiciously-fast (~2 us)
compact decode GEMM in the 241 us trace.

It verifies the compact dispatch computes EXACTLY the active-expert set:
  (A) every ACTIVE expert's BM=128 output block equals a plain-PyTorch grouped
      GEMM reference (FP8 tol abs<2.0, rel<0.05), and
  (B) every INACTIVE expert's output block is LEFT UNTOUCHED (a NaN sentinel)
      -> no missed expert (the "2 us under-compute bug") and no extra work.

--------------------------------------------------------------------------
IMPORTANT FINDING (operand-ordering bug in the public layer API)
--------------------------------------------------------------------------
PersistentKernel.fp8_group_gemm_layer(..., meta=...) registers its TB operators
in the order [a, b, sfa, sfb, m_indices, output, meta] (meta appended LAST). But
the codegen contract (src/kernel/task_register.cc register_fp8_group_gemm_variant
+ graph.cc tuple (num_inputs=6, num_outputs=1) + tma.cuh
TASK_FP8_GROUP_GEMM_*_SM100) requires the D-output TMA descriptor to come from
outputs[0] and the active-expert mask from input_ptrs[5]. With meta appended
last, the positional split makes inputs[5]=output and outputs[0]=meta -> the
GEMM's D store is routed into the tiny `meta` buffer and the active mask is read
from the `output` buffer (garbage). This is the SAME class of bug that was fixed
for moe_silu_mul on 2026-05-14 (see the "CRITICAL ORDERING" comment in
persistent_kernel.py ~3938: the required order is [silu_input, meta,
silu_output]); the grouped-GEMM path never received the analogous fix.

The required (correct) order is [a, b, sfa, sfb, m_indices, meta, output].

This test therefore runs BOTH:
  * PART 1 — the public API exactly as-is, expected to mis-bind so the `output`
    tensor is never written (all-NaN). This documents/guards the bug.
  * PART 2 — the kernel driven with the CORRECTED operand order, which MUST
    pass full compact correctness. This is the real kernel-correctness gate and
    proves the ~2 us is legitimate compact behavior, not a work-skipping bug.

`EXPECT_API_BUG = True` encodes the current (buggy) behavior. When the API
ordering is fixed in persistent_kernel.py, flip it to False (Part 1 then must
match Part 2) and this test keeps guarding the fix.

Run:
    CUDA_VISIBLE_DEVICES=<g> .venv/bin/python \
        tests/runtime_python/blackwell/sm100_fp8_group_gemm_decode/test_compact_decode_testmode.py
"""

import os
import sys
import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel, TBGraph, CyTBGraph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import (  # noqa: E402
    quantize_fp8_blockk,
    pack_moe_scale_ue8m0,
    unpack_ue8m0_scale,
    group_gemm_compact_ref,
)

# Current public fp8_group_gemm_layer(meta=...) mis-orders operands (meta after
# output) -> the `output` tensor is never written. Set False once the layer is
# fixed to register [a,b,sfa,sfb,m_indices,meta,output].
EXPECT_API_BUG = False  # 2026-06-07: operand-order bug FIXED in persistent_kernel.py (_fp8_group_gemm_layer_impl now registers meta before output)


# ================================================================
# DSv3 TP=4 EP=2 DECODE config
#   NUM_EXPERTS=256, ep_size=2 -> num_local_experts E = 128
#   bm_padding=128 -> M_total = 128*128 = 16384
#   W13: K = hidden = 7168, N = 2 * routed_intermediate
#        routed_tp_size = world/ep = 4/2 = 2 -> routed_intermediate = 2048/2 = 1024
#        N_w13 = 2*1024 = 2048
# ================================================================
E = 128                 # num_local_experts (EP=2). MUST be <=128: the compact
                        # kernel's prologue scans experts [0,128) only.
BM_PADDING = 128
M_TOTAL = E * BM_PADDING  # 16384
K = 7168                 # hidden (W13 reduction dim)
N = 2048                 # 2 * routed_moe_intermediate_size (W13 output)
MBT = 1                  # bs=1 decode (one token this pass)
TOPK = 8                 # DSv3 num_experts_per_tok
META_LEN = M_TOTAL + MBT * TOPK  # row length of the (2, META_LEN) meta buffer

# Decode regime: a single token routed to a few distinct LOCAL experts at
# non-trivial ids (a mis-indexed compact scan would be caught). The "4active"
# decode point cited in the compact kernel header.
ACTIVE_EXPERTS = [0, 17, 63, 100]


def build_meta(active_experts, device):
    """Replicate moe_permute_sm100's meta row-1 layout:
        meta[1, e]          = active_expert_mask[e]  (0/1)
        meta[1, E_LOCAL+e]  = actual_count[e]
    The kernel reads active_expert_mask at input_ptrs[5] + active_mask_offset,
    with active_mask_offset = meta.dim(1) (= META_LEN) -> the first META_LEN
    entries of row 1.
    """
    meta = torch.zeros(2, META_LEN, dtype=torch.int32, device=device)
    for e in active_experts:
        meta[1, e] = 1            # active_expert_mask[e] = 1
        meta[1, E + e] = 1        # actual_count[e] = 1 (one decode row)
    return meta


def _make_inputs(device):
    torch.manual_seed(7)
    # Activations A (permuted, dense-padded): each expert owns BM_PADDING rows
    # from e*BM_PADDING; an active expert carries a real decode token in row 0,
    # the rest are zero. Inactive experts' blocks are all zero. Mirrors
    # moe_permute's output for a bs=1 decode pass.
    a_val = torch.zeros(M_TOTAL, K, device=device, dtype=torch.float32)
    for e in ACTIVE_EXPERTS:
        a_val[e * BM_PADDING] = torch.randn(K, device=device) * 0.1
    a_fp8, a_sf_packed, a_deq_scale = quantize_fp8_blockk(a_val)

    # Weights B (E, N, K) FP8 + per-(E, N/128, K/128) block scale, packed to the
    # (num_sf_k, E*N) UE8M0 transposed layout the kernel reads.
    b_val = torch.randn(E, N, K, device=device, dtype=torch.float32) / (K ** 0.5)
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    nk = K // 128
    bb = b_val.reshape(E, N, nk, 128)
    b_scale = (bb.abs().amax(dim=3) / FP8_MAX).clamp(min=1e-12)   # (E, N, nk)
    b_fp8 = (bb / b_scale.unsqueeze(3)).reshape(E, N, K).to(torch.float8_e4m3fn)
    sfb_packed = pack_moe_scale_ue8m0(b_scale.reshape(E * N, nk).contiguous())
    b_deq_scale = unpack_ue8m0_scale(sfb_packed, E * N, nk)       # (E*N, nk)

    num_sf_k = (nk + 3) // 4
    assert a_sf_packed.shape == (num_sf_k, M_TOTAL), a_sf_packed.shape
    assert sfb_packed.shape == (num_sf_k, E * N), sfb_packed.shape

    m_indices = (torch.arange(M_TOTAL, dtype=torch.int32, device=device)
                 // BM_PADDING).contiguous()
    meta = build_meta(ACTIVE_EXPERTS, device)
    ref = group_gemm_compact_ref(a_fp8, a_deq_scale, b_fp8, b_deq_scale,
                                 m_indices, ACTIVE_EXPERTS, BM_PADDING)
    return (a_fp8.view(torch.uint8).contiguous(),
            b_fp8.view(torch.uint8).contiguous(),
            a_sf_packed, sfb_packed, m_indices, meta, ref)


def _run_once(corrected_order):
    """Compile + run the compact group GEMM once and return the output tensor.

    corrected_order=False -> use the public fp8_group_gemm_layer API as-is.
    corrected_order=True  -> register operands as [a,b,sfa,sfb,m_indices,meta,
                             output] (meta BEFORE output) — the binding the
                             codegen contract actually requires.
    """
    device = "cuda"
    a_u8, b_u8, sfa, sfb, m_indices, meta, ref = _make_inputs(device)
    output = torch.full((M_TOTAL, N), float("nan"),
                        dtype=torch.bfloat16, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = MBT
    params["max_num_batched_requests"] = MBT
    pk = PersistentKernel(**params)

    a_dt = pk.attach_input(a_u8, name="a_fp8")
    b_dt = pk.attach_input(b_u8, name="b_fp8")
    sfa_dt = pk.attach_input(sfa, name="sfa_packed")
    sfb_dt = pk.attach_input(sfb, name="sfb_packed")
    mi_dt = pk.attach_input(m_indices, name="m_indices")
    out_dt = pk.attach_input(output, name="output")
    meta_dt = pk.attach_input(meta, name="meta")

    if not corrected_order:
        # Public API exactly as shipped (meta appended after output internally).
        pk.fp8_group_gemm_layer(
            a_fp8=a_dt, b_fp8=b_dt, sfa_packed=sfa_dt, sfb_packed=sfb_dt,
            m_indices=mi_dt, output=out_dt, num_workers=num_workers,
            meta=meta_dt,
        )
    else:
        # CORRECTED operand order: meta BEFORE output. Mirrors the documented
        # moe_silu_mul ordering fix. Drives the largem (=compact) task directly.
        active_mask_offset = meta_dt.dim(1)
        gg_params = [M_TOTAL, N, K, E, num_workers, active_mask_offset]
        tb = TBGraph(CyTBGraph((num_workers, 1, 1), (256, 1, 1), 1, 64))
        tb.new_input(a_dt, (-1, -1, -1), -1, True)
        tb.new_input(b_dt, (-1, -1, -1), -1, True)
        tb.new_input(sfa_dt, (-1, -1, -1), -1, True)
        tb.new_input(sfb_dt, (-1, -1, -1), -1, True)
        tb.new_input(mi_dt, (-1, -1, -1), -1, True)
        tb.new_input(meta_dt, (-1, -1, -1), -1, True)   # meta BEFORE output
        tb.new_input(out_dt, (-1, -1, -1), -1, True)    # output LAST = outputs[0]
        pk.kn_graph.customized(
            [a_dt, b_dt, sfa_dt, sfb_dt, mi_dt, meta_dt, out_dt], tb)
        pk.kn_graph.register_task(tb, "fp8_group_gemm_largem_sm100", gg_params)

    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    pk()
    torch.cuda.synchronize()
    out_f = output.float().clone()
    pk.finalize()
    return out_f, ref


def _check_compact(out_f, ref, label):
    """Return (active_ok, inactive_untouched, worst_abs, worst_rel, any_nan)."""
    active_ok = True
    any_active_nan = False
    worst_abs = worst_rel = 0.0
    worst_e = -1
    for e in ACTIVE_EXPERTS:
        rows = slice(e * BM_PADDING, e * BM_PADDING + BM_PADDING)
        o_blk, r_blk = out_f[rows], ref[rows]
        if torch.isnan(o_blk).any():
            any_active_nan = True
            active_ok = False
            print(f"    [{label}] active expert {e}: output block has NaN "
                  f"(kernel did NOT write it)")
            continue
        diff = (o_blk - r_blk).abs()
        max_abs = diff.max().item()
        max_rel = max_abs / max(r_blk.abs().max().item(), 1e-6)
        if max_abs > worst_abs:
            worst_abs, worst_rel, worst_e = max_abs, max_rel, e
        blk_ok = (max_abs < 2.0 and max_rel < 0.05)
        active_ok &= blk_ok
        print(f"    [{label}] active expert {e:3d}: abs={max_abs:.4f} "
              f"rel={max_rel:.4f}  {'OK' if blk_ok else 'MISMATCH'}")
    # inactive blocks must remain untouched (still NaN)
    touched = [e for e in range(E) if e not in ACTIVE_EXPERTS
               and torch.isfinite(out_f[e * BM_PADDING:e * BM_PADDING + BM_PADDING]).any()]
    inactive_untouched = (len(touched) == 0)
    if touched:
        print(f"    [{label}] {len(touched)} INACTIVE blocks WRITTEN "
              f"(over-compute): {touched[:16]}")
    else:
        print(f"    [{label}] all {E - len(ACTIVE_EXPERTS)} inactive blocks "
              f"UNTOUCHED")
    if worst_e >= 0:
        print(f"    [{label}] worst active block: expert {worst_e} "
              f"abs={worst_abs:.4f} rel={worst_rel:.4f}")
    return active_ok, inactive_untouched, worst_abs, worst_rel, any_active_nan


def test_compact_decode_testmode():
    print(f"\n{'=' * 72}")
    print("Test mode: fp8_group_gemm  (largem -> COMPACT decode dispatch)")
    print(f"  E={E}, M_total={M_TOTAL}, K={K}, N={N}, bm_pad={BM_PADDING}")
    print(f"  active_experts={ACTIVE_EXPERTS}  (bs=1 decode regime)")

    # ---- PART 2 (run first; it's the real kernel-correctness gate) ----
    print("\n[PART 2] CORRECTED operand order [a,b,sfa,sfb,m_indices,meta,output]")
    out_fix, ref = _run_once(corrected_order=True)
    fix_active_ok, fix_inact_ok, fix_abs, fix_rel, _ = _check_compact(
        out_fix, ref, "fixed")
    kernel_correct = fix_active_ok and fix_inact_ok

    # ---- PART 1 (public API as-is — documents the binding bug) ----
    print("\n[PART 1] PUBLIC API as-is  fp8_group_gemm_layer(meta=...)")
    out_api, _ = _run_once(corrected_order=False)
    api_active_ok, api_inact_ok, api_abs, api_rel, api_any_nan = _check_compact(
        out_api, ref, "public")

    print(f"\n{'-' * 72}")
    print("VERDICT")
    print(f"  Kernel (COMPACT, fixed order) correct: {kernel_correct}  "
          f"(active_ok={fix_active_ok}, inactive_untouched={fix_inact_ok}, "
          f"worst abs={fix_abs:.4f} rel={fix_rel:.4f})")
    print(f"  Public fp8_group_gemm_layer(meta=...) correct: "
          f"{api_active_ok and api_inact_ok}")

    # The compact kernel MUST be correct under the fixed binding. This is the
    # standardized correctness gate: it proves the ~2 us compact decode GEMM
    # processes EXACTLY the active experts (no missed -> no NaN; no extra ->
    # inactive untouched), i.e. NOT a work-skipping bug.
    assert kernel_correct, (
        "COMPACT decode group GEMM is INCORRECT even with the fixed operand "
        f"order: active_ok={fix_active_ok} inactive_untouched={fix_inact_ok} "
        f"worst_abs={fix_abs} worst_rel={fix_rel} — this would be the real "
        "work-skipping bug.")

    if EXPECT_API_BUG:
        # Guard the KNOWN operand-ordering bug in the public layer: it must NOT
        # write the `output` tensor (active blocks stay NaN). If this assertion
        # starts failing, the API ordering was fixed -> flip EXPECT_API_BUG.
        assert api_any_nan and not (api_active_ok and api_inact_ok), (
            "fp8_group_gemm_layer(meta=...) UNEXPECTEDLY produced a correct "
            "result — the operand-ordering bug appears fixed. Set "
            "EXPECT_API_BUG=False so Part 1 is asserted correct like Part 2.")
        print("\n  NOTE: public API mis-binds (meta after output) -> `output` "
              "left NaN. KNOWN bug, guarded by EXPECT_API_BUG=True.")
    else:
        assert api_active_ok and api_inact_ok, (
            "EXPECT_API_BUG=False but the public fp8_group_gemm_layer(meta=...) "
            f"is wrong: active_ok={api_active_ok} "
            f"inactive_untouched={api_inact_ok}")

    print("\nPASSED: fp8_group_gemm COMPACT decode correctness gate")


if __name__ == "__main__":
    test_compact_decode_testmode()
