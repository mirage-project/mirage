"""DSV3 permuted grouped FP8 GEMM via PersistentKernel test_mode.

Exercises `fp8_group_gemm_layer` (the production MoE call) end-to-end
through the full MPK compile/run pipeline. The layer registers a single
grouped-GEMM kernel, fp8_group_gemm_largem_sm100 (BN=128, NS=6).

Two DSV3 routed-expert shapes are covered (ep_size=1 -> routed_tp = world_size):
  * W13 (gate||up):  K = HIDDEN = 7168,            N = 2*MOE_INTERMEDIATE/tp
  * W2  (down):      K = MOE_INTERMEDIATE/tp,      N = HIDDEN = 7168
plus a small-M decode-niche arm (K=7168, MPE<=8) so the small rows-per-expert
regime is covered too (the W13/W2 production shapes use MPE=128).

Layout / sizing decisions:
  * The kernel tile dispatch is M_total-driven: total = ceil(M_total/BM)*nn,
    and m_indices is read under an `m_start < M_total` guard
    (fp8_group_gemm_sm100_common.cuh L152/L289), so it is correct for ANY
    E/MPE. The production W13/W2 arms use the ep>=2 design point E=128,
    MPE=128 (M_total=16384), sweeping TP through N (W13) and K (W2); the
    decode arms sweep MPE in {1,2,4,8} at E=32 (M_total <= 256). bs=16
    (MPE>=16) is covered by the E=128 arm.
  * Scales are built directly via the shared UE8M0 helper + the transposed
    (num_sf_k, dim) packer in pytorch_reference (the layer's own SFA/SFB
    contract, identical to builder._pack_moe_scale_ue8m0). The reference
    dequants the SAME fp8 bytes + SAME UE8M0 scales the kernel consumes, so
    the comparison is numerically near-exact.

Run:
    python tests/runtime_python/blackwell/sm100_fp8_group_gemm_decode/test_fp8_group_gemm_decode_testmode.py
"""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402
from pytorch_reference import (  # noqa: E402
    quantize_rowblock_ue8m0,
    pack_sf_transposed,
    grouped_gemm_ref,
    cosine_sim,
    rel_mean,
)

HIDDEN = 7168
MOE_INTERMEDIATE = 2048  # per-expert intermediate (routed); TP=1


def _w13_n(tp: int) -> int:
    # gate||up routed: N = 2 * (MOE_INTERMEDIATE // routed_tp). Mult of 128.
    n = 2 * (MOE_INTERMEDIATE // tp)
    assert n % 128 == 0, (tp, n)
    return n


def _w2_k(tp: int) -> int:
    k = MOE_INTERMEDIATE // tp
    assert k % 128 == 0, (tp, k)
    return k


def _run_case(label, tp, bs, E, MPE, K, N, seed=42, active_experts=None):
    """One grouped-GEMM config. M_total = E*MPE. Returns (passed, cos, rel, tag).

    active_experts: optional list of expert ids. When set, a production-style
    meta buffer is passed (mask = row 1's first E int32s, see
    _fp8_group_gemm_layer_impl) marking ONLY those experts active — the kernel
    must skip every other expert's tiles (their output rows stay zero) WITHOUT
    deadlocking. This pins the accumulator-ring regression: the
    btf/bte ring was phased on the raw tile-iter counter, which advances across
    skipped tiles while arrivals only happen for processed ones — a scattered
    mask + multi-tile-iter (total tiles >> num_workers) made mb_wait spin
    forever. None = legacy nullptr-mask path (process every tile).
    """
    M_total = E * MPE
    regime = "small-M" if (K > 4096 and MPE <= 8) else "large-M"
    tag = (f"[{label}] tp={tp} bs={bs} E={E} MPE={MPE} M_total={M_total} "
           f"K={K} N={N} ({regime})"
           + (f" mask={len(active_experts)}/{E}" if active_experts else ""))
    print(f"\n{'='*80}\n{tag}\n{'='*80}", flush=True)

    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    a_bf16 = torch.randn((M_total, K), device=device, dtype=torch.bfloat16,
                         generator=g) * 0.5
    b_bf16 = torch.randn((E, N, K), device=device, dtype=torch.bfloat16,
                         generator=g) * 0.5

    # Quantize A (M_total x K) and B (E*N x K) to fp8 + per-(row,128-K) UE8M0.
    a_fp8, sa_dec, sa_enc = quantize_rowblock_ue8m0(a_bf16)
    b_flat = b_bf16.reshape(E * N, K)
    b_fp8_flat, sb_dec_flat, sb_enc_flat = quantize_rowblock_ue8m0(b_flat)
    b_fp8 = b_fp8_flat.reshape(E, N, K).contiguous()
    nk = K // 128
    sb_dec = sb_dec_flat.reshape(E, N, nk)

    # Transposed UE8M0-packed scales (the layer's SFA/SFB contract).
    sfa_packed = pack_sf_transposed(sa_enc)             # (num_sf_k, M_total)
    sfb_packed = pack_sf_transposed(sb_enc_flat)        # (num_sf_k, E*N)

    # Expert per row; kernel reads one per BM=128 block.
    m_indices = (torch.arange(M_total, device=device, dtype=torch.int32)
                 // MPE).contiguous()

    ref = grouped_gemm_ref(a_fp8, sa_dec, b_fp8, sb_dec, m_indices)
    output = torch.zeros((M_total, N), device=device, dtype=torch.bfloat16)

    meta = None
    if active_experts is not None:
        # Production meta layout (_fp8_group_gemm_layer_impl): 2D int32; the
        # active-expert mask is the first E entries of ROW 1 (flat offset =
        # meta.dim(1)). Width just needs >= E.
        meta = torch.zeros((2, max(E, 8)), device=device, dtype=torch.int32)
        active_idx = torch.tensor(sorted(active_experts), device=device,
                                  dtype=torch.long)
        meta[1, active_idx] = 1
        # Reference: inactive experts' tiles are skipped -> their output rows
        # keep the zero init.
        inactive_row = (meta[1, :E] == 0)[m_indices.to(torch.long)]
        ref = ref.clone()
        ref[inactive_row] = 0

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    # TP is a per-rank SHAPE selector only (N/K shard); world_size=1 so no
    # NVSHMEM/MPI is needed (decision-log convention).
    params["world_size"] = 1
    params["max_num_batched_tokens"] = max(1, bs)
    params["max_num_batched_requests"] = max(1, bs)
    pk = PersistentKernel(**params)

    a_dt = pk.attach_input(a_fp8.view(torch.uint8), name="a_fp8")
    b_dt = pk.attach_input(b_fp8.view(torch.uint8), name="b_fp8")
    sfa_dt = pk.attach_input(sfa_packed, name="sfa_packed")
    sfb_dt = pk.attach_input(sfb_packed, name="sfb_packed")
    mi_dt = pk.attach_input(m_indices, name="m_indices")
    out_dt = pk.attach_input(output, name="output")
    meta_dt = pk.attach_input(meta, name="meta") if meta is not None else None

    pk.fp8_group_gemm_layer(
        a_fp8=a_dt, b_fp8=b_dt,
        sfa_packed=sfa_dt, sfb_packed=sfb_dt,
        m_indices=mi_dt, output=out_dt,
        num_workers=num_workers,
        meta=meta_dt,
    )

    compile_dir = os.path.join(
        THIS_DIR, f".pk_compile_{label}_tp{tp}_bs{bs}_mpe{MPE}")
    os.makedirs(compile_dir, exist_ok=True)
    pk.compile(output_dir=compile_dir)
    pk()
    torch.cuda.synchronize()

    row_is_zero = output.float().abs().sum(dim=1) == 0
    if active_experts is not None:
        # Masked case: inactive experts' rows MUST stay zero; active rows
        # must be written. cos/rel are computed on the full matrix (zeros
        # match the zeroed reference rows).
        active_row = (meta[1, :E] == 1)[m_indices.to(torch.long)]
        rows_ok = (not row_is_zero[active_row].any()) and \
            row_is_zero[~active_row].all().item()
    else:
        rows_ok = not row_is_zero.any()
    zero_rows = row_is_zero.nonzero(as_tuple=True)[0]
    cos = cosine_sim(output, ref)
    rel = rel_mean(output, ref)
    max_diff = (output.float() - ref.float()).abs().max().item()
    # grouped fp8: cosine > 0.99 OR rel <= 5% (decision-log fp8 tolerance).
    passed = (cos > 0.99 or rel <= 0.05) and rows_ok
    print(f"  cos={cos:.6f} rel={rel*100:.4f}% max_abs_diff={max_diff:.4f} "
          f"zero_rows={zero_rows.numel()} -> "
          f"{'PASS' if passed else 'FAIL'}", flush=True)

    pk.finalize()
    return passed, cos, rel, tag


def main():
    results = []
    smoke = os.environ.get("MPK_SMOKE") == "1"

    if smoke:
        results.append(_run_case("W13", 1, 128, E=128, MPE=128,
                                 K=HIDDEN, N=_w13_n(1)))
        results.append(_run_case("decode", 1, 8, E=32, MPE=8,
                                 K=HIDDEN, N=_w13_n(1)))
        return _summary(results)

    # ── Production W13/W2 arms at the ep>=2 design point ──
    # E=128, MPE=128 -> M_total=16384. TP swept through N (W13) and K (W2).
    # Hits TP in {1,2,4,8}; the bs axis is not a GEMM shape lever here (the
    # permuted M_total is expert-padded, independent of token count).
    for tp in (1, 2, 4, 8):
        results.append(_run_case("W13", tp, 128, E=128, MPE=128,
                                 K=HIDDEN, N=_w13_n(tp)))
    for tp in (1, 2, 4, 8):
        results.append(_run_case("W2", tp, 128, E=128, MPE=128,
                                 K=_w2_k(tp), N=HIDDEN))

    # ── Decode-niche arms (small rows-per-expert, K>4096 & MPE<=8) ──
    # MPE in {1,2,4,8} maps the decode rows-per-expert (bs) axis; the largem
    # kernel is M_total-driven so any E/MPE is correct. E=32 exercises a
    # smaller expert count. bs=16 (MPE>=16) is covered by the E=128 arm above.
    for MPE in (1, 2, 4, 8):
        results.append(_run_case("decode", 1, MPE, E=32, MPE=MPE,
                                 K=HIDDEN, N=_w13_n(1)))
    # A TP corner (tp=8 shrinks N) so a decode arm sees a sharded N too.
    results.append(_run_case("decode", 8, 8, E=32, MPE=8,
                             K=HIDDEN, N=_w13_n(8)))

    # ── MASKED largem (production active-skip path) ──
    # Scattered 6-of-128 active experts + M_total=16384 ⇒ total tiles >>
    # num_workers (multi-tile-iter) + mixed skip — the exact geometry of the
    # accumulator-ring deadlock (this case HANGS on the pre-fix
    # kernel). W13 and W2 shapes at tp=2.
    _scattered = [3, 17, 42, 77, 101, 120]
    results.append(_run_case("W13-mask", 2, 8, E=128, MPE=128,
                             K=HIDDEN, N=_w13_n(2),
                             active_experts=_scattered))
    results.append(_run_case("W2-mask", 2, 8, E=128, MPE=128,
                             K=_w2_k(2), N=HIDDEN,
                             active_experts=_scattered))

    return _summary(results)


def _summary(results):
    print(f"\n{'='*80}\nSummary (fp8_group_gemm_decode):\n{'='*80}", flush=True)
    all_passed = True
    for passed, cos, rel, tag in results:
        print(f"  {'PASS' if passed else 'FAIL'}  cos={cos:.5f} "
              f"rel={rel*100:.4f}%  {tag}", flush=True)
        all_passed = all_passed and passed
    n_pass = sum(1 for r in results if r[0])
    print(f"\n{'ALL PASS' if all_passed else 'SOME FAILED'} "
          f"({n_pass}/{len(results)})", flush=True)
    return 0 if all_passed else 1


def test_fp8_group_gemm_decode_testmode():
    rc = main()
    assert rc == 0, "some fp8_group_gemm_decode configs failed"


if __name__ == "__main__":
    sys.exit(main())
