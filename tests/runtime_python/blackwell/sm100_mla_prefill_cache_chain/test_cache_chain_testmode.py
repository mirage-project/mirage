"""M1->M2 bridge: unabsorbed MLA prefill fed from the REAL compressed-latent
cache path, through MPK test_mode (DSV3 TP8 per-rank shapes).

Chain under test (the M2 materialized-prefill dataflow, scratch/preview_refactor.md
§8.3 / Codex bridge-check):

    kv_buf [seq_pad, 576]               (THE decode-shared compressed cache:
      |                                  [kv_norm(c_latent) 512 | rope'd k_pe 64])
      ├─ ckv_sep = narrow(:, 0:512)     (strided view, row stride 576)
      │     └─ quantize_fp8 (f32 scale) ──► latent_fp8 + scale
      │            ├─ fp8_gemm_dense (kv_b_k) ──► k_nope [seq_pad, H*128] (TRANSIENT)
      │            └─ fp8_gemm_dense (kv_b_v) ──► v      [seq_pad, H*128] (TRANSIENT)
      └─ kpe_sep = narrow(:, 512:576)   (strided view, fed STRAIGHT to attention TMA)

    mla_prefill_tp8_chunked(q_nope, q_pe, k_nope, kpe_sep, v) ──► attn_out

vs a PyTorch oracle that mirrors each stage (host fp8 quantize + reference GEMM +
causal absolute-position attention, bare 1/sqrt(192) scale).

What this proves beyond the per-kernel tests:
  * the cache->attention handoff: attention runs off per-head K/V materialized
    from the SAME compressed cache decode reads (preview_refactor.md §1.2bis);
  * narrow-view STRIDED inputs work end-to-end: quantize reads ckv_sep
    (stride 576) and the attention K_rope TMA descriptor honours stride[0]
    (tma.cuh param_id==3) — no gather/copy needed in the M2 builder design;
  * P_prev>0 causal masking against real materialized history.

KEY DECISIONS (mirroring the proven sibling tests):
  * kv rows padded to a 128 multiple (chunked TMA BN=128; zero tail quantizes
    to zero k/v rows -> masked, no 0*NaN).
  * mbt = seq_pad so default-meta active_rows covers every GEMM row
    (runtime_m = min(M, active_rows)); attention M comes from mla_params
    literals (test-mode branch), independent of mbt.
  * GEMM variant by max_seq_length: 4096 -> mediumm (the production kv_b
    variant, runtime_m_mode=1 path's home) and one 256 -> smallm spot case.
  * quantize: scale_ue8m0=False (f32 [rows, K/128] scale — the dense-GEMM
    activation layout), active_mode=0 (always-run; prefill-gating is an e2e
    concern, not part of this chain's math).
  * SCALE = bare 1/sqrt(192) (what the chunked codegen hardcodes).

Run:
    python tests/runtime_python/blackwell/sm100_mla_prefill_cache_chain/test_cache_chain_testmode.py
"""

import math
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DENSE_DIR = os.path.abspath(os.path.join(THIS_DIR, "../sm100_fp8_gemm_dense"))
for d in (THIS_DIR, _DENSE_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402
from pytorch_reference import (  # noqa: E402  (sm100_fp8_gemm_dense/)
    quantize_a_f32scale,
    quantize_b_f32scale,
    reference_gemm,
)

H = 16            # TP8 per-rank local heads (128 / 8)
KV_LORA = 512     # compressed latent dim
ROPE = 64         # k_pe dim
D_KV = KV_LORA + ROPE  # 576 cache row
QK_NOPE = 128
V_DIM = 128
SM_SCALE = 1.0 / math.sqrt(192.0)  # bare, matches chunked codegen


def _pad128(n: int) -> int:
    return ((n + 127) // 128) * 128


def _oracle(kv_buf, q_nope, q_pe, wk_fp8, swk, wv_fp8, swv, q_len, kv_len,
            p_prev):
    """Mirror the chain stage-by-stage on the host (fp32 accumulation)."""
    seq_pad = kv_buf.shape[0]
    latent = kv_buf[:, :KV_LORA].to(torch.bfloat16)        # [seq_pad, 512]
    kpe = kv_buf[:kv_len, KV_LORA:].float()                # [kv_len, 64]

    # Stage 1: fp8 quantize of the latent rows (host mirror of quantize_fp8).
    a_fp8, sa = quantize_a_f32scale(latent)
    # Stage 2: kv_b dense GEMMs (reference, applies sa/sb like the kernel).
    k_flat = reference_gemm(a_fp8, sa, wk_fp8, swk)        # [seq_pad, H*128]
    v_flat = reference_gemm(a_fp8, sa, wv_fp8, swv)
    k_nope = k_flat.float().view(seq_pad, H, QK_NOPE)[:kv_len]
    v = v_flat.float().view(seq_pad, H, V_DIM)[:kv_len]

    # Stage 3: causal unabsorbed attention, absolute positions.
    qn = q_nope.float()                                    # [q_len, H, 128]
    qp = q_pe.float()                                      # [q_len, H, 64]
    s_nope = torch.einsum("shd,thd->sht", qn, k_nope)
    s_pe = torch.einsum("shr,tr->sht", qp, kpe)            # k_pe shared/head
    scores = (s_nope + s_pe) * SM_SCALE
    qpos = torch.arange(q_len, device=scores.device) + p_prev
    kpos = torch.arange(kv_len, device=scores.device)
    allow = kpos[None, :] <= qpos[:, None]                 # [q_len, kv_len]
    scores = scores.masked_fill(~allow.unsqueeze(1), float("-inf"))
    probs = scores.softmax(dim=-1)
    return torch.einsum("sht,thd->shd", probs, v)          # [q_len, H, 128]


def run_case(q_len, p_prev, max_seq_length, seed=7):
    kv_len = p_prev + q_len
    seq_pad = _pad128(kv_len)
    variant = "smallm" if max_seq_length <= 512 else "mediumm"
    tag = (f"q={q_len} P_prev={p_prev} kv={kv_len} pad={seq_pad} "
           f"H={H} [{variant}]")
    print(f"\n{'='*74}\n{tag}\n{'='*74}", flush=True)

    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed + q_len + p_prev)

    def r(*shape, scale=0.5):
        return (torch.randn(*shape, generator=g, device=device) * scale).to(
            torch.bfloat16)

    # The compressed cache: real rows [0, kv_len), zero tail (BN padding).
    kv_buf = torch.zeros((seq_pad, D_KV), device=device, dtype=torch.bfloat16)
    kv_buf[:kv_len] = r(kv_len, D_KV)
    # Already-projected/rotated Q (the §8.1 ABI boundary).
    q_nope = r(q_len, H, QK_NOPE)
    q_pe = r(q_len, H, ROPE)
    # kv_b weights, checkpoint-style FP8 + 128x128-block f32 scale.
    wk_bf16 = r(H * QK_NOPE, KV_LORA, scale=0.3)
    wv_bf16 = r(H * V_DIM, KV_LORA, scale=0.3)
    wk_fp8, swk = quantize_b_f32scale(wk_bf16)
    wv_fp8, swv = quantize_b_f32scale(wv_bf16)

    ref = _oracle(kv_buf, q_nope, q_pe, wk_fp8, swk, wv_fp8, swv,
                  q_len, kv_len, p_prev)

    # Transient materialization targets + attention output.
    latent_fp8 = torch.zeros((seq_pad, KV_LORA), device=device,
                             dtype=torch.float8_e4m3fn)
    latent_scale = torch.zeros((seq_pad, KV_LORA // 128), device=device,
                               dtype=torch.float32)
    k_nope_buf = torch.zeros((seq_pad, H * QK_NOPE), device=device,
                             dtype=torch.bfloat16)
    v_buf = torch.zeros((seq_pad, H * V_DIM), device=device,
                        dtype=torch.bfloat16)
    out = torch.zeros((q_len, H, V_DIM), device=device, dtype=torch.bfloat16)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    # mbt = seq_pad so default-meta active_rows covers every GEMM row.
    params["max_num_batched_tokens"] = seq_pad
    params["max_num_batched_requests"] = 1
    params["max_seq_length"] = max_seq_length
    pk = PersistentKernel(**params)

    kv_dt = pk.attach_input(kv_buf, name="kv_buf")
    qn_dt = pk.attach_input(q_nope, name="q_nope")
    qp_dt = pk.attach_input(q_pe, name="q_pe")
    wk_dt = pk.attach_input(wk_fp8, name="kv_b_k_w")
    swk_dt = pk.attach_input(swk, name="kv_b_k_s")
    wv_dt = pk.attach_input(wv_fp8, name="kv_b_v_w")
    swv_dt = pk.attach_input(swv, name="kv_b_v_s")
    lf_dt = pk.attach_input(latent_fp8, name="latent_fp8")
    ls_dt = pk.attach_input(latent_scale, name="latent_scale")
    kn_dt = pk.attach_input(k_nope_buf, name="k_nope")
    v_dt = pk.attach_input(v_buf, name="v")
    out_dt = pk.attach_input(out, name="attn_out")

    # The two strided views of the cache (row stride 576).
    ckv_sep = pk.narrow(kv_dt, 1, 0, KV_LORA)
    kpe_sep = pk.narrow(kv_dt, 1, KV_LORA, ROPE)

    # Stage 1: quantize the latent view (one CTA per row).
    # Column-slice contract (quantize_fp8_layer docstring): the narrow view
    # carries the base offset, but the row stride does NOT auto-derive — it
    # defaults to the view width (512), silently reading the wrong rows.
    # Pass the parent row width (576) explicitly, like the QKV-a call sites.
    pk.quantize_fp8_layer(
        input=ckv_sep,
        output_fp8=lf_dt,
        output_scale=ls_dt,
        grid_dim=(seq_pad, 1, 1),
        block_dim=(128, 1, 1),
        scale_ue8m0=False,
        active_mode=0,
        hidden_size_override=KV_LORA,
        input_stride_override=D_KV,
    )
    # Stage 2: kv_b_k / kv_b_v dense FP8 GEMMs.
    gemm_layer = (pk.fp8_gemm_dense_smallm_layer if max_seq_length <= 512
                  else pk.fp8_gemm_dense_mediumm_layer)
    gemm_layer(input_fp8=lf_dt, weight_fp8=wk_dt, input_scale=ls_dt,
               weight_scale=swk_dt, output=kn_dt, num_workers=num_workers)
    gemm_layer(input_fp8=lf_dt, weight_fp8=wv_dt, input_scale=ls_dt,
               weight_scale=swv_dt, output=v_dt, num_workers=num_workers)
    # Stage 3: chunked unabsorbed attention; k_rope comes STRAIGHT from the
    # cache view (strided TMA), k_nope/v from the transient GEMM outputs.
    pk.mla_prefill_tp8_chunked_layer(
        q_nope=qn_dt,
        q_pe=qp_dt,
        k_nope=kn_dt,
        k_rope=kpe_sep,
        v=v_dt,
        output=out_dt,
        mla_params=(H, q_len, kv_len, p_prev),
        grid_dim=(H, (q_len + 63) // 64, 1),
        block_dim=(128, 1, 1),
        qfused_mode=0,
    )

    folder = os.path.join(THIS_DIR, f".pk_chain_{q_len}_{p_prev}_{variant}")
    os.makedirs(folder, exist_ok=True)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    got = out.float()
    nan = int(torch.isnan(got).sum().item())
    max_diff = (got - ref).abs().max().item()
    rel = ((got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-9)).item()
    cos = torch.nn.functional.cosine_similarity(
        got.flatten(), ref.flatten(), dim=0).item()
    # FP8-stack tolerance (dense-test convention): cosine-dominant; the
    # per-element max can reach ~1e-1 from a single borderline fp8 rounding
    # flip on a high-softmax-weight score (kernel quantize vs host mirror).
    passed = (nan == 0) and cos > 0.995 and rel <= 0.05 and max_diff < 0.3
    print(f"  max_diff={max_diff:.5f} rel={rel*100:.3f}% cos={cos:.6f} "
          f"nan={nan} -> {'PASS' if passed else 'FAIL'}", flush=True)

    pk.finalize()
    return passed, max_diff, cos, nan, tag  # rel printed above


def main():
    # (q_len, p_prev, max_seq_length): P_prev>0 + non-multiple kv_len are the
    # core M1 contract; mediumm = production kv_b variant + one smallm spot.
    cases = [
        (64, 0, 4096),     # first prefill, single q-block
        (64, 17, 4096),    # prior history, non-multiple kv_len (81)
        (128, 31, 4096),   # multi-q-block + history (kv 159, partial block)
        (64, 128, 4096),   # long prior history (kv 192)
        (64, 17, 256),     # smallm variant spot check
    ]
    results = [run_case(*c) for c in cases]
    print(f"\n{'='*74}\nSummary\n{'='*74}", flush=True)
    ok = True
    for passed, md, cos, nan, tag in results:
        print(f"  {'PASS' if passed else 'FAIL'}  max_diff={md:.5f} "
              f"cos={cos:.6f} nan={nan}  {tag}", flush=True)
        ok = ok and passed
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} "
          f"({sum(r[0] for r in results)}/{len(results)})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
