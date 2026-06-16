"""Bisect the cache-chain failure: per-stage intermediate checks + view-vs-dense
swaps. One case (q=64, P_prev=17, mediumm).

Flags (env):
  DENSE_QUANT_IN=1  -> feed quantize a DENSE copy of kv_buf[:, :512] (no view)
  DENSE_KROPE=1     -> feed attention a DENSE copy of kv_buf[:, 512:] (no view)
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
from pytorch_reference import (  # noqa: E402
    quantize_a_f32scale,
    quantize_b_f32scale,
    reference_gemm,
)

H, KV_LORA, ROPE = 16, 512, 64
D_KV = KV_LORA + ROPE
SM_SCALE = 1.0 / math.sqrt(192.0)


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0).item()


def main():
    dense_quant_in = os.environ.get("DENSE_QUANT_IN") == "1"
    dense_krope = os.environ.get("DENSE_KROPE") == "1"
    q_len, p_prev, msl = 64, 17, 4096
    kv_len = p_prev + q_len
    seq_pad = 128
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(7 + q_len + p_prev)

    def r(*shape, scale=0.5):
        return (torch.randn(*shape, generator=g, device=device) * scale).to(
            torch.bfloat16)

    kv_buf = torch.zeros((seq_pad, D_KV), device=device, dtype=torch.bfloat16)
    kv_buf[:kv_len] = r(kv_len, D_KV)
    q_nope = r(q_len, H, 128)
    q_pe = r(q_len, H, ROPE)
    wk_fp8, swk = quantize_b_f32scale(r(H * 128, KV_LORA, scale=0.3))
    wv_fp8, swv = quantize_b_f32scale(r(H * 128, KV_LORA, scale=0.3))

    # dense copies (for the swap flags)
    latent_dense = kv_buf[:, :KV_LORA].contiguous()
    kpe_dense = kv_buf[:, KV_LORA:].contiguous()

    # stage references
    ref_fp8, ref_sa = quantize_a_f32scale(kv_buf[:, :KV_LORA].to(torch.bfloat16))
    ref_k = reference_gemm(ref_fp8, ref_sa, wk_fp8, swk)
    ref_v = reference_gemm(ref_fp8, ref_sa, wv_fp8, swv)

    latent_fp8 = torch.zeros((seq_pad, KV_LORA), device=device,
                             dtype=torch.float8_e4m3fn)
    latent_scale = torch.zeros((seq_pad, KV_LORA // 128), device=device,
                               dtype=torch.float32)
    k_nope_buf = torch.zeros((seq_pad, H * 128), device=device,
                             dtype=torch.bfloat16)
    v_buf = torch.zeros((seq_pad, H * 128), device=device,
                        dtype=torch.bfloat16)
    out = torch.zeros((q_len, H, 128), device=device, dtype=torch.bfloat16)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = seq_pad
    params["max_num_batched_requests"] = 1
    params["max_seq_length"] = msl
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
    ld_dt = pk.attach_input(latent_dense, name="latent_dense")
    kd_dt = pk.attach_input(kpe_dense, name="kpe_dense")

    quant_in = ld_dt if dense_quant_in else pk.narrow(kv_dt, 1, 0, KV_LORA)
    krope_in = kd_dt if dense_krope else pk.narrow(kv_dt, 1, KV_LORA, ROPE)

    pk.quantize_fp8_layer(
        input=quant_in, output_fp8=lf_dt, output_scale=ls_dt,
        grid_dim=(seq_pad, 1, 1), block_dim=(128, 1, 1),
        scale_ue8m0=False, active_mode=0,
        # Column-slice contract: the view carries the base offset, but the
        # row stride must be passed explicitly (defaults to the view width).
        hidden_size_override=KV_LORA,
        input_stride_override=(None if dense_quant_in else D_KV))
    pk.fp8_gemm_dense_mediumm_layer(
        input_fp8=lf_dt, weight_fp8=wk_dt, input_scale=ls_dt,
        weight_scale=swk_dt, output=kn_dt, num_workers=num_workers)
    pk.fp8_gemm_dense_mediumm_layer(
        input_fp8=lf_dt, weight_fp8=wv_dt, input_scale=ls_dt,
        weight_scale=swv_dt, output=v_dt, num_workers=num_workers)
    pk.mla_prefill_tp8_chunked_layer(
        q_nope=qn_dt, q_pe=qp_dt, k_nope=kn_dt, k_rope=krope_in, v=v_dt,
        output=out_dt, mla_params=(H, q_len, kv_len, p_prev),
        grid_dim=(H, (q_len + 63) // 64, 1), block_dim=(128, 1, 1),
        qfused_mode=0)

    folder = os.path.join(
        THIS_DIR, f".pk_dbg_{int(dense_quant_in)}{int(dense_krope)}")
    os.makedirs(folder, exist_ok=True)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    print(f"\nDENSE_QUANT_IN={int(dense_quant_in)} DENSE_KROPE={int(dense_krope)}")
    # Stage 1: quantize output vs host reference
    got_deq = latent_fp8.float() * latent_scale.repeat_interleave(128, dim=1)
    ref_deq = ref_fp8.float() * ref_sa.repeat_interleave(128, dim=1)
    print(f"  quantize: cos={cos(got_deq, ref_deq):.6f} "
          f"fp8_bytes_equal={torch.equal(latent_fp8.view(torch.uint8), ref_fp8.view(torch.uint8))} "
          f"scale_maxdiff={(latent_scale - ref_sa).abs().max().item():.3e}")
    # Stage 2: GEMM outputs
    print(f"  k_nope:   cos={cos(k_nope_buf, ref_k):.6f} "
          f"maxdiff={(k_nope_buf.float() - ref_k.float()).abs().max().item():.4f}")
    print(f"  v:        cos={cos(v_buf, ref_v):.6f}")
    # Stage 3: attention vs oracle (using the REFERENCE k/v + real kpe)
    k3 = ref_k.float().view(seq_pad, H, 128)[:kv_len]
    v3 = ref_v.float().view(seq_pad, H, 128)[:kv_len]
    kpe = kv_buf[:kv_len, KV_LORA:].float()
    s = (torch.einsum("shd,thd->sht", q_nope.float(), k3)
         + torch.einsum("shr,tr->sht", q_pe.float(), kpe)) * SM_SCALE
    qpos = torch.arange(q_len, device=device) + p_prev
    kpos = torch.arange(kv_len, device=device)
    s = s.masked_fill(~(kpos[None, :] <= qpos[:, None]).unsqueeze(1),
                      float("-inf"))
    ref_out = torch.einsum("sht,thd->shd", s.softmax(dim=-1), v3)
    print(f"  attn:     cos={cos(out, ref_out):.6f} "
          f"maxdiff={(out.float() - ref_out).abs().max().item():.4f}")
    pk.finalize()


if __name__ == "__main__":
    main()
