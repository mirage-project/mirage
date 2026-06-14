"""PB: full DFlash draft DecoderLayer (layer 0) via MPK test-mode vs HF dump.

Reference (dflash.py):
  r1=h_in; a=self_attn(input_layernorm(h_in)); h2=r1+a;
  m=mlp(post_attention_layernorm(h2)); out=h2+m            == out::layers.0

Split-KV architecture (mbt=B): context K/V are materialized once (here in torch via
validated refs) and fed as CACHE inputs; the draft forward (block, B rows) runs fully
in-graph with max_num_batched_tokens=B. This mirrors the real materialize/draft-forward
split and avoids the single-mbt conflict between ctx_len-row and B-row tensors.

Run: CUDA_VISIBLE_DEVICES=2 python test_draft_layer_testmode.py
"""
import json
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.models.utils import grid_for_rmsnorm_linear_layer

sys.path.insert(0, os.path.dirname(__file__))
from pytorch_reference import (load_weight, rms_norm, linear, dflash_norm_rope,  # noqa
                               CKPT)
from verify_attention_ref import build_yarn_cos_sin  # noqa: E402

DUMPS = os.path.join(os.path.dirname(__file__),
                     "../../../../demo/qwen3/dflash_correctness/dumps")
NQ, NKV, D = 64, 8, 128


def load_dump(name):
    fn = name.replace("::", "__").replace(".", "_") + ".pt"
    return torch.load(os.path.join(DUMPS, fn))


def gpl(out):
    return 96 if out % 96 == 0 else 64


def main():
    device, dtype = "cuda", torch.bfloat16
    meta = json.load(open(os.path.join(DUMPS, "meta.json")))
    sw = meta["sliding_window"] if meta["layer_types"][0] == "sliding_attention" else 0

    noise = load_dump("in::noise_embedding").to(device, dtype).squeeze(0)
    ctx = load_dump("out::hidden_norm").to(device, dtype).squeeze(0)
    ref_layer = load_dump("out::layers.0").to(device, torch.float32).squeeze(0)
    B, H = noise.shape
    ctx_len = ctx.shape[0]
    T = ctx_len + B
    q_size, kv_size = NQ * D, NKV * D
    I = load_weight("layers.0.mlp.gate_proj.weight").shape[0]

    # weights
    iln_w = load_weight("layers.0.input_layernorm.weight").contiguous()
    q_w = load_weight("layers.0.self_attn.q_proj.weight").contiguous()
    k_w = load_weight("layers.0.self_attn.k_proj.weight").contiguous()
    v_w = load_weight("layers.0.self_attn.v_proj.weight").contiguous()
    o_w = load_weight("layers.0.self_attn.o_proj.weight").contiguous()
    qn = load_weight("layers.0.self_attn.q_norm.weight").contiguous()
    kn = load_weight("layers.0.self_attn.k_norm.weight").contiguous()
    pln_w = load_weight("layers.0.post_attention_layernorm.weight").contiguous()
    gate_w = load_weight("layers.0.mlp.gate_proj.weight").contiguous()
    up_w = load_weight("layers.0.mlp.up_proj.weight").contiguous()
    down_w = load_weight("layers.0.mlp.down_proj.weight").contiguous()

    cos, sin = build_yarn_cos_sin(CKPT, torch.arange(T), device, dtype)
    cos_blk = cos[ctx_len:T].contiguous(); sin_blk = sin[ctx_len:T].contiguous()

    # ---- materialize context K/V in torch (validated refs) -> cache inputs ----
    ctx_k_raw = linear(ctx, k_w)                          # [ctx_len, kv_size]
    ctx_k = dflash_norm_rope(ctx_k_raw.view(ctx_len, NKV, D), kn,
                             cos[:ctx_len], sin[:ctx_len]).view(ctx_len, kv_size).contiguous()
    ctx_v = linear(ctx, v_w).contiguous()                 # [ctx_len, kv_size]

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = B          # only the block is token-batched
    params["max_num_batched_requests"] = 1
    params["use_cutlass_kernel"] = True
    pk = PersistentKernel(**params)
    bd = (256, 1, 1)

    # inputs
    noise_dt = pk.attach_input(noise.contiguous(), name="noise")
    ck = pk.attach_input(ctx_k, name="ctx_k"); cv = pk.attach_input(ctx_v, name="ctx_v")
    ilnw = pk.attach_input(iln_w, name="iln_w")
    qw = pk.attach_input(q_w, name="q_w"); kw = pk.attach_input(k_w, name="k_w")
    vw = pk.attach_input(v_w, name="v_w"); ow = pk.attach_input(o_w, name="o_w")
    qnw = pk.attach_input(qn, name="qn"); knw = pk.attach_input(kn, name="kn")
    plnw = pk.attach_input(pln_w, name="pln_w")
    cosB = pk.attach_input(cos_blk, name="cosB"); sinB = pk.attach_input(sin_blk, name="sinB")
    gw = pk.attach_input(gate_w, name="gate_w"); uw = pk.attach_input(up_w, name="up_w")
    gu = pk.shuffle_tensors(inputs=[gw, uw], shuffled_dim=0,
                            num_groups=grid_for_rmsnorm_linear_layer(2 * I) // 2,
                            name="gateup_w")
    dw = pk.attach_input(down_w, name="down_w")

    # intermediates (new_tensor)
    h = pk.new_tensor((B, H), name="h")
    q_raw = pk.new_tensor((B, q_size), name="q_raw"); Q = pk.new_tensor((B, q_size), name="Q")
    bk_raw = pk.new_tensor((B, kv_size), name="bk_raw"); BK = pk.new_tensor((B, kv_size), name="BK")
    bv = pk.new_tensor((B, kv_size), name="bv")
    attn = pk.new_tensor((B, q_size), name="attn")
    a = pk.new_tensor((B, H), name="a")
    h2 = pk.new_tensor((B, H), name="h2"); h3 = pk.new_tensor((B, H), name="h3")
    mlp_mid = pk.new_tensor((B, 2 * I), name="mlp_mid"); silu_out = pk.new_tensor((B, I), name="silu_out")
    m = pk.new_tensor((B, H), name="m")
    out = torch.zeros(B, H, dtype=dtype, device=device)
    out_dt = pk.attach_input(out, name="layer_out")

    # input_layernorm
    pk.rmsnorm_layer(input=noise_dt, weight=ilnw, output=h, grid_dim=(B, 1, 1), block_dim=bd)
    # block q/k/v
    pk.linear_layer(input=h, weight=qw, output=q_raw, grid_dim=(gpl(q_size), 1, 1), block_dim=bd)
    pk.dflash_norm_rope_layer(x=q_raw, weight=qnw, cos=cosB, sin=sinB, output=Q,
                              grid_dim=(1, 1, 1), block_dim=bd, head_dim=D)
    pk.linear_layer(input=h, weight=kw, output=bk_raw, grid_dim=(gpl(kv_size), 1, 1), block_dim=bd)
    pk.dflash_norm_rope_layer(x=bk_raw, weight=knw, cos=cosB, sin=sinB, output=BK,
                              grid_dim=(1, 1, 1), block_dim=bd, head_dim=D)
    pk.linear_layer(input=h, weight=vw, output=bv, grid_dim=(gpl(kv_size), 1, 1), block_dim=bd)
    # attention (split ctx/block) + o_proj
    pk.dflash_attention_layer(q=Q, ctx_k=ck, ctx_v=cv, blk_k=BK, blk_v=bv, output=attn,
                              grid_dim=(1, 1, 1), block_dim=bd, sliding_window=sw, head_dim=D)
    pk.linear_layer(input=attn, weight=ow, output=a, grid_dim=(gpl(H), 1, 1), block_dim=bd)
    # h2 = r1 + a
    pk.elementwise_add_layer(input_a=noise_dt, input_b=a, output=h2, grid_dim=(B, 1, 1), block_dim=bd)
    # post_attn_norm + MLP + final residual
    pk.rmsnorm_layer(input=h2, weight=plnw, output=h3, grid_dim=(B, 1, 1), block_dim=bd)
    gut = grid_for_rmsnorm_linear_layer(2 * I)
    pk.linear_layer(input=h3, weight=gu, output=mlp_mid, grid_dim=(gut, 1, 1), block_dim=bd)
    pk.silu_mul_layer(input=mlp_mid, output=silu_out, grid_dim=(gut // 2, 1, 1), block_dim=bd)
    pk.linear_layer(input=silu_out, weight=dw, output=m, grid_dim=(gpl(H), 1, 1), block_dim=bd)
    pk.elementwise_add_layer(input_a=h2, input_b=m, output=out_dt, grid_dim=(B, 1, 1), block_dim=bd)

    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()

    err = (out.float() - ref_layer).abs().max().item()
    rel = err / ref_layer.abs().max().item()
    print(f"ctx_len={ctx_len} B={B} sw={sw}")
    print(f"[MPK full-layer vs dump out::layers.0] maxerr {err:.4f} relmax {rel:.5f} "
          f"(refmax {ref_layer.abs().max().item():.2f})")
    ok = rel < 0.03
    print("PASSED" if ok else "FAILED")
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
