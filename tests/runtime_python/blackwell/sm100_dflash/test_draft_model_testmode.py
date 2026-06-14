"""PB->PC: full L=6 DFlash draft model forward via MPK test-mode vs 6-layer dump.

Stacks 6 DecoderLayers in one megakernel (mbt=B; per-layer context-KV materialized
in torch as cache inputs), then final norm. Aligns final_hidden to dumps6/final_hidden.
At ctx_len=16 the sliding window is inactive (T<2048), so all layers are full non-causal
(matching the reference dump's all-zero mask).

Run: CUDA_VISIBLE_DEVICES=2 python test_draft_model_testmode.py
"""
import json
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.models.utils import grid_for_rmsnorm_linear_layer

sys.path.insert(0, os.path.dirname(__file__))
from pytorch_reference import (load_weight, linear, dflash_norm_rope, CKPT)  # noqa
from verify_attention_ref import build_yarn_cos_sin  # noqa: E402

DUMPS6 = os.path.join(os.path.dirname(__file__),
                      "../../../../demo/qwen3/dflash_correctness/dumps6")
NQ, NKV, D = 64, 8, 128
L = 6


def load_dump(name):
    fn = name.replace("::", "__").replace(".", "_") + ".pt"
    return torch.load(os.path.join(DUMPS6, fn))


def gpl(out):
    return 96 if out % 96 == 0 else 64


def main():
    device, dtype = "cuda", torch.bfloat16
    noise = load_dump("in::noise_embedding").to(device, dtype).squeeze(0)
    ctx = load_dump("out::hidden_norm").to(device, dtype).squeeze(0)
    ref_final = load_dump("final_hidden").to(device, torch.float32).squeeze(0)
    B, H = noise.shape
    ctx_len = ctx.shape[0]
    T = ctx_len + B
    q_size, kv_size = NQ * D, NKV * D
    I = load_weight("layers.0.mlp.gate_proj.weight").shape[0]

    cos, sin = build_yarn_cos_sin(CKPT, torch.arange(T), device, dtype)
    cos_blk = cos[ctx_len:T].contiguous(); sin_blk = sin[ctx_len:T].contiguous()
    norm_w = load_weight("norm.weight").contiguous()

    # per-layer weights + torch-materialized context K/V
    W = []
    for i in range(L):
        p = f"layers.{i}."
        kn = load_weight(p + "self_attn.k_norm.weight").contiguous()
        kw = load_weight(p + "self_attn.k_proj.weight").contiguous()
        vw = load_weight(p + "self_attn.v_proj.weight").contiguous()
        ck = dflash_norm_rope(linear(ctx, kw).view(ctx_len, NKV, D), kn,
                              cos[:ctx_len], sin[:ctx_len]).view(ctx_len, kv_size).contiguous()
        cv = linear(ctx, vw).contiguous()
        W.append(dict(
            iln=load_weight(p + "input_layernorm.weight").contiguous(),
            q=load_weight(p + "self_attn.q_proj.weight").contiguous(),
            k=kw, v=vw, o=load_weight(p + "self_attn.o_proj.weight").contiguous(),
            qn=load_weight(p + "self_attn.q_norm.weight").contiguous(), kn=kn,
            pln=load_weight(p + "post_attention_layernorm.weight").contiguous(),
            gate=load_weight(p + "mlp.gate_proj.weight").contiguous(),
            up=load_weight(p + "mlp.up_proj.weight").contiguous(),
            down=load_weight(p + "mlp.down_proj.weight").contiguous(),
            ck=ck, cv=cv))

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = B
    params["max_num_batched_requests"] = 1
    params["use_cutlass_kernel"] = True
    pk = PersistentKernel(**params)
    bd = (256, 1, 1)

    cosB = pk.attach_input(cos_blk, name="cosB"); sinB = pk.attach_input(sin_blk, name="sinB")
    gut = grid_for_rmsnorm_linear_layer(2 * I)

    hidden = pk.attach_input(noise.contiguous(), name="noise")  # hidden_0
    for i, w in enumerate(W):
        a = lambda t, nm: pk.attach_input(t, name=f"L{i}_{nm}")
        iln = a(w["iln"], "iln"); qw = a(w["q"], "q"); kw = a(w["k"], "k")
        vw = a(w["v"], "v"); ow = a(w["o"], "o"); qnw = a(w["qn"], "qn")
        knw = a(w["kn"], "kn"); plnw = a(w["pln"], "pln")
        gw = a(w["gate"], "gate"); uw = a(w["up"], "up"); dw = a(w["down"], "down")
        ck = a(w["ck"], "ck"); cv = a(w["cv"], "cv")
        gu = pk.shuffle_tensors(inputs=[gw, uw], shuffled_dim=0,
                                num_groups=gut // 2, name=f"L{i}_gateup")

        h = pk.new_tensor((B, H), name=f"L{i}_h")
        q_raw = pk.new_tensor((B, q_size), name=f"L{i}_qraw"); Q = pk.new_tensor((B, q_size), name=f"L{i}_Q")
        bk_raw = pk.new_tensor((B, kv_size), name=f"L{i}_bkraw"); BK = pk.new_tensor((B, kv_size), name=f"L{i}_BK")
        bv = pk.new_tensor((B, kv_size), name=f"L{i}_bv")
        attn = pk.new_tensor((B, q_size), name=f"L{i}_attn"); aout = pk.new_tensor((B, H), name=f"L{i}_a")
        h2 = pk.new_tensor((B, H), name=f"L{i}_h2"); h3 = pk.new_tensor((B, H), name=f"L{i}_h3")
        mid = pk.new_tensor((B, 2 * I), name=f"L{i}_mid"); su = pk.new_tensor((B, I), name=f"L{i}_su")
        m = pk.new_tensor((B, H), name=f"L{i}_m")
        nxt = pk.new_tensor((B, H), name=f"L{i}_out")

        pk.rmsnorm_layer(input=hidden, weight=iln, output=h, grid_dim=(B, 1, 1), block_dim=bd)
        pk.linear_layer(input=h, weight=qw, output=q_raw, grid_dim=(gpl(q_size), 1, 1), block_dim=bd)
        pk.dflash_norm_rope_layer(x=q_raw, weight=qnw, cos=cosB, sin=sinB, output=Q, grid_dim=(1, 1, 1), block_dim=bd, head_dim=D)
        pk.linear_layer(input=h, weight=kw, output=bk_raw, grid_dim=(gpl(kv_size), 1, 1), block_dim=bd)
        pk.dflash_norm_rope_layer(x=bk_raw, weight=knw, cos=cosB, sin=sinB, output=BK, grid_dim=(1, 1, 1), block_dim=bd, head_dim=D)
        pk.linear_layer(input=h, weight=vw, output=bv, grid_dim=(gpl(kv_size), 1, 1), block_dim=bd)
        pk.dflash_attention_layer(q=Q, ctx_k=ck, ctx_v=cv, blk_k=BK, blk_v=bv, output=attn, grid_dim=(1, 1, 1), block_dim=bd, sliding_window=0, head_dim=D)
        pk.linear_layer(input=attn, weight=ow, output=aout, grid_dim=(gpl(H), 1, 1), block_dim=bd)
        pk.elementwise_add_layer(input_a=hidden, input_b=aout, output=h2, grid_dim=(B, 1, 1), block_dim=bd)
        pk.rmsnorm_layer(input=h2, weight=plnw, output=h3, grid_dim=(B, 1, 1), block_dim=bd)
        pk.linear_layer(input=h3, weight=gu, output=mid, grid_dim=(gut, 1, 1), block_dim=bd)
        pk.silu_mul_layer(input=mid, output=su, grid_dim=(gut // 2, 1, 1), block_dim=bd)
        pk.linear_layer(input=su, weight=dw, output=m, grid_dim=(gpl(H), 1, 1), block_dim=bd)
        pk.elementwise_add_layer(input_a=h2, input_b=m, output=nxt, grid_dim=(B, 1, 1), block_dim=bd)
        hidden = nxt

    nw = pk.attach_input(norm_w, name="norm_w")
    final = torch.zeros(B, H, dtype=dtype, device=device)
    final_dt = pk.attach_input(final, name="final")
    pk.rmsnorm_layer(input=hidden, weight=nw, output=final_dt, grid_dim=(B, 1, 1), block_dim=bd)

    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()

    err = (final.float() - ref_final).abs().max().item()
    rel = err / ref_final.abs().max().item()
    print(f"[MPK L={L} model vs dump final_hidden] maxerr {err:.4f} relmax {rel:.5f} "
          f"(refmax {ref_final.abs().max().item():.3f})")
    ok = rel < 0.05
    print("PASSED" if ok else "FAILED")
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
