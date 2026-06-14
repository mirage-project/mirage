"""PB layer alignment: full DFlash attention path via MPK test-mode vs HF dump.

Path (all MPK):  linear(k/v) -> norm_rope(k) -> ; linear(q) -> norm_rope(q) ->
                 dflash_attention -> linear(o)   == out::layers.0.self_attn

Boundary: feed hidden_combined = [ctx(hidden_norm) ++ block(input_layernorm)] and
h_block as precomputed inputs (fc/input_layernorm validated separately). cos/sin
built from the YaRN rotary (verify_attention_ref proved this source).

Run: CUDA_VISIBLE_DEVICES=2 python test_attn_layer_testmode.py
"""
import json
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(__file__))
from pytorch_reference import load_weight, CKPT, EPS  # noqa: E402
from verify_attention_ref import build_yarn_cos_sin  # noqa: E402

DUMPS = os.path.join(os.path.dirname(__file__),
                     "../../../../demo/qwen3/dflash_correctness/dumps")
NQ, NKV, D = 64, 8, 128


def load_dump(name):
    fn = name.replace("::", "__").replace(".", "_") + ".pt"
    return torch.load(os.path.join(DUMPS, fn))


def grid_for_plain_linear(out):
    return 96 if out % 96 == 0 else 64


def main():
    device, dtype = "cuda", torch.bfloat16
    meta = json.load(open(os.path.join(DUMPS, "meta.json")))
    sw = meta["sliding_window"] if meta["layer_types"][0] == "sliding_attention" else 0

    ctx = load_dump("out::hidden_norm").to(device, dtype).squeeze(0)            # [ctx_len,H]
    h_block = load_dump("out::layers.0.input_layernorm").to(device, dtype).squeeze(0)  # [B,H]
    ref_attn = load_dump("out::layers.0.self_attn").to(device, torch.float32).squeeze(0)
    ctx_len, H = ctx.shape
    B = h_block.shape[0]
    T = ctx_len + B
    hidden_combined = torch.cat([ctx, h_block], dim=0).contiguous()            # [T,H]

    cos, sin = build_yarn_cos_sin(CKPT, torch.arange(T), device, dtype)        # [T,d]
    cos = cos.contiguous(); sin = sin.contiguous()
    cos_blk = cos[ctx_len:T].contiguous(); sin_blk = sin[ctx_len:T].contiguous()

    q_w = load_weight("layers.0.self_attn.q_proj.weight").contiguous()
    k_w = load_weight("layers.0.self_attn.k_proj.weight").contiguous()
    v_w = load_weight("layers.0.self_attn.v_proj.weight").contiguous()
    o_w = load_weight("layers.0.self_attn.o_proj.weight").contiguous()
    qn = load_weight("layers.0.self_attn.q_norm.weight").contiguous()
    kn = load_weight("layers.0.self_attn.k_norm.weight").contiguous()
    q_size, kv_size = NQ * D, NKV * D

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = T
    params["max_num_batched_requests"] = 1
    params["use_cutlass_kernel"] = True
    pk = PersistentKernel(**params)

    bd = (256, 1, 1)
    # tensors
    hc = pk.attach_input(hidden_combined, name="hidden_combined")
    hb = pk.attach_input(h_block.contiguous(), name="h_block")
    qw = pk.attach_input(q_w, name="q_w")
    kw = pk.attach_input(k_w, name="k_w")
    vw = pk.attach_input(v_w, name="v_w")
    ow = pk.attach_input(o_w, name="o_w")
    qnw = pk.attach_input(qn, name="qn"); knw = pk.attach_input(kn, name="kn")
    cosA = pk.attach_input(cos, name="cosA"); sinA = pk.attach_input(sin, name="sinA")
    cosB = pk.attach_input(cos_blk, name="cosB"); sinB = pk.attach_input(sin_blk, name="sinB")

    k_raw = pk.attach_input(torch.zeros(T, kv_size, dtype=dtype, device=device), name="k_raw")
    v_raw = pk.attach_input(torch.zeros(T, kv_size, dtype=dtype, device=device), name="v_raw")
    K = pk.attach_input(torch.zeros(T, kv_size, dtype=dtype, device=device), name="K")
    q_raw = pk.attach_input(torch.zeros(B, q_size, dtype=dtype, device=device), name="q_raw")
    Q = pk.attach_input(torch.zeros(B, q_size, dtype=dtype, device=device), name="Q")
    attn = pk.attach_input(torch.zeros(B, q_size, dtype=dtype, device=device), name="attn")
    o_out = torch.zeros(B, H, dtype=dtype, device=device)
    oo = pk.attach_input(o_out, name="o_out")

    # K/V projections over [ctx++block]
    pk.linear_layer(input=hc, weight=kw, output=k_raw,
                    grid_dim=(grid_for_plain_linear(kv_size), 1, 1), block_dim=bd)
    pk.linear_layer(input=hc, weight=vw, output=v_raw,
                    grid_dim=(grid_for_plain_linear(kv_size), 1, 1), block_dim=bd)
    pk.dflash_norm_rope_layer(x=k_raw, weight=knw, cos=cosA, sin=sinA, output=K,
                              grid_dim=(1, 1, 1), block_dim=bd, head_dim=D)
    # Q projection over block
    pk.linear_layer(input=hb, weight=qw, output=q_raw,
                    grid_dim=(grid_for_plain_linear(q_size), 1, 1), block_dim=bd)
    pk.dflash_norm_rope_layer(x=q_raw, weight=qnw, cos=cosB, sin=sinB, output=Q,
                              grid_dim=(1, 1, 1), block_dim=bd, head_dim=D)
    # attention + o_proj
    pk.dflash_attention_layer(q=Q, k=K, v=v_raw, output=attn,
                              grid_dim=(1, 1, 1), block_dim=bd,
                              sliding_window=sw, head_dim=D)
    pk.linear_layer(input=attn, weight=ow, output=oo,
                    grid_dim=(grid_for_plain_linear(H), 1, 1), block_dim=bd)

    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()
    pk.finalize()

    err = (o_out.float() - ref_attn).abs().max().item()
    rel = err / ref_attn.abs().max().item()
    print(f"ctx_len={ctx_len} B={B} sw={sw}")
    print(f"[MPK attn-path vs dump] maxerr {err:.4f} relmax {rel:.4f}")
    ok = rel < 0.03
    print("PASSED" if ok else "FAILED")
    assert ok


if __name__ == "__main__":
    main()
