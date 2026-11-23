# Copyright 2026 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""gpt_oss_rope_append (tasks/gpt_oss_rope_append.cuh, the full layers past 4K positions on Blackwell) and
FlashInfer's decode attention after it, against the layer kernel's own path (sm90_attention_split + sm90_attention_merge) on the same inputs:
  - the new token's K / V appended to the caches: bit for bit;
  - seq_lens: each request's positions with the new token;
  - q: the bias and RoPE of an fp64 replica, within one bf16 rounding step;
  - FlashInfer's output on q, the caches and seq_lens (trtllm-gen on B200, xqa on H100): within bf16 rounding of the
    fp64 attention (with the sinks), as is the split's.
The layer code is arch-neutral (H100 and B200). FlashInfer is needed (skipped without it)."""
import math

import pytest
import torch

QH, KVH, D, PAGE = 64, 8, 64, 64
HK = 256
W = (QH // KVH + 2) * D

flashinfer = pytest.importorskip("flashinfer")


def _case(gen, R, cached):
    pages = [math.ceil((c + 1) / PAGE) for c in cached]
    kv_indptr = torch.tensor([0] + list(torch.tensor(pages).cumsum(0)), dtype=torch.int32)
    last = torch.tensor([(c + 1) - (p - 1) * PAGE for c, p in zip(cached, pages)], dtype=torch.int32)
    return dict(kv_indptr=kv_indptr, last_page_len=last, qo_indptr=torch.arange(R + 1, dtype=torch.int32))


def _qkv_ref(qkv, bias, cos, sin, cached):
    """fp64 q, k, v of each request's token with the bias (rounded to bf16) and RoPE: q [R, QH, D], k, v [R, KVH, D]."""
    R = qkv.shape[0]
    x = (qkv.view(R, KVH, W).double() + bias.view(KVH, W).double()).to(torch.bfloat16).double()
    p = torch.tensor(cached, device=qkv.device)
    c, s = cos[p].double()[:, None], sin[p].double()[:, None]       # [R, 1, D]
    rot = lambda v: torch.cat([v[..., :D // 2] * c[..., :D // 2] - v[..., D // 2:] * s[..., :D // 2],
                               v[..., D // 2:] * c[..., D // 2:] + v[..., :D // 2] * s[..., D // 2:]], -1)
    q = rot(x[:, :, :QH // KVH * D].reshape(R, QH, D))   # query head h QPK + i: KV head h's i-th
    k = rot(x[:, :, QH // KVH * D:(QH // KVH + 1) * D])
    v = x[:, :, (QH // KVH + 1) * D:]
    return q, k, v


def _attention_ref(q, K, V, sinks, cached):
    """fp64 decode attention with the sinks: q [R, QH, D] (bf16 values), K / V [R, S, KVH, D] -> [R, QH, D]."""
    R = q.shape[0]
    out = torch.zeros(R, QH, D, dtype=torch.float64, device=q.device)
    for r in range(R):
        n = cached[r] + 1
        kk = K[r, :n].double().repeat_interleave(QH // KVH, 1)
        vv = V[r, :n].double().repeat_interleave(QH // KVH, 1)
        s = torch.einsum("hd,nhd->hn", q[r].double(), kk) / math.sqrt(D)
        s = torch.cat([s, sinks.double()[:, None]], 1)
        out[r] = torch.einsum("hn,nhd->hd", torch.softmax(s, 1)[:, :n], vv)
    return out


def _build(tmp_path, tag, bufs, R, trtllm):
    from mirage.mpk import compiler, plan_gen
    from mirage.mpk.static_megakernel import StaticMegakernel, sm90_flags, sm100_flags
    from mirage.mpk.static_harness import Harness
    mpk = StaticMegakernel(num_gpus=1)
    A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in bufs.items()}
    mpk._tokens(R)
    mpk.gemm_tile_layer(input=A["h"], weight=A["w_qkv"], output=A["qkv"], combine="store", store_parts=1)
    if trtllm:
        mpk.gpt_oss_rope_append_layer(A["qkv"], A["b_qkv"], A["cos"], A["sin"], A["qo_indptr"], A["kv_indptr"],
                                      A["kv_indices"], A["last_page_len"], A["k_cache"], A["v_cache"], A["seq_lens"],
                                      A["q_rope"], num_q_heads=QH)
    else:
        mpk.sm90_attention_split_layer(A["qkv"], A["k_cache"], A["v_cache"], A["cos"], A["sin"], A["qo_indptr"],
                                       A["kv_indptr"], A["kv_indices"], A["last_page_len"], A["parts"],
                                       num_q_heads=QH, qkv_bias=A["b_qkv"])
        mpk.sm90_attention_merge_layer(A["parts"], A["qo_indptr"], A["out"], num_q_heads=QH, sinks=A["sinks"])
    d = tmp_path / tag
    paths, _ = compiler.compile_plan(mpk, plan_gen.round_robin_plan(mpk), str(d / "schedules"))
    fl = sm100_flags() if torch.cuda.get_device_capability()[0] == 10 else sm90_flags()
    sk = mpk.compile_static(paths, gpu_tensors=[bufs], out_dir=str(d / "build"), extra_flags=fl)
    return sk, Harness(sk)


CASES = {   # R: (max cached, cases)
    64: (8192, [("equal 8K", lambda g: [8191] * 64), ("mixed", lambda g: torch.randint(0, 8192, (64,), generator=g).tolist())]),
    8: (32768, [("mixed 32K", lambda g: [32767, 31, 9000, 20000, 0, 32000, 64, 17000]),
                ("short", lambda g: [100 + r for r in range(8)])]),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("R", [64, 8])
def test_gpt_oss_rope_append_then_flashinfer(tmp_path, R):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "demo", "gpt_oss_static"))
    from trtllm_attn import TrtllmAttention
    from mirage.mpk.static_megakernel import sm90_attention_parts_size
    dev = "cuda"
    gen = torch.Generator().manual_seed(R)
    max_cached, cases = CASES[R]
    max_seq = (max_cached + 1 + PAGE - 1) // PAGE * PAGE
    MP = max_seq // PAGE
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    bufs = {
        "h": torch.zeros(R, HK, dtype=torch.bfloat16, device=dev),
        "w_qkv": torch.zeros(KVH * W, HK, dtype=torch.bfloat16, device=dev),
        "qkv": torch.zeros(R, KVH * W, dtype=torch.float32, device=dev),
        "b_qkv": torch.zeros(KVH * W, dtype=torch.bfloat16, device=dev),
        "cos": (torch.rand(max_seq, D, generator=gen) * 2 - 1).to(torch.bfloat16).to(dev),
        "sin": (torch.rand(max_seq, D, generator=gen) * 2 - 1).to(torch.bfloat16).to(dev),
        "qo_indptr": torch.zeros(R + 1, dtype=torch.int32, device=dev),
        "kv_indptr": torch.zeros(R + 1, dtype=torch.int32, device=dev),
        # request r's page j is r MP + j (the demo's layout: FlashInfer's constant block table)
        "kv_indices": torch.zeros(R * MP, dtype=torch.int32, device=dev),
        "last_page_len": torch.zeros(R, dtype=torch.int32, device=dev),
        "k_cache": torch.zeros(R * MP, PAGE, KVH, D, dtype=torch.bfloat16, device=dev),
        "v_cache": torch.zeros(R * MP, PAGE, KVH, D, dtype=torch.bfloat16, device=dev),
        "parts": torch.zeros(sm90_attention_parts_size(R, QH, D, nsm), dtype=torch.float32, device=dev),
        "sinks": torch.randn(QH, generator=gen).to(torch.bfloat16).to(dev),
        "out": torch.zeros(R, QH * D, dtype=torch.bfloat16, device=dev),
        "seq_lens": torch.zeros(R, dtype=torch.int32, device=dev),
        "q_rope": torch.zeros(R, QH * D, dtype=torch.bfloat16, device=dev),
        "attn": torch.zeros(R, QH * D, dtype=torch.bfloat16, device=dev),
    }
    plain = _build(tmp_path, "plain", bufs, R, False)
    split = _build(tmp_path, "trtllm", bufs, R, True)
    att = TrtllmAttention(R, QH, KVH, D, PAGE, max_seq)
    sinks32 = TrtllmAttention.sinks_of(bufs["sinks"].view(KVH, QH // KVH))
    try:
        for name, mk in cases:
            cached = mk(gen)
            c = _case(gen, R, cached)
            hx = torch.randn(R, HK, generator=gen).to(torch.bfloat16)
            wq = (torch.randn(KVH * W, HK, generator=gen) / 16).to(torch.bfloat16)
            bias = (torch.randn(KVH * W, generator=gen) * 0.1).to(torch.bfloat16)
            k0 = torch.randn(R * MP, PAGE, KVH, D, generator=gen).to(torch.bfloat16)
            v0 = torch.randn(R * MP, PAGE, KVH, D, generator=gen).to(torch.bfloat16)
            ids = torch.cat([torch.arange(r * MP, r * MP + int(c["kv_indptr"][r + 1] - c["kv_indptr"][r]))
                             for r in range(R)]).to(torch.int32)
            res = {}
            for tag, (sk, h) in (("plain", plain), ("trtllm", split)):
                bufs["h"].copy_(hx), bufs["w_qkv"].copy_(wq), bufs["b_qkv"].copy_(bias)
                for k in ("qo_indptr", "kv_indptr", "last_page_len"):
                    bufs[k].copy_(c[k])
                bufs["kv_indices"].zero_()
                bufs["kv_indices"][:len(ids)].copy_(ids)
                bufs["k_cache"].copy_(k0), bufs["v_cache"].copy_(v0)
                for k in ("out", "q_rope", "attn"):
                    bufs[k].fill_(float("nan"))
                bufs["seq_lens"].fill_(-1)
                torch.cuda.synchronize()
                h.launch(timeout_s=60)
                torch.cuda.synchronize()
                if tag == "trtllm":
                    att.launch(bufs["q_rope"], bufs["k_cache"], bufs["v_cache"], bufs["seq_lens"], bufs["attn"], sinks32)
                    torch.cuda.synchronize()
                res[tag] = {k: bufs[k].clone() for k in ("out", "attn", "q_rope", "seq_lens", "k_cache", "v_cache", "qkv")}
            P, Tr = res["plain"], res["trtllm"]
            assert torch.equal(P["qkv"], Tr["qkv"]), f"{name}: the GEMM's qkv differs"
            assert torch.equal(P["k_cache"], Tr["k_cache"]) and torch.equal(P["v_cache"], Tr["v_cache"]), \
                f"{name}: the appended K / V differ from the split task's"
            assert Tr["seq_lens"].tolist() == [x + 1 for x in cached], f"{name}: seq_lens {Tr['seq_lens'].tolist()}"
            q, kn, vn = _qkv_ref(P["qkv"], bias.to(dev), bufs["cos"], bufs["sin"], cached)
            qk = Tr["q_rope"].view(R, QH, D)
            assert (qk.double() - q.to(torch.bfloat16).double()).abs().max() <= q.abs().max() * 2 ** -7, f"{name}: q"
            exact = (qk == q.to(torch.bfloat16)).double().mean().item()
            assert exact > 0.99, f"{name}: q equals the fp64 replica's rounding for {exact:.4f} only"
            # the caches with the new token (both paths appended it, bit for bit: checked above)
            K = Tr["k_cache"].view(R, MP * PAGE, KVH, D)
            V = Tr["v_cache"].view(R, MP * PAGE, KVH, D)
            ref = _attention_ref(qk, K, V, bufs["sinks"].float(), cached)
            a = Tr["attn"].view(R, QH, D).double()
            b = P["out"].view(R, QH, D).double()
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), f"{name}: non-finite output"
            ea = ((a - ref).flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1)).max().item()
            eb = ((b - ref).flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1)).max().item()
            print(f"{name}: q exact {exact:.4f}; vs fp64: flashinfer {ea:.2e}, split {eb:.2e}")
            assert ea < 1e-2 and eb < 1e-2, f"{name}: vs fp64 flashinfer {ea:.2e} split {eb:.2e}"
    finally:
        plain[0].finalize()
        split[0].finalize()
