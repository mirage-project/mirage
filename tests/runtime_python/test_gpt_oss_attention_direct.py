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
"""The full layers' split attention in its direct mode (Blackwell up to 4K positions: sm90_attention_split_layer
full_direct, a unit a request and head group writing the output with the sinks, no merge) against the split + merge on the same
inputs, and both against an fp64 attention with the sinks:
  - the new token's K / V appended to the caches: bit for bit (the same code);
  - the outputs: within bf16 rounding of the fp64 reference (relRMS < 1e-2 a request; a dropped tile or head group, a
    missed slice or a wrong combine is O(0.1 .. 1)).
Shapes: 64 requests (head groups of 4: 128 units), 16 (one head a unit), 8 (one head a unit, 64 units), with first
tokens (no cached tile) and mixed lengths. The layer code is arch-neutral (H100 and B200)."""

import pytest
import torch

from test_gpt_oss_rope_append import D, HK, KVH, PAGE, QH, W, _attention_ref, _case

CASES = {
    64: (8192, [("equal 8K", lambda g: [8191] * 64), ("mixed", lambda g: torch.randint(0, 8192, (64,), generator=g).tolist())]),
    16: (4096, [("short", lambda g: [127 + r for r in range(16)]),
                ("mixed", lambda g: [0] + torch.randint(1, 4096, (15,), generator=g).tolist())]),
    8: (32768, [("mixed 32K", lambda g: [32767, 31, 9000, 20000, 0, 32000, 64, 17000]),
                ("short", lambda g: [100 + r for r in range(8)])]),
}


def _build(tmp_path, tag, bufs, R, direct):
    from mirage.mpk import compiler, plan_gen
    from mirage.mpk.static_megakernel import StaticMegakernel, sm90_flags, sm100_flags
    from mirage.mpk.static_harness import Harness
    mpk = StaticMegakernel(num_gpus=1)
    A = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in bufs.items()}
    mpk._tokens(R)
    mpk.gemm_tile_layer(input=A["h"], weight=A["w_qkv"], output=A["qkv"], combine="store", store_parts=1)
    if direct:
        mpk.sm90_attention_split_layer(A["qkv"], A["k_cache"], A["v_cache"], A["cos"], A["sin"], A["qo_indptr"],
                                       A["kv_indptr"], A["kv_indices"], A["last_page_len"], A["parts"],
                                       num_q_heads=QH, qkv_bias=A["b_qkv"], output=A["out"], sinks=A["sinks"],
                                       full_direct=True)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("R", [64, 16, 8])
def test_gpt_oss_attention_full_direct(tmp_path, R):
    from mirage.mpk.static_megakernel import sm90_attention_parts_size
    dev = "cuda"
    gen = torch.Generator().manual_seed(100 + R)
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
        "kv_indices": torch.zeros(R * MP, dtype=torch.int32, device=dev),
        "last_page_len": torch.zeros(R, dtype=torch.int32, device=dev),
        "k_cache": torch.zeros(R * MP, PAGE, KVH, D, dtype=torch.bfloat16, device=dev),
        "v_cache": torch.zeros(R * MP, PAGE, KVH, D, dtype=torch.bfloat16, device=dev),
        "parts": torch.zeros(sm90_attention_parts_size(R, QH, D, nsm), dtype=torch.float32, device=dev),
        "sinks": torch.randn(QH, generator=gen).to(torch.bfloat16).to(dev),
        "out": torch.zeros(R, QH * D, dtype=torch.bfloat16, device=dev),
    }
    plain = _build(tmp_path, "plain", bufs, R, False)
    direct = _build(tmp_path, "direct", bufs, R, True)
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
            for tag, (sk, h) in (("plain", plain), ("direct", direct)):
                bufs["h"].copy_(hx), bufs["w_qkv"].copy_(wq), bufs["b_qkv"].copy_(bias)
                for k in ("qo_indptr", "kv_indptr", "last_page_len"):
                    bufs[k].copy_(c[k])
                bufs["kv_indices"].zero_()
                bufs["kv_indices"][:len(ids)].copy_(ids)
                bufs["k_cache"].copy_(k0), bufs["v_cache"].copy_(v0)
                bufs["out"].fill_(float("nan")), bufs["parts"].fill_(float("nan"))
                torch.cuda.synchronize()
                h.launch(timeout_s=60)
                torch.cuda.synchronize()
                res[tag] = {k: bufs[k].clone() for k in ("out", "k_cache", "v_cache", "qkv")}
            P, Dr = res["plain"], res["direct"]
            assert torch.equal(P["qkv"], Dr["qkv"]), f"{name}: the GEMM's qkv differs"
            assert torch.equal(P["k_cache"], Dr["k_cache"]) and torch.equal(P["v_cache"], Dr["v_cache"]), \
                f"{name}: the appended K / V differ"
            # q as the tasks make it (bias rounded to bf16, RoPE in fp32 rounded): the fp64 reference's inputs
            x = (P["qkv"].view(R, KVH, W).double() + bias.to(dev).view(KVH, W).double()).to(torch.bfloat16).double()
            p = torch.tensor(cached, device=dev)
            cs, sn = bufs["cos"][p].double()[:, None], bufs["sin"][p].double()[:, None]
            qx = x[:, :, :QH // KVH * D].reshape(R, QH, D)
            q = torch.cat([qx[..., :D // 2] * cs[..., :D // 2] - qx[..., D // 2:] * sn[..., :D // 2],
                           qx[..., D // 2:] * cs[..., D // 2:] + qx[..., :D // 2] * sn[..., D // 2:]], -1)
            q = q.to(torch.bfloat16)
            K = Dr["k_cache"].view(R, MP * PAGE, KVH, D)
            V = Dr["v_cache"].view(R, MP * PAGE, KVH, D)
            ref = _attention_ref(q, K, V, bufs["sinks"].float(), cached)
            a = Dr["out"].view(R, QH, D).double()
            b = P["out"].view(R, QH, D).double()
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), f"{name}: non-finite output"
            ea = ((a - ref).flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1)).max().item()
            eb = ((b - ref).flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1)).max().item()
            print(f"{name}: vs fp64: direct {ea:.2e}, split + merge {eb:.2e}")
            assert ea < 1e-2 and eb < 1e-2, f"{name}: vs fp64 direct {ea:.2e} split {eb:.2e}"
            assert ea <= 1.5 * eb + 1e-3, f"{name}: direct less accurate {ea:.2e} vs {eb:.2e}"
    finally:
        plain[0].finalize()
        direct[0].finalize()
