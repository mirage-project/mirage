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
"""The shipped B200 layer plans (demo/gpt_oss_static/plans/b200) fit the graphs gpt_oss_static.py builds: for every
batch, "sliding", "full" (the direct full attention, no merge node) and the trtllm-gen full layers' halves "full_a"
(the norm, the QKV GEMM, gpt_oss_rope_append) and "full_b" (o_proj, the gate) on 148 SMs (sm_100, FlashInfer's MoE with
the fold, zero tensors at the plan's requests): load_plan takes the file and compiler.compile_plan places every task of
the graph exactly once (schedules only, no build)."""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H, E, TOPK, QH, KVH, D, PAGE, MAX_SEQ, WINDOW = 2880, 128, 4, 64, 8, 64, 64, 4096, 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (the graph's tensors)")
@pytest.mark.parametrize("R", [2, 8, 16, 32, 64])
def test_b200_plans_fit(monkeypatch, tmp_path, R):
    monkeypatch.syspath_prepend(os.path.join(ROOT, "demo/gpt_oss_static"))
    monkeypatch.delitem(sys.modules, "gpt_oss_static", raising=False)
    import gpt_oss_static as g
    from mirage.mpk import compiler
    from mirage.mpk.static_megakernel import StaticMegakernel, sm90_attention_parts_size
    monkeypatch.setattr(g, "CC", 100)
    init = StaticMegakernel.__init__
    monkeypatch.setattr(StaticMegakernel, "__init__", lambda self, *a, **k: init(self, *a, **{"num_sms": 148, **k}))
    T = next(b for b in g.BUCKETS if b >= R)
    monkeypatch.setattr(g, "T", T)
    MP = MAX_SEQ // PAGE
    z = lambda *s, dtype=torch.bfloat16: torch.zeros(*s, dtype=dtype, device="cuda")
    i32 = lambda x: x.to(torch.int32).cuda()
    # Model.layer_tensors' tensors on Blackwell (the plans' node ids follow the graph inputs' count)
    t = {"cos": z(MAX_SEQ, D), "sin": z(MAX_SEQ, D), "qo_indptr": i32(torch.arange(R + 1)),
         "kv_indptr": i32(torch.arange(R + 1)), "kv_indices": z(R * MP, dtype=torch.int32),
         "last_page_len": i32(torch.ones(R)), "num_tokens": i32(torch.full((1,), R)),
         "attn_parts": z(sm90_attention_parts_size(R, QH, D, 148), dtype=torch.float32),
         "h": z(T, H), "qkv32": z(g.QKV_PARTS * T, (QH + 2 * KVH) * D, dtype=torch.float32),
         "qkv": z(T, (QH + 2 * KVH) * D), "attn": z(T, QH * D), "o32": z(g.O_PARTS * T, H, dtype=torch.float32),
         "x1": z(T, H), "h2": z(T, 3072 * 33 // 32, dtype=torch.uint8), "topk_w": z(T, TOPK, dtype=torch.float32),
         "routing": z(E + TOPK, T, dtype=torch.int32), "eo": z(T, TOPK, H), "act": z(1, dtype=torch.float16),
         "moe_y": z(T, H), "x_res": z(T, H),
         "ln1": z(H), "ln2": z(H), "w_qkv": z((QH + 2 * KVH) * D, g.QKV_K), "b_qkv": z((QH + 2 * KVH) * D),
         "w_o": z(H, QH * D), "b_o": z(H), "sinks": z(KVH, QH // KVH), "w_router": z(E, H), "b_router": z(E),
         "w13": z(1, dtype=torch.uint8), "b13": z(1), "w2": z(1, dtype=torch.uint8), "b2": z(1),
         "k_cache": z(R * MP, PAGE, KVH, D), "v_cache": z(R * MP, PAGE, KVH, D),
         "x_in": z(T, H), "x_out": z(T, H), "moe_tab": z(4, dtype=torch.int64)}
    t_trt = {**t, "q_rope": z(T, QH * D), "seq_lens": z(R, dtype=torch.int32)}   # Model.trt
    m = type("Cfg", (), dict(eps=1e-5, QH=QH, KVH=KVH, E=E, TOPK=TOPK, limit=7.0, max_seq=MAX_SEQ, R=R, window=WINDOW))
    plans = os.path.join(ROOT, "demo/gpt_oss_static/plans/b200")
    names = {}
    for kind, tensors, window, part in (("sliding", t, WINDOW, None), ("full", t, 0, None), ("full_a", t_trt, 0, "a"),
                                        ("full_b", t_trt, 0, "b")):
        mpk = StaticMegakernel(num_gpus=1, target_cc=100)
        g.build_layer(m, mpk, tensors, window, part)
        plan = g.load_plan(mpk, g.plan_path(m, plans, kind))
        assert plan is not None, (kind, R)
        compiler.compile_plan(mpk, plan, str(tmp_path / kind))   # every task once, on a valid SM
        names[kind] = [n.name for n in compiler.graph_nodes(mpk)]
    assert "sm90_attention_merge" not in names["full"] and "sm90_attention_split" in names["full"]
    assert names["full_a"][-1] == "gpt_oss_rope_append" and names["full_b"][-1] == "sm90_gate"
