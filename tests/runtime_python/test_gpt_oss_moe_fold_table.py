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

"""GPT-OSS fold mode's host side (demo/gpt_oss_static/flashinfer_moe.py FlashinferMoe.set_fold): the table of the
unfinalized MoE outputs each layer's gpt_oss_moe_norm reads. Eager steps write row i + 1 after layer i's call; a CUDA
graph's replay restores its own capture's addresses whatever ran before it (eager steps, another graph). FlashInfer is a
stub returning fresh [gemm2_output, expert_weights, expanded_idx_to_permuted_idx] tensors as its launcher does."""
import os
import sys
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
L, R, E, TOPK, H = 4, 3, 128, 4, 2880


def _stub_flashinfer(calls):
    def moe(**kw):
        calls.append(kw)
        if kw["do_finalize"]:
            kw["output"].fill_(1.0)
            return [kw["output"]]
        g2 = torch.empty(64, 3072, dtype=torch.bfloat16, device="cuda")
        g2.fill_(float(len(calls)))
        w = torch.empty(R, TOPK, dtype=torch.bfloat16, device="cuda").fill_(0.5)
        idx = torch.arange(R * TOPK, dtype=torch.int32, device="cuda")
        return [g2, w, idx]

    fi = types.ModuleType("flashinfer")
    fi.trtllm_fp4_block_scale_moe = moe
    fi.trtllm_fp4_block_scale_routed_moe = moe
    fp4 = types.ModuleType("flashinfer.fp4_quantization")
    fp4.nvfp4_block_scale_interleave = lambda x: x
    fm = types.ModuleType("flashinfer.fused_moe")
    core = types.ModuleType("flashinfer.fused_moe.core")
    core.RoutingMethodType = types.SimpleNamespace(Renormalize=1)
    core.get_w2_permute_indices_with_cache = lambda *a, **k: None
    return {"flashinfer": fi, "flashinfer.fp4_quantization": fp4, "flashinfer.fused_moe": fm,
            "flashinfer.fused_moe.core": core}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fold_table_eager_and_graph_replays(monkeypatch):
    calls = []
    for k, v in _stub_flashinfer(calls).items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.syspath_prepend(os.path.join(ROOT, "demo/gpt_oss_static"))
    monkeypatch.delitem(sys.modules, "flashinfer_moe", raising=False)
    import flashinfer_moe

    y = torch.zeros(8, H, dtype=torch.bfloat16, device="cuda")
    moe = flashinfer_moe.FlashinferMoe(R, E, TOPK, H, 7.0, y, logits=True)
    table = torch.full((L + 1, 4), -7, dtype=torch.int64, device="cuda")
    table[0].zero_()
    moe.set_fold(table)
    routing = torch.zeros(E + TOPK, 8, dtype=torch.int32, device="cuda")
    h2 = torch.zeros(8, 3072 * 33 // 32, dtype=torch.uint8, device="cuda")

    def step():
        for i in range(L):
            if i == 0:
                moe.begin_step()
            moe.launch(routing, None, h2, {k: None for k in ("fw13", "fs13", "fb13", "fw2", "fs2", "fb2")}, layer=i)

    def rows_of_keep():
        return torch.tensor([[0, 0, 0, 0]] + [[g.data_ptr(), i.data_ptr(), w.data_ptr(), R]
                                              for g, w, i in (moe.keep[j] for j in range(L))], dtype=torch.int64)

    step()   # eager
    torch.cuda.synchronize()
    assert torch.equal(table.cpu(), rows_of_keep())
    assert all(not c["do_finalize"] and c["output"] is moe.y for c in calls)

    graphs, want = [], []
    for _ in range(2):   # two captures (as a sweep's contexts), each its own addresses
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        graphs.append(g)
        want.append(rows_of_keep())
    step()   # an eager step between: the table holds its addresses
    torch.cuda.synchronize()
    eager = rows_of_keep()
    assert torch.equal(table.cpu(), eager)
    for k in (0, 1, 0):
        table.fill_(-1)
        graphs[k].replay()
        torch.cuda.synchronize()
        assert torch.equal(table.cpu(), want[k]), f"graph {k}'s replay left another table"

    # verify_fold's override: the finalized call into its own output, no table write
    before = table.clone()
    ref = torch.zeros_like(moe.y)
    moe.launch(routing, None, h2, {k: None for k in ("fw13", "fs13", "fb13", "fw2", "fs2", "fb2")},
               do_finalize=True, output=ref)
    torch.cuda.synchronize()
    assert calls[-1]["do_finalize"] and calls[-1]["output"] is ref and torch.equal(table, before)
