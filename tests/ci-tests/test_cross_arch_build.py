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

"""Build kernels for every supported architecture on whatever GPU CI has,
without running them: MPK's paged attention, and the GPT-OSS layers of the
static megakernel for Hopper and Blackwell.

Codegen and nvcc do not need the target GPU, so this catches template
argument mismatches and other build breaks on paths that have no hardware in
CI (sm90 today).
"""

import importlib.util
import os
import subprocess
import sys

import pytest
import torch

import mirage as mi
from mirage.mpk.persistent_kernel import PersistentKernel

NUM_KV_HEADS = 1
NUM_Q_HEADS = 8
MAX_SEQ_LENGTH = 512
NUM_KV_CHUNKS = 2
NUM_TOKENS = 8

# (head_dim, page_size) per target. sm90 uses PAGE_SIZE > HEAD_DIM so that a
# page stride landing in the ROTARY_DIM slot fails the kernel's static_assert
# instead of compiling. sm100 split-KV needs pages of 128 rows.
SHAPES = {90: (64, 128), 100: (128, 128)}


def _make_pk(target_cc):
    _, page_size = SHAPES[target_cc]
    workers, schedulers = mi.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=workers,
        num_local_schedulers=schedulers,
        max_seq_length=MAX_SEQ_LENGTH,
        max_num_batched_requests=1,
        max_num_batched_tokens=NUM_TOKENS,
        max_num_pages=MAX_SEQ_LENGTH // page_size,
        page_size=page_size,
    )
    pk = PersistentKernel(**params)
    # Layers pick their kernel from target_cc, so set it before adding any.
    pk.target_cc = target_cc
    return pk


def _attention_inputs(pk, head_dim, page_size):
    dtype = torch.bfloat16
    qkv = torch.zeros(NUM_TOKENS, (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * head_dim,
                      dtype=dtype, device="cuda")
    cache = torch.zeros(pk.max_num_pages, page_size, NUM_KV_HEADS, head_dim,
                        dtype=dtype, device="cuda")
    norm = torch.ones(head_dim, dtype=dtype, device="cuda")
    rope = torch.ones(MAX_SEQ_LENGTH, head_dim, dtype=dtype, device="cuda")
    return dict(
        input=pk.attach_input(qkv, name="qkv"),
        k_cache=pk.attach_input(cache, name="k_cache"),
        v_cache=pk.attach_input(cache.clone(), name="v_cache"),
        q_norm=pk.attach_input(norm, name="q_norm"),
        k_norm=pk.attach_input(norm.clone(), name="k_norm"),
        cos_pos_embed=pk.attach_input(rope, name="cos"),
        sin_pos_embed=pk.attach_input(rope.clone(), name="sin"),
    )


def _build_paged_attention(target_cc):
    head_dim, page_size = SHAPES[target_cc]
    pk = _make_pk(target_cc)
    out = torch.zeros(NUM_TOKENS, NUM_Q_HEADS * head_dim,
                      dtype=torch.bfloat16, device="cuda")
    pk.paged_attention_layer(
        **_attention_inputs(pk, head_dim, page_size),
        output=pk.attach_input(out, name="out"),
        grid_dim=(1, NUM_KV_HEADS, 1),
        block_dim=(128, 1, 1),
    )
    pk.compile(build_only=True)


def _build_paged_attention_split_kv(target_cc):
    head_dim, page_size = SHAPES[target_cc]
    pk = _make_pk(target_cc)
    qo_per_kv = NUM_Q_HEADS // NUM_KV_HEADS
    lse = pk.new_tensor(
        dims=(NUM_TOKENS, NUM_KV_CHUNKS * qo_per_kv, NUM_KV_HEADS),
        strides=(NUM_KV_CHUNKS * NUM_Q_HEADS, 1, NUM_KV_CHUNKS * qo_per_kv),
        dtype=mi.float32, name="lse", io_category="cuda_tensor")
    out_tmp = pk.new_tensor(
        dims=(NUM_TOKENS, NUM_KV_CHUNKS * qo_per_kv * head_dim, NUM_KV_HEADS),
        strides=(NUM_KV_CHUNKS * NUM_Q_HEADS * head_dim, 1,
                 NUM_KV_CHUNKS * qo_per_kv * head_dim),
        dtype=mi.bfloat16, name="out_tmp", io_category="cuda_tensor")
    out = torch.zeros(NUM_TOKENS, NUM_Q_HEADS * head_dim,
                      dtype=torch.bfloat16, device="cuda")
    pk.paged_attention_split_kv_layer(
        **_attention_inputs(pk, head_dim, page_size),
        lse=lse,
        output=out_tmp,
        attention_params=(NUM_Q_HEADS, NUM_KV_CHUNKS),
        grid_dim=(1, NUM_KV_HEADS, NUM_KV_CHUNKS),
        block_dim=(128, 1, 1),
    )
    pk.paged_attention_split_kv_merge_layer(
        lse=lse,
        output_tmp=out_tmp,
        output=pk.attach_input(out, name="out"),
        attention_params=(NUM_Q_HEADS, head_dim),
        grid_dim=(1, NUM_KV_HEADS, 1),
        block_dim=(128, 1, 1),
    )
    pk.compile(build_only=True)


def _gpt_oss_demo():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    demo = os.path.join(root, "demo", "gpt_oss_static")
    sys.path.insert(0, demo)
    spec = importlib.util.spec_from_file_location(
        "gpt_oss_static", os.path.join(demo, "gpt_oss_static.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, demo


def _build_gpt_oss_static(kind, cc=90, native=False):
    """One GPT-OSS layer (kind: sliding or full attention) of the static
    megakernel, its tensors zeros, built with the demo's serving flags: for
    sm_90a (20B, the shipped H100 plan on 132 SMs, else round robin) or sm_100a
    (148 SMs, round robin; native: 120B with Blackwell's MXFP8 x MXFP4 experts
    in the layer kernel)."""
    from mirage.mpk import compiler, plan_gen
    from mirage.mpk.static_megakernel import (StaticMegakernel,
                                              sm90_attention_parts_size)

    g, demo = _gpt_oss_demo()
    g.CC = cc
    nsm = 148 if cc == 100 else 132
    H, I, E, TOPK, QH, KVH, D, MAX_SEQ = 2880, 2880, 128 if native else 32, 4, 64, 8, 64, 1024
    T, PAGE, R13, R2 = g.T, g.PAGE, 128, 64
    kt = (H + 127) // 128

    def z(*shape, dtype=torch.bfloat16):
        return torch.zeros(*shape, dtype=dtype, device="cuda")

    t = {
        "cos": z(MAX_SEQ, D), "sin": z(MAX_SEQ, D),
        "qo_indptr": torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        "kv_indptr": torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        "kv_indices": torch.arange(MAX_SEQ // PAGE, dtype=torch.int32,
                                   device="cuda"),
        "last_page_len": torch.ones(1, dtype=torch.int32, device="cuda"),
        "num_tokens": torch.ones(1, dtype=torch.int32, device="cuda"),
        "attn_parts": z(sm90_attention_parts_size(1, QH, D, nsm),
                        dtype=torch.float32),
        "h": z(T, H), "qkv32": z(g.qkv_parts() * T, (QH + 2 * KVH) * D, dtype=torch.float32),
        "qkv": z(T, (QH + 2 * KVH) * D), "attn": z(T, QH * D),
        "o32": z(g.O_PARTS * T, H, dtype=torch.float32), "x1": z(T, H),
        "h2": z(T, H, dtype=torch.float16),
        "topk_w": z(T, TOPK, dtype=torch.float32),
        "routing": z(E, T, dtype=torch.int32),
        "act": z(E, T, I, dtype=torch.float16), "eo": z(T, TOPK, H),
        "ln1": z(H), "ln2": z(H), "w_qkv": g.pad_k(z((QH + 2 * KVH) * D, H)),
        "b_qkv": z((QH + 2 * KVH) * D), "w_o": z(H, QH * D), "b_o": z(H),
        "sinks": z(KVH, QH // KVH), "w_router": z(E, H), "b_router": z(E),
        "w13": z(E, 2 * I // R13, kt, R13 * 68, dtype=torch.uint8),
        "b13": z(E, 2 * I),
        "w2": z(E, H // R2, kt, R2 * 68, dtype=torch.uint8), "b2": z(E, H),
        "k_cache": z(MAX_SEQ // PAGE, PAGE, KVH, D),
        "v_cache": z(MAX_SEQ // PAGE, PAGE, KVH, D),
        "x_in": z(T, H), "x_out": z(T, H),
    }
    if native:
        P = g.NATIVE_PAD
        t.update({"z_q": z(T, P, dtype=torch.uint8),
                  "w13": z(E, P // 64, P // 128, 8192, dtype=torch.uint8),
                  "w13_scales": z(E, P // 64, P // 128, 512, dtype=torch.uint8),
                  "w2": z(E, P // 128, P // 128, 8192, dtype=torch.uint8),
                  "w2_scales": z(E, P // 128, P // 128, 512, dtype=torch.uint8),
                  "b13": z(E, 2 * P), "b2": z(E, P)})
    # sm100 without the native experts: FlashInfer's inputs from the gate (bf16
    # h in padded rows, the top-k ids past routing's E rows)
    fi = cc == 100 and not native
    if fi:
        t.update({"h2": z(T, 3072 * 33 // 32, dtype=torch.uint8),   # MXFP8 h
                  "routing": z(E + TOPK, T, dtype=torch.int32),
                  "y_in": z(T, H), "x_res": z(T, H)})
    m = type("Cfg", (), dict(eps=1e-5, QH=QH, KVH=KVH, E=E, limit=7.0,
                             max_seq=MAX_SEQ, R=1,
                             moe_kind="flashinfer" if fi else "mpk", fi_logits=fi))
    mpk = StaticMegakernel(num_gpus=1, num_sms=nsm, target_cc=cc)
    g.build_layer(m, mpk, t, 128 if kind == "sliding" else 0)
    # the shipped H100 plan, else round robin with the router over 4 tasks a
    # token (the gate's changeable grid dim)
    gate = next(n.graph_idx for n in compiler.graph_nodes(mpk)
                if n.name == "sm90_gate")
    grids = {gate: (T, 4, 1)}
    if cc == 100:   # QKV: 64-row tiles in its fewest K parts (the attention adds them); o_proj in 2 (the gate adds them)
        gemms = [n for n in compiler.graph_nodes(mpk) if n.name == "gemm_tile"]
        grids[gemms[0].graph_idx] = min((c for c in compiler.candidate_grids(gemms[0])
                                         if c[0] == gemms[0].grid[0] * 2), key=lambda c: c[1])
        grids[gemms[1].graph_idx] = (gemms[1].grid[0], 2, 1)
    plan = (cc == 90 and g.load_plan(mpk, g.plan_path(
        m, os.path.join(demo, "plans", "h100"), kind))) or \
        plan_gen.round_robin_plan(mpk, grids)
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                       f"mpk_cross_arch_gpt_oss_{kind}_{cc}_{os.getpid()}")
    paths, _ = compiler.compile_plan(mpk, plan, os.path.join(out, "schedules"))
    mpk.compile_static(paths, gpu_tensors=[t], out_dir=os.path.join(out, "build"),
                       extra_flags=g.flags(), build_only=True)


def _build_gpt_oss_static_head(cc=90):
    """The GPT-OSS static head (final norm, lm_head, the next step's tokens)
    at a small vocabulary, two requests, round-robin over the target's SMs,
    built for sm_90a or sm_100a."""
    from mirage.mpk import compiler, plan_gen
    from mirage.mpk.static_megakernel import StaticMegakernel

    g, _ = _gpt_oss_demo()
    g.CC = cc
    H, V, T, MAX_SEQ, R = 2880, 8192, g.T, 1024, 2   # two requests

    def z(*shape, dtype=torch.bfloat16):
        return torch.zeros(*shape, dtype=dtype, device="cuda")

    t = {
        "x": z(T, H), "norm": z(H), "lm_head": z(V, H), "h": z(T, H),
        "logits": z(T, V, dtype=torch.float32), "embed": z(V, H),
        "prompt": z(R, 16, dtype=torch.long), "gen": z(R, 64, dtype=torch.long),
        "x_embed": z(T, H),
        "kv_indptr": torch.arange(R + 1, dtype=torch.int32, device="cuda"),
        "last_page_len": torch.ones(R, dtype=torch.int32, device="cuda"),
        "kv_indices": torch.zeros(R * MAX_SEQ // 64, dtype=torch.int32, device="cuda"),
        "kv_indices_ring": torch.zeros(R * MAX_SEQ // 64, dtype=torch.int32, device="cuda"),
    }
    if cc == 100:   # FlashInfer's experts: the final norm adds their sum
        t.update({"y_in": z(T, H), "x_res": z(T, H)})
    m = type("Cfg", (), dict(eps=1e-5, max_seq=MAX_SEQ, RP=4))
    mpk = StaticMegakernel(num_gpus=1, num_sms=148 if cc == 100 else 132,
                           target_cc=cc)
    g.build_head(m, mpk, t)
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                       f"mpk_cross_arch_gpt_oss_head_{cc}_{os.getpid()}")
    paths, _ = compiler.compile_plan(mpk, plan_gen.round_robin_plan(mpk),
                                     os.path.join(out, "schedules"))
    mpk.compile_static(paths, gpu_tensors=[t], out_dir=os.path.join(out, "build"),
                       extra_flags=g.flags(), build_only=True)


BUILDS = {
    "paged_attention": _build_paged_attention,
    "paged_attention_split_kv": _build_paged_attention_split_kv,
}
STATIC_BUILDS = {
    "gpt_oss_static_sliding": lambda: _build_gpt_oss_static("sliding"),
    "gpt_oss_static_full": lambda: _build_gpt_oss_static("full"),
    "gpt_oss_static_head": _build_gpt_oss_static_head,
    "gpt_oss_static_sliding_sm100": lambda: _build_gpt_oss_static("sliding", 100),
    "gpt_oss_static_full_sm100": lambda: _build_gpt_oss_static("full", 100),
    "gpt_oss_static_head_sm100": lambda: _build_gpt_oss_static_head(100),
    "gpt_oss_static_native_sm100": lambda: _build_gpt_oss_static(
        "full", 100, native=True),
}


# Task variants accumulate in a process-wide registry and are all emitted into
# the next build, so each target is built in its own process.
@pytest.mark.parametrize("target_cc", sorted(SHAPES))
@pytest.mark.parametrize("build", sorted(BUILDS))
def test_build(build, target_cc):
    subprocess.run([sys.executable, __file__, build, str(target_cc)],
                   check=True)


@pytest.mark.parametrize("build", sorted(STATIC_BUILDS))
def test_static_build(build):
    subprocess.run([sys.executable, __file__, build], check=True)


if __name__ == "__main__":
    if sys.argv[1] in STATIC_BUILDS:
        STATIC_BUILDS[sys.argv[1]]()
    else:
        BUILDS[sys.argv[1]](int(sys.argv[2]))
