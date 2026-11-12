"""Build kernels for every supported architecture on whatever GPU CI has,
without running them: MPK's paged attention, and the GPT-OSS layers of the
Hopper static megakernel.

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


def _build_gpt_oss_static(kind):
    """One GPT-OSS-20B layer (kind: sliding or full attention) of the Hopper
    static megakernel, its tensors zeros, on the shipped H100 plan (132 SMs) or
    round robin, built for sm_90a with the demo's serving flags."""
    from mirage.mpk import compiler, plan_gen
    from mirage.mpk.static_megakernel import (StaticMegakernel,
                                              sm90_attention_parts_size)

    g, demo = _gpt_oss_demo()
    H, I, E, TOPK, QH, KVH, D, MAX_SEQ = 2880, 2880, 32, 4, 64, 8, 64, 1024
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
        "attn_parts": z(sm90_attention_parts_size(T, QH, D,
                                                  MAX_SEQ // g.KV_CHUNK)),
        "h": z(T, H), "qkv32": z(T, (QH + 2 * KVH) * D, dtype=torch.float32),
        "qkv": z(T, (QH + 2 * KVH) * D), "attn": z(T, QH * D),
        "o32": z(T, H, dtype=torch.float32), "x1": z(T, H),
        "h2": z(T, H, dtype=torch.float16),
        "topk_w": z(T, TOPK, dtype=torch.float32),
        "routing": z(E, T, dtype=torch.int32),
        "active": z(E + 1, dtype=torch.int32),
        "act": z(TOPK, T, I, dtype=torch.float16), "eo": z(T, TOPK, H),
        "ln1": z(H), "ln2": z(H), "w_qkv": z((QH + 2 * KVH) * D, H),
        "b_qkv": z((QH + 2 * KVH) * D), "w_o": z(H, QH * D), "b_o": z(H),
        "sinks": z(KVH, QH // KVH), "w_router": z(E, H), "b_router": z(E),
        "w13": z(E, 2 * I // R13, kt, R13 * 68, dtype=torch.uint8),
        "b13": z(E, 2 * I),
        "w2": z(E, H // R2, kt, R2 * 68, dtype=torch.uint8), "b2": z(E, H),
        "k_cache": z(MAX_SEQ // PAGE, PAGE, KVH, D),
        "v_cache": z(MAX_SEQ // PAGE, PAGE, KVH, D),
        "x_in": z(T, H), "x_out": z(T, H),
    }
    m = type("Cfg", (), dict(eps=1e-5, QH=QH, E=E, limit=7.0,
                             max_seq=MAX_SEQ, R=1))
    mpk = StaticMegakernel(num_gpus=1, num_sms=132, target_cc=90)
    g.build_layer(m, mpk, t, 128 if kind == "sliding" else 0)
    # the shipped H100 plan, else round robin with the router over 4 tasks a
    # token (the gate's changeable grid dim)
    gate = next(n.graph_idx for n in compiler.graph_nodes(mpk)
                if n.name == "sm90_gate")
    plan = g.load_plan(mpk, g.plan_path(m, os.path.join(demo, "plans", "h100"),
                                        kind)) or \
        plan_gen.round_robin_plan(mpk, {gate: (T, 4, 1)})
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                       f"mpk_cross_arch_gpt_oss_{kind}_{os.getpid()}")
    paths, _ = compiler.compile_plan(mpk, plan, os.path.join(out, "schedules"))
    mpk.compile_static(paths, gpu_tensors=[t], out_dir=os.path.join(out, "build"),
                       extra_flags=g.FLAGS, build_only=True)


def _build_gpt_oss_static_head():
    """The GPT-OSS static head (final norm, lm_head, the next step's tokens)
    at a small vocabulary, two requests, round-robin over 132 SMs, built for
    sm_90a."""
    from mirage.mpk import compiler, plan_gen
    from mirage.mpk.static_megakernel import StaticMegakernel

    g, _ = _gpt_oss_demo()
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
    }
    m = type("Cfg", (), dict(eps=1e-5, max_seq=MAX_SEQ))
    mpk = StaticMegakernel(num_gpus=1, num_sms=132, target_cc=90)
    g.build_head(m, mpk, t)
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"),
                       f"mpk_cross_arch_gpt_oss_head_{os.getpid()}")
    paths, _ = compiler.compile_plan(mpk, plan_gen.round_robin_plan(mpk),
                                     os.path.join(out, "schedules"))
    mpk.compile_static(paths, gpu_tensors=[t], out_dir=os.path.join(out, "build"),
                       extra_flags=g.FLAGS, build_only=True)


BUILDS = {
    "paged_attention": _build_paged_attention,
    "paged_attention_split_kv": _build_paged_attention_split_kv,
}
STATIC_BUILDS = {
    "gpt_oss_static_sliding": lambda: _build_gpt_oss_static("sliding"),
    "gpt_oss_static_full": lambda: _build_gpt_oss_static("full"),
    "gpt_oss_static_head": _build_gpt_oss_static_head,
}


# Task variants accumulate in a process-wide registry and are all emitted into
# the next build, so each target is built in its own process.
@pytest.mark.parametrize("target_cc", sorted(SHAPES))
@pytest.mark.parametrize("build", sorted(BUILDS))
def test_build(build, target_cc):
    subprocess.run([sys.executable, __file__, build, str(target_cc)],
                   check=True)


# The Hopper static megakernel builds for sm90 only.
@pytest.mark.parametrize("build", sorted(STATIC_BUILDS))
def test_static_build(build):
    subprocess.run([sys.executable, __file__, build], check=True)


if __name__ == "__main__":
    if sys.argv[1] in STATIC_BUILDS:
        STATIC_BUILDS[sys.argv[1]]()
    else:
        BUILDS[sys.argv[1]](int(sys.argv[2]))
