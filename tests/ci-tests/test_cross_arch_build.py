"""Build paged attention for every supported architecture on whatever GPU CI
has, without running it.

Codegen and nvcc do not need the target GPU, so this catches template
argument mismatches on a path that has no hardware in CI (sm90 today).
"""

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


BUILDS = {
    "paged_attention": _build_paged_attention,
    "paged_attention_split_kv": _build_paged_attention_split_kv,
}


# Task variants accumulate in a process-wide registry and are all emitted into
# the next build, so each target is built in its own process.
@pytest.mark.parametrize("target_cc", sorted(SHAPES))
@pytest.mark.parametrize("build", sorted(BUILDS))
def test_build(build, target_cc):
    subprocess.run([sys.executable, __file__, build, str(target_cc)],
                   check=True)


if __name__ == "__main__":
    BUILDS[sys.argv[1]](int(sys.argv[2]))
