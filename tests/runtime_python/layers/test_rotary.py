"""Sanity test for ``layers.RotaryEmbedding``.

Unlike the rest of the Phase-2 catalog tests, ``RotaryEmbedding``
emits **no MPK task** — its sole compile-time job is to attach the
precomputed cos/sin buffers as DTensors so the attention kernel
(``pk.attention_layer``) can consume them. There is therefore nothing
to ``pk.compile()`` and nothing to ``pk()``-launch here; the test
verifies only:

1. The precomputation produces tables of the right shape and dtype.
2. ``forward(positions)`` indexes those tables correctly.
3. Inside ``with pk.compile_scope():``, ``compile()`` calls
   ``pk.attach_input`` on both buffers and returns two ``DTensor``s
   whose ``num_dims`` / ``dim(1)`` match what
   ``persistent_kernel.attention_layer`` asserts.

DO NOT execute this file as part of Phase 2 — Phase 4 runs it on a
free GPU. The ``mirage`` conda env is required. The test is written
to run on CPU as well (it never touches the kernel runtime), but it
imports ``mirage`` to construct the ``PersistentKernel`` which in
practice requires CUDA.
"""

import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_rotary_embedding():
    torch.manual_seed(0)

    # Architecture choices match a Qwen3-style attention head:
    #   * head_dim divisible by 2 (rotate_half requires it).
    #   * max_position_embeddings = 2048, well below the 4096 the
    #     demo currently slices to but big enough to exercise the
    #     position-indexing path.
    head_dim = 128
    max_pos = 2048
    base = 10000.0

    # ------------------------------------------------------------------
    # 1. Buffer shape + dtype contract.
    # ------------------------------------------------------------------
    rotary = layers.RotaryEmbedding(
        head_dim=head_dim,
        max_position_embeddings=max_pos,
        base=base,
        prefix="test_",
    )

    assert rotary.cos.shape == (max_pos, head_dim), (
        f"cos.shape={tuple(rotary.cos.shape)}, expected "
        f"({max_pos}, {head_dim})"
    )
    assert rotary.sin.shape == (max_pos, head_dim), (
        f"sin.shape={tuple(rotary.sin.shape)}, expected "
        f"({max_pos}, {head_dim})"
    )
    assert rotary.cos.dtype == torch.bfloat16, (
        f"cos.dtype={rotary.cos.dtype}, expected torch.bfloat16. "
        "attention_layer consumes cos_pos_embed/sin_pos_embed as "
        "bf16 DTensors."
    )
    assert rotary.sin.dtype == torch.bfloat16

    # Buffers should not appear in state_dict() — they're non-persistent
    # so HF safetensor loading doesn't trip on missing/extra keys.
    sd_keys = set(rotary.state_dict().keys())
    assert "cos" not in sd_keys and "sin" not in sd_keys, (
        f"cos/sin must be non-persistent buffers; state_dict keys: {sd_keys}"
    )

    # ------------------------------------------------------------------
    # 2. forward() indexing.
    # ------------------------------------------------------------------
    positions = torch.arange(0, 16, dtype=torch.long)
    cos_fwd, sin_fwd = rotary.forward(positions)
    assert cos_fwd.shape == (positions.shape[0], head_dim)
    assert sin_fwd.shape == (positions.shape[0], head_dim)
    assert cos_fwd.dtype == torch.bfloat16
    assert sin_fwd.dtype == torch.bfloat16
    # Position 0 has all freqs = 0, so cos == 1, sin == 0 exactly
    # (both halves of the head_dim, because of the
    # ``torch.cat((freqs, freqs), dim=-1)`` convention).
    assert torch.all(cos_fwd[0] == 1.0), (
        f"cos at position 0 must be all ones; got {cos_fwd[0][:4]}"
    )
    assert torch.all(sin_fwd[0] == 0.0), (
        f"sin at position 0 must be all zeros; got {sin_fwd[0][:4]}"
    )
    # The two halves of head_dim must be equal (HF rotate_half
    # convention), at every position.
    half = head_dim // 2
    torch.testing.assert_close(cos_fwd[..., :half], cos_fwd[..., half:])
    torch.testing.assert_close(sin_fwd[..., :half], sin_fwd[..., half:])

    # ------------------------------------------------------------------
    # 3. compile() wiring.
    # ------------------------------------------------------------------
    # Move the buffers to CUDA so ``attach_input`` (which stores raw
    # device pointers) sees device tensors. ``.to`` migrates registered
    # buffers as well as parameters.
    rotary = rotary.to(device="cuda")

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    pk = PersistentKernel(**params)

    with pk.compile_scope():
        cos_dt, sin_dt = rotary.compile()

    assert cos_dt is not None, "compile() returned cos_dt=None"
    assert sin_dt is not None, "compile() returned sin_dt=None"

    # The shape/dim invariants ``attention_layer`` will assert
    # (persistent_kernel.py:781-784):
    assert cos_dt.num_dims == 2, (
        f"cos_dt.num_dims={cos_dt.num_dims}, expected 2 — "
        "attention_layer asserts ``cos_pos_embed.num_dims == 2``."
    )
    assert sin_dt.num_dims == 2, (
        f"sin_dt.num_dims={sin_dt.num_dims}, expected 2."
    )
    assert cos_dt.dim(0) == max_pos, (
        f"cos_dt.dim(0)={cos_dt.dim(0)}, expected {max_pos}."
    )
    assert cos_dt.dim(1) == head_dim, (
        f"cos_dt.dim(1)={cos_dt.dim(1)}, expected {head_dim} — "
        "attention_layer asserts ``cos_pos_embed.dim(1) == head_dim``."
    )
    assert sin_dt.dim(0) == max_pos
    assert sin_dt.dim(1) == head_dim

    # No kernel to launch — RotaryEmbedding emits no MPK task.
    # Nothing to compare against; the wiring is the contract.
    print(
        f"PASSED: layers.RotaryEmbedding precompute "
        f"({max_pos}, {head_dim}) bf16 buffers and produces 2-D "
        f"DTensors that match attention_layer's preconditions."
    )

    pk.finalize()


if __name__ == "__main__":
    try:
        test_rotary_embedding()
    except AssertionError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
