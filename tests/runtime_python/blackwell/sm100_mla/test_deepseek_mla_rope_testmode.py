"""DeepSeek-V3 MLA RoPE tasks through MPK test_mode.

Covers the THREE real rope layers the DSV3 builder uses (builder.py L2646/2660/
2672 and the MTP sibling L4108/4117/4125):

  * deepseek_mla_rope_q_fused_layer  — DECODE layout. Rotates the TAIL 64 dims
        of each head's 576-wide slice in q_nope_pe [mbt, H*576].
  * deepseek_mla_rope_q_split_layer  — PREFILL layout. Two sub-variants:
        qfused_mode=0: standalone q_pe [mbt, H*64], rotates each head's 64 dims.
        qfused_mode=1: row-swap fused q_b_prefill_fused [mbt, H*192], rotates the
                       pe slice at [H*128 + head*64 : +64] within each row.
  * deepseek_mla_rope_k_layer        — rotates the FIRST 64 dims of k_pe with
        row stride K_PE_STRIDE (128).

Rotation convention: GPT-J / interleaved (is_neox_style=False) — pairs (0,1),
(2,3),... with repeat_interleave cos/sin, matching vLLM/SGLang and the kernel
``deepseek_mla_rope_sm100_task_impl`` (deepseek_mla_rope_sm100.cuh L58-106).

The cos/sin table is built with the EXACT DeepSeek-V3 YARN definition
(rope_theta=10000, qk_rope_head_dim=64, factor=40, beta_fast=32, beta_slow=1,
mscale=mscale_all_dim=1.0, original_max_position_embeddings=4096) via
``pytorch_reference.build_dsv3_yarn_rope_tables`` — identical to
``builder._precompute_rope_embeddings``. Both kernel and reference read the SAME
table, so YARN lives in one place; correctness is decided by the rotation
convention + slice layout (which is what these tests verify).

phase_gate=0 for every config → the kernel always rotates (no Q_LEN gate), so
the result is deterministic regardless of the prefill/decode q_len.

H = num_local_q_heads = 128 // tp -> {128, 64, 32, 16}. bs = number of tokens in
a single prefill request (= seq_len, positions 0..bs-1 since step==0).

Run:
    pytest tests/runtime_python/blackwell/sm100_mla/test_deepseek_mla_rope_testmode.py -s
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.models.deepseek_v3 import tasks as dsv3_tasks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytorch_reference import build_dsv3_yarn_rope_tables, rope_rotate_gptj  # noqa: E402

FUSED_HEAD_DIM = 576
ROPE_DIM = 64
K_PE_STRIDE = 128
QB_PREFILL_HEAD_DIM = 192  # nope 128 || pe 64 per head in the row-swap fused buf
ATOL = 1e-2
RTOL = 1e-2

# Union-of-axes TP×bs matrix (decision-log policy for shape-sharded layers).
# H is TP-sharded (= 128 // tp), so tp is a genuine shape axis.
# {tp=1}×{bs=1,2,4,8,16} ∪ {bs=16}×{tp=2,4,8} ∪ corner {tp=8,bs=1} = 9 configs.
MATRIX = (
    [(1, bs) for bs in (1, 2, 4, 8, 16)]
    + [(tp, 16) for tp in (2, 4, 8)]
    + [(8, 1)]
)


def _common_params(seq_len):
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = seq_len
    params["max_num_batched_requests"] = 1
    params["max_seq_length"] = seq_len
    params["max_num_pages"] = 1
    params["page_size"] = seq_len
    # Single prefill request of length seq_len. prepare_next_batch fills
    # request_ids[0]=0, qo_indptr_buffer=[0, seq_len], step[0]=0, so the kernel
    # rotates token positions 0..seq_len-1 (position = step + token_idx).
    params["meta_tensors"] = {
        "prompt_lengths": torch.tensor(
            [seq_len], dtype=torch.int32, device="cuda"),
    }
    return params


def _build_and_run(params, register_fn, folder):
    pk = PersistentKernel(**params)
    register_fn(pk)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()
    pk.finalize()


def _check(name, tp, bs, out, ref):
    max_diff = (out.float() - ref.float()).abs().max().item()
    ok = torch.allclose(out, ref, atol=ATOL, rtol=RTOL)
    status = "PASS" if ok else "FAIL"
    print(f"  [{name}] tp={tp} H={128 // tp} bs={bs}: "
          f"max_diff={max_diff:.6f} -> {status}")
    assert ok, (
        f"{name} tp={tp} bs={bs} max_diff={max_diff} exceeds "
        f"atol={ATOL} rtol={RTOL}")
    return max_diff


def _run_q_fused(tp, bs, folder):
    """deepseek_mla_rope_q_fused_layer — decode layout: rotate the tail 64 of
    each head's 576-wide slice in q_nope_pe [bs, H*576]."""
    torch.manual_seed(7)
    H = 128 // tp
    seq_len = bs
    cos, sin = build_dsv3_yarn_rope_tables(seq_len)

    q = torch.randn(seq_len, H * FUSED_HEAD_DIM,
                    dtype=torch.bfloat16, device="cuda") * 0.1
    ref = q.clone()
    rv = ref.view(seq_len, H, FUSED_HEAD_DIM)
    tail = rv[:, :, FUSED_HEAD_DIM - ROPE_DIM:].permute(1, 0, 2)  # [H, S, 64]
    rv[:, :, FUSED_HEAD_DIM - ROPE_DIM:] = rope_rotate_gptj(
        tail, cos, sin).permute(1, 0, 2)

    params = _common_params(seq_len)

    def reg(pk):
        q_dt = pk.attach_input(q, name="q_fused")
        cos_dt = pk.attach_input(cos, name="rope_cos")
        sin_dt = pk.attach_input(sin, name="rope_sin")
        dsv3_tasks.deepseek_mla_rope_q_fused_layer(
            pk,
            q_nope_pe=q_dt,
            cos_pos_embed=cos_dt,
            sin_pos_embed=sin_dt,
            num_heads=H,
            grid_dim=(1, H, 1),
            block_dim=(128, 1, 1),
            q_tile_size=seq_len,
            phase_gate=0,
        )

    _build_and_run(params, reg, folder)
    return _check("q_fused", tp, bs, q, ref)


def _run_q_split(tp, bs, folder, qfused_mode):
    """deepseek_mla_rope_q_split_layer.

    qfused_mode=0: standalone q_pe [bs, H*64], rotate each head's 64.
    qfused_mode=1: row-swap fused [bs, H*192], rotate pe at [H*128 + head*64].
    """
    torch.manual_seed(7)
    H = 128 // tp
    seq_len = bs
    cos, sin = build_dsv3_yarn_rope_tables(seq_len)

    if qfused_mode == 0:
        q = torch.randn(seq_len, H * ROPE_DIM,
                        dtype=torch.bfloat16, device="cuda") * 0.1
        ref = q.clone()
        rv = ref.view(seq_len, H, ROPE_DIM)
        pe = rv.permute(1, 0, 2)  # [H, S, 64]
        rv[:, :, :] = rope_rotate_gptj(pe, cos, sin).permute(1, 0, 2)
    else:
        # Row-swap fused buffer: each row is [H*128 nope || H*64 pe].
        q = torch.randn(seq_len, H * QB_PREFILL_HEAD_DIM,
                        dtype=torch.bfloat16, device="cuda") * 0.1
        ref = q.clone()
        pe_base = H * 128
        pe_block = ref[:, pe_base:pe_base + H * ROPE_DIM].view(seq_len, H, ROPE_DIM)
        pe = pe_block.permute(1, 0, 2)  # [H, S, 64]
        pe_block[:, :, :] = rope_rotate_gptj(pe, cos, sin).permute(1, 0, 2)

    params = _common_params(seq_len)

    def reg(pk):
        q_dt = pk.attach_input(q, name="q_split")
        cos_dt = pk.attach_input(cos, name="rope_cos")
        sin_dt = pk.attach_input(sin, name="rope_sin")
        dsv3_tasks.deepseek_mla_rope_q_split_layer(
            pk,
            q_pe=q_dt,
            cos_pos_embed=cos_dt,
            sin_pos_embed=sin_dt,
            num_heads=H,
            grid_dim=(1, H, 1),
            block_dim=(128, 1, 1),
            q_tile_size=seq_len,
            qfused_mode=qfused_mode,
            phase_gate=0,
        )

    _build_and_run(params, reg, folder)
    name = f"q_split(mode={qfused_mode})"
    return _check(name, tp, bs, q, ref)


def _run_rope_k(tp, bs, folder):
    """deepseek_mla_rope_k_layer — rotate the FIRST 64 of k_pe [bs, 128]."""
    torch.manual_seed(7)
    seq_len = bs
    cos, sin = build_dsv3_yarn_rope_tables(seq_len)

    k = torch.zeros(seq_len, K_PE_STRIDE, dtype=torch.bfloat16, device="cuda")
    k[:, :ROPE_DIM] = torch.randn(
        seq_len, ROPE_DIM, dtype=torch.bfloat16, device="cuda") * 0.1
    ref = k.clone()
    ref[:, :ROPE_DIM] = rope_rotate_gptj(ref[:, :ROPE_DIM], cos, sin)

    params = _common_params(seq_len)

    def reg(pk):
        k_dt = pk.attach_input(k, name="k_pe")
        cos_dt = pk.attach_input(cos, name="rope_cos")
        sin_dt = pk.attach_input(sin, name="rope_sin")
        dsv3_tasks.deepseek_mla_rope_k_layer(
            pk,
            k_pe=k_dt,
            cos_pos_embed=cos_dt,
            sin_pos_embed=sin_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
            q_tile_size=seq_len,
        )

    _build_and_run(params, reg, folder)
    # rope_k has no head sharding (NUM_HEADS=1); tp only varies seq via bs here,
    # but we still tag tp for matrix bookkeeping.
    return _check("rope_k", tp, bs, k, ref)


def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    results = []  # (variant, tp, bs, max_diff)
    n_fail = 0

    print("=== q_fused (decode layout: tail-64 of H*576) ===")
    for tp, bs in MATRIX:
        try:
            d = _run_q_fused(tp, bs, folder)
            results.append(("q_fused", tp, bs, d))
        except AssertionError as e:
            n_fail += 1
            print(f"  FAIL: {e}")

    print("=== q_split mode=0 (standalone H*64) ===")
    for tp, bs in MATRIX:
        try:
            d = _run_q_split(tp, bs, folder, qfused_mode=0)
            results.append(("q_split0", tp, bs, d))
        except AssertionError as e:
            n_fail += 1
            print(f"  FAIL: {e}")

    print("=== q_split mode=1 (row-swap fused H*192) ===")
    for tp, bs in MATRIX:
        try:
            d = _run_q_split(tp, bs, folder, qfused_mode=1)
            results.append(("q_split1", tp, bs, d))
        except AssertionError as e:
            n_fail += 1
            print(f"  FAIL: {e}")

    print("=== rope_k (first-64 of k_pe stride 128) ===")
    # rope_k is not head-sharded; sweep bs (the only axis that matters) once.
    for bs in (1, 2, 4, 8, 16):
        try:
            d = _run_rope_k(1, bs, folder)
            results.append(("rope_k", 1, bs, d))
        except AssertionError as e:
            n_fail += 1
            print(f"  FAIL: {e}")

    n_pass = len(results)
    n_total = n_pass + n_fail
    print(f"\nROPE SUMMARY: {n_pass}/{n_total} PASS")
    if n_fail == 0:
        print("ALL PASS")
    return n_fail


def test_deepseek_mla_rope_q_fused_testmode():
    folder = os.path.dirname(os.path.abspath(__file__))
    for tp, bs in MATRIX:
        _run_q_fused(tp, bs, folder)


def test_deepseek_mla_rope_q_split_testmode():
    folder = os.path.dirname(os.path.abspath(__file__))
    for tp, bs in MATRIX:
        _run_q_split(tp, bs, folder, qfused_mode=0)
        _run_q_split(tp, bs, folder, qfused_mode=1)


def test_deepseek_mla_rope_k_testmode():
    folder = os.path.dirname(os.path.abspath(__file__))
    for bs in (1, 2, 4, 8, 16):
        _run_rope_k(1, bs, folder)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
