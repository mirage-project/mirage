import math
import shutil

import pytest
import torch


HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
COMPRESS_RATIO = 4
COFF = 2
KV_SCORE_DIM = 4 * HEAD_DIM
C4_PAGE_SIZE = 128
NORM_EPS = 1e-6
COMPRESS_ROPE_THETA = 160000.0


def _skip_without_sm100_and_nvcc():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MPK C4 compressor tests")
    major, _minor = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip("dsv4_c4_compress_sm100 requires SM100+")
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc is required to compile MPK test kernels")


def _make_rope_cos_sin(max_seq_len, device):
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.arange(0, ROPE_HEAD_DIM, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (COMPRESS_ROPE_THETA ** (freqs / ROPE_HEAD_DIM))
    angles = torch.outer(positions, inv_freq)
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


def _apply_gptj_rope(x, rope_row):
    cos = rope_row[: ROPE_HEAD_DIM // 2]
    sin = rope_row[ROPE_HEAD_DIM // 2 :]
    rope = x[..., NOPE_HEAD_DIM:].float()
    even = rope[..., 0::2]
    odd = rope[..., 1::2]
    rotated = torch.empty_like(rope)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = odd * cos + even * sin
    return torch.cat([x[..., :NOPE_HEAD_DIM], rotated], dim=-1)


def _rmsnorm(x, weight):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + NORM_EPS) * weight


def _compress_from_slots(kv_slots, score_slots, ape, norm_weight, rope_row):
    score = score_slots.float() + ape.unsqueeze(0)
    weights = torch.softmax(score, dim=1)
    pooled = (kv_slots.float() * weights).sum(dim=1)
    pooled = _rmsnorm(pooled, norm_weight.float())
    return _apply_gptj_rope(pooled, rope_row)


def _empty_reference(batch, num_c4_slots, device):
    state = torch.zeros(batch, 8, KV_SCORE_DIM, dtype=torch.float32, device=device)
    cache = torch.zeros(num_c4_slots, HEAD_DIM, dtype=torch.float32, device=device)
    return state, cache


def _reference_prefill(flat_kv_score, seq_lens, token_meta, ape, norm_weight, rope_cos_sin):
    device = flat_kv_score.device
    batch = len(seq_lens)
    num_c4_slots = int(torch.clamp(token_meta[:, 1], min=0).max().item()) + 1
    state, cache = _empty_reference(batch, num_c4_slots, device)

    cursor = 0
    for b, seqlen in enumerate(seq_lens):
        req = flat_kv_score[cursor : cursor + seqlen]
        req_meta = token_meta[cursor : cursor + seqlen]
        cutoff = seqlen - (seqlen % COMPRESS_RATIO)
        num_blocks = cutoff // COMPRESS_RATIO

        for block_idx in range(num_blocks):
            start = block_idx * COMPRESS_RATIO
            cur = req[start : start + COMPRESS_RATIO]
            kv_slots = torch.zeros(1, 8, HEAD_DIM, device=device)
            score_slots = torch.full((1, 8, HEAD_DIM), -float("inf"), device=device)
            if block_idx > 0:
                prev = req[start - COMPRESS_RATIO : start]
                kv_slots[:, :4] = prev[:, :HEAD_DIM].unsqueeze(0)
                score_slots[:, :4] = prev[:, 2 * HEAD_DIM : 3 * HEAD_DIM].unsqueeze(0)
            kv_slots[:, 4:] = cur[:, HEAD_DIM : 2 * HEAD_DIM].unsqueeze(0)
            score_slots[:, 4:] = cur[:, 3 * HEAD_DIM :].unsqueeze(0)

            abs_pos = int(req_meta[start + COMPRESS_RATIO - 1, 0].item())
            c4_slot = int(req_meta[start + COMPRESS_RATIO - 1, 1].item())
            assert c4_slot >= 0
            cache[c4_slot] = _compress_from_slots(
                kv_slots, score_slots, ape, norm_weight, rope_cos_sin[abs_pos + 1 - COMPRESS_RATIO]
            ).squeeze(0)

        if cutoff >= COMPRESS_RATIO:
            state[b, :4] = req[cutoff - COMPRESS_RATIO : cutoff]
        remainder = seqlen % COMPRESS_RATIO
        if remainder:
            state[b, 4 : 4 + remainder] = req[cutoff:]
        cursor += seqlen

    return state, cache


def _reference_decode(flat_kv_score, seq_lens, token_meta, ape, norm_weight, rope_cos_sin):
    device = flat_kv_score.device
    batch = len(seq_lens)
    num_c4_slots = int(torch.clamp(token_meta[:, 1], min=0).max().item()) + 1
    state, cache = _empty_reference(batch, num_c4_slots, device)

    cursor = 0
    for b, seqlen in enumerate(seq_lens):
        for i in range(seqlen):
            token = flat_kv_score[cursor + i]
            abs_pos = int(token_meta[cursor + i, 0].item())
            current_slot = 4 + (abs_pos % COMPRESS_RATIO)
            state[b, current_slot] = token
            if (abs_pos + 1) % COMPRESS_RATIO != 0:
                continue

            kv_slots = torch.zeros(1, 8, HEAD_DIM, device=device)
            score_slots = torch.full((1, 8, HEAD_DIM), -float("inf"), device=device)
            if abs_pos + 1 > COMPRESS_RATIO:
                kv_slots[:, :4] = state[b, :4, :HEAD_DIM].unsqueeze(0)
                score_slots[:, :4] = state[b, :4, 2 * HEAD_DIM : 3 * HEAD_DIM].unsqueeze(0)
            kv_slots[:, 4:] = state[b, 4:, HEAD_DIM : 2 * HEAD_DIM].unsqueeze(0)
            score_slots[:, 4:] = state[b, 4:, 3 * HEAD_DIM :].unsqueeze(0)

            c4_slot = int(token_meta[cursor + i, 1].item())
            assert c4_slot >= 0
            cache[c4_slot] = _compress_from_slots(
                kv_slots, score_slots, ape, norm_weight, rope_cos_sin[abs_pos + 1 - COMPRESS_RATIO]
            ).squeeze(0)
            state[b, :4] = state[b, 4:].clone()
        cursor += seqlen

    return state, cache


def _make_case(seq_lens, device):
    gen = torch.Generator(device=device).manual_seed(1234)
    total_tokens = sum(seq_lens)
    max_blocks_per_req = max(1, math.ceil(max(seq_lens) / COMPRESS_RATIO))
    num_c4_slots = len(seq_lens) * max_blocks_per_req
    num_c4_pages = math.ceil(num_c4_slots / C4_PAGE_SIZE)

    kv_score = torch.randn(
        total_tokens, KV_SCORE_DIM, generator=gen, device=device, dtype=torch.float32
    )
    token_meta = torch.full((total_tokens, 2), -1, dtype=torch.int32, device=device)
    cursor = 0
    for b, seqlen in enumerate(seq_lens):
        for pos in range(seqlen):
            row = cursor + pos
            token_meta[row, 0] = pos
            if (pos + 1) % COMPRESS_RATIO == 0:
                token_meta[row, 1] = b * max_blocks_per_req + pos // COMPRESS_RATIO
        cursor += seqlen

    raw_ape = torch.randn(
        COMPRESS_RATIO, COFF * HEAD_DIM, generator=gen, device=device, dtype=torch.float32
    )
    ape = torch.empty(2 * COMPRESS_RATIO, HEAD_DIM, device=device, dtype=torch.float32)
    ape[:COMPRESS_RATIO] = raw_ape[:, :HEAD_DIM]
    ape[COMPRESS_RATIO:] = raw_ape[:, HEAD_DIM:]
    norm_weight = torch.randn(HEAD_DIM, generator=gen, device=device, dtype=torch.float32)
    rope_cos_sin = _make_rope_cos_sin(max(seq_lens) + 1, device)
    return kv_score, token_meta, ape, norm_weight, rope_cos_sin, num_c4_pages


def _make_meta_tensors(seq_lens, max_seq_length, max_num_pages, device):
    batch = len(seq_lens)
    qo = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    for i, seqlen in enumerate(seq_lens):
        qo[i + 1] = qo[i] + seqlen
    return {
        "step": torch.zeros(batch, dtype=torch.int32, device=device),
        "tokens": torch.zeros(batch, max_seq_length, dtype=torch.int64, device=device),
        "input_tokens": torch.zeros(sum(seq_lens), 1, dtype=torch.int64, device=device),
        "output_tokens": torch.zeros(sum(seq_lens), 1, dtype=torch.int64, device=device),
        "num_new_tokens": torch.tensor(seq_lens, dtype=torch.int32, device=device),
        "prompt_lengths": torch.tensor(seq_lens, dtype=torch.int32, device=device),
        "qo_indptr_buffer": qo,
        "paged_kv_indptr_buffer": torch.zeros(batch + 1, dtype=torch.int32, device=device),
        "paged_kv_indices_buffer": torch.zeros(max_num_pages, dtype=torch.int32, device=device),
        "paged_kv_last_page_len_buffer": torch.ones(batch, dtype=torch.int32, device=device),
        "paged_kv_indices_snapshot": torch.zeros(max_num_pages, dtype=torch.int32, device=device),
    }


def _run_mpk_once(seq_lens):
    _skip_without_sm100_and_nvcc()
    try:
        from mirage.mpk.persistent_kernel import PersistentKernel
        from mirage.mpk.layers.deepseek_v4 import DeepSeekV4C4Compressor
    except ImportError as exc:
        pytest.skip(f"MPK Python extension is not importable: {exc}")

    device = "cuda"
    kv_score, token_meta, ape, norm_weight, rope_cos_sin, num_c4_pages = _make_case(seq_lens, device)
    batch = len(seq_lens)
    max_seq_length = max(seq_lens)
    max_num_pages = max(1, num_c4_pages)
    state_cache = torch.zeros(batch, 8, KV_SCORE_DIM, dtype=torch.float32, device=device)
    c4_cache = torch.zeros(num_c4_pages, C4_PAGE_SIZE, HEAD_DIM, dtype=torch.bfloat16, device=device)

    meta_tensors = _make_meta_tensors(seq_lens, max_seq_length, max_num_pages, device)
    mpk = PersistentKernel(
        mode="offline",
        world_size=1,
        mpi_rank=0,
        num_workers=1,
        num_local_schedulers=1,
        num_remote_schedulers=0,
        max_seq_length=max_seq_length,
        max_num_batched_requests=batch,
        max_num_batched_tokens=sum(seq_lens),
        max_num_pages=max_num_pages,
        page_size=C4_PAGE_SIZE,
        meta_tensors=meta_tensors,
        profiler_tensor=None,
        trace_name="dsv4_c4_compressor_test",
        spec_decode_config=None,
        use_cutlass_kernel=False,
        test_mode=True,
    )
    try:
        layer = DeepSeekV4C4Compressor()
        layer.compile(
            mpk,
            kv_score=mpk.attach_input(kv_score, "dsv4_c4_kv_score"),
            token_meta=mpk.attach_input(token_meta, "dsv4_c4_token_meta"),
            state_cache=mpk.attach_input(state_cache, "dsv4_c4_state_cache"),
            c4_cache=mpk.attach_input(c4_cache, "dsv4_c4_cache"),
            ape=mpk.attach_input(ape, "dsv4_c4_ape"),
            norm_weight=mpk.attach_input(norm_weight, "dsv4_c4_norm_weight"),
            rope_cos_sin=mpk.attach_input(rope_cos_sin, "dsv4_c4_rope_cos_sin"),
            grid_dim=(batch, 1, 1),
            block_dim=(128, 1, 1),
        )
        mpk.compile()
        mpk.run_test_mode()
        torch.cuda.synchronize()
    finally:
        if getattr(mpk, "_is_compiled", False) and not getattr(mpk, "__finalized__", False):
            mpk.finalize()

    return kv_score, token_meta, ape, norm_weight, rope_cos_sin, state_cache, c4_cache


def _run_mpk_incremental_decode(total_tokens=16):
    _skip_without_sm100_and_nvcc()
    try:
        from mirage.mpk.persistent_kernel import PersistentKernel
        from mirage.mpk.layers.deepseek_v4 import DeepSeekV4C4Compressor
    except ImportError as exc:
        pytest.skip(f"MPK Python extension is not importable: {exc}")

    device = "cuda"
    full_seq_lens = [total_tokens]
    full_kv_score, full_token_meta, ape, norm_weight, rope_cos_sin, num_c4_pages = _make_case(
        full_seq_lens, device
    )
    kv_score = torch.empty(1, KV_SCORE_DIM, dtype=torch.float32, device=device)
    token_meta = torch.empty(1, 2, dtype=torch.int32, device=device)
    state_cache = torch.zeros(1, 8, KV_SCORE_DIM, dtype=torch.float32, device=device)
    c4_cache = torch.zeros(num_c4_pages, C4_PAGE_SIZE, HEAD_DIM, dtype=torch.bfloat16, device=device)

    meta_tensors = _make_meta_tensors([1], total_tokens, max(1, num_c4_pages), device)
    mpk = PersistentKernel(
        mode="offline",
        world_size=1,
        mpi_rank=0,
        num_workers=1,
        num_local_schedulers=1,
        num_remote_schedulers=0,
        max_seq_length=total_tokens,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=max(1, num_c4_pages),
        page_size=C4_PAGE_SIZE,
        meta_tensors=meta_tensors,
        profiler_tensor=None,
        trace_name="dsv4_c4_compressor_decode_test",
        spec_decode_config=None,
        use_cutlass_kernel=False,
        test_mode=True,
    )
    try:
        layer = DeepSeekV4C4Compressor()
        layer.compile(
            mpk,
            kv_score=mpk.attach_input(kv_score, "dsv4_c4_decode_kv_score"),
            token_meta=mpk.attach_input(token_meta, "dsv4_c4_decode_token_meta"),
            state_cache=mpk.attach_input(state_cache, "dsv4_c4_decode_state_cache"),
            c4_cache=mpk.attach_input(c4_cache, "dsv4_c4_decode_cache"),
            ape=mpk.attach_input(ape, "dsv4_c4_decode_ape"),
            norm_weight=mpk.attach_input(norm_weight, "dsv4_c4_decode_norm_weight"),
            rope_cos_sin=mpk.attach_input(rope_cos_sin, "dsv4_c4_decode_rope_cos_sin"),
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        mpk.compile()
        for pos in range(total_tokens):
            kv_score.copy_(full_kv_score[pos : pos + 1])
            token_meta.copy_(full_token_meta[pos : pos + 1])
            mpk.run_test_mode()
        torch.cuda.synchronize()
    finally:
        if getattr(mpk, "_is_compiled", False) and not getattr(mpk, "__finalized__", False):
            mpk.finalize()

    return full_kv_score, full_token_meta, ape, norm_weight, rope_cos_sin, state_cache, c4_cache


@pytest.mark.parametrize("seq_lens", ([3], [4], [7], [8], [3, 4], [3, 4, 7, 8]))
def test_reference_prefill_covers_deepseek_v4_c4_cases(seq_lens):
    device = "cpu"
    kv_score, token_meta, ape, norm_weight, rope_cos_sin, _ = _make_case(seq_lens, device)
    state, cache = _reference_prefill(kv_score, seq_lens, token_meta, ape, norm_weight, rope_cos_sin)

    assert state.shape == (len(seq_lens), 8, KV_SCORE_DIM)
    assert cache.shape[-1] == HEAD_DIM
    expected_writes = sum(seqlen // COMPRESS_RATIO for seqlen in seq_lens)
    assert int((token_meta[:, 1] >= 0).sum().item()) == expected_writes


def test_reference_decode_writes_only_on_c4_boundaries():
    seq_lens = [16]
    device = "cpu"
    kv_score, token_meta, ape, norm_weight, rope_cos_sin, _ = _make_case(seq_lens, device)
    state, cache = _reference_decode(kv_score, seq_lens, token_meta, ape, norm_weight, rope_cos_sin)

    emitted_positions = token_meta[token_meta[:, 1] >= 0, 0].tolist()
    assert emitted_positions == [3, 7, 11, 15]
    assert state.shape == (1, 8, KV_SCORE_DIM)
    assert cache.shape == (4, HEAD_DIM)


def test_mpk_dsv4_c4_compress_registration_smoke():
    _skip_without_sm100_and_nvcc()
    _run_mpk_once([1])


def test_mpk_dsv4_c4_prefill_matches_deepseek_reference():
    seq_lens = [3, 4, 7, 8]
    kv_score, token_meta, ape, norm_weight, rope_cos_sin, state_cache, c4_cache = _run_mpk_once(seq_lens)
    expected_state, expected_cache = _reference_prefill(
        kv_score, seq_lens, token_meta, ape, norm_weight, rope_cos_sin
    )
    actual_cache = c4_cache.reshape(-1, HEAD_DIM)[: expected_cache.shape[0]].float()

    torch.testing.assert_close(state_cache, expected_state, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=2e-2, atol=2e-2)


def test_mpk_dsv4_c4_decode_matches_deepseek_reference():
    seq_lens = [16]
    kv_score, token_meta, ape, norm_weight, rope_cos_sin, state_cache, c4_cache = (
        _run_mpk_incremental_decode(total_tokens=seq_lens[0])
    )
    expected_state, expected_cache = _reference_decode(
        kv_score, seq_lens, token_meta, ape, norm_weight, rope_cos_sin
    )
    actual_cache = c4_cache.reshape(-1, HEAD_DIM)[: expected_cache.shape[0]].float()

    torch.testing.assert_close(state_cache, expected_state, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=2e-2, atol=2e-2)
