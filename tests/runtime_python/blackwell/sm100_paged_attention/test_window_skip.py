"""The sliding window's leading-tile skip, on a request with a cached prefix.

Decoding a window layer starts the KV loop past the leading tiles. Three
things depend on that loop start: the prologue's page index, the double-buffer
phase (counted from the first tile visited, not tile 0), and the "apply Q RoPE
once" guard. This fixture checks addressing and buffer parity with identity
RoPE. Detecting missing or repeated rotation requires a nonidentity fixture.

This needs num_tokens < seq_len, which the megakernel's test mode cannot
produce -- it resets request_ids to -1 at init, so every request prefills
whole. Hence the direct launcher, which takes the page table verbatim.

Build first:  python setup.py build_ext --inplace
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NUM_KV_HEADS = 1
NUM_QO_PER_KV = 8
NUM_Q_HEADS = NUM_KV_HEADS * NUM_QO_PER_KV
HEAD_DIM = 64
PAGE_SIZE = 64
MAX_NUM_PAGES = 8
MAX_SEQ_LEN = 256
KV_TILE_SIZE = 64          # must match the kernel's tile
NUM_TOKENS = 8             # new tokens this call
SEQ_LEN = 200              # cached prefix + the new tokens
# A non-identity page table proves the skip still resolves pages through the
# indirection, never assuming page i holds tokens [64i, 64i+64).
PAGE_TABLE = [5, 2, 7, 1]
WINDOWS = (0, 96, 32)      # skips 0, 1 (odd) and 2 (even) leading tiles


def skipped_tiles(window_size):
    if window_size <= 0:
        return 0
    return max(SEQ_LEN - NUM_TOKENS - window_size + 1, 0) // KV_TILE_SIZE


def gather(cache, num_rows):
    """Flatten the paged cache into contiguous [num_rows, HEAD_DIM]."""
    out = torch.empty(num_rows, HEAD_DIM, dtype=cache.dtype, device=cache.device)
    for pos in range(num_rows):
        out[pos] = cache[PAGE_TABLE[pos // PAGE_SIZE], pos % PAGE_SIZE, 0]
    return out


def reference(qkv, k_cache, v_cache, window_size):
    q = qkv[:, : NUM_Q_HEADS * HEAD_DIM].view(NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM)
    k_new = qkv[:, NUM_Q_HEADS * HEAD_DIM : (NUM_Q_HEADS + 1) * HEAD_DIM]
    v_new = qkv[:, (NUM_Q_HEADS + 1) * HEAD_DIM :]

    prefix = SEQ_LEN - NUM_TOKENS
    k = torch.cat([gather(k_cache, prefix), k_new], dim=0)
    v = torch.cat([gather(v_cache, prefix), v_new], dim=0)

    scores = torch.einsum("thd,sd->ths", q.float(), k.float()) / (HEAD_DIM ** 0.5)
    key_pos = torch.arange(SEQ_LEN, device=qkv.device)
    query_pos = torch.arange(prefix, SEQ_LEN, device=qkv.device)
    keep = key_pos[None, :] <= query_pos[:, None]
    if window_size > 0:
        keep &= key_pos[None, :] > query_pos[:, None] - window_size

    scores = scores.masked_fill(~keep[:, None, :], float("-inf"))
    out = torch.einsum("ths,sd->thd", torch.softmax(scores, dim=-1), v.float())
    return out.reshape(NUM_TOKENS, NUM_Q_HEADS * HEAD_DIM).to(qkv.dtype)


def suffix_write_mask(device):
    """Physical cache rows the eight suffix tokens are allowed to overwrite."""
    mask = torch.zeros(MAX_NUM_PAGES, PAGE_SIZE, dtype=torch.bool, device=device)
    for pos in range(SEQ_LEN - NUM_TOKENS, SEQ_LEN):
        mask[PAGE_TABLE[pos // PAGE_SIZE], pos % PAGE_SIZE] = True
    return mask


def check_cache(cache, before, expected_suffix, write_mask):
    if not torch.isfinite(cache).all().item():
        raise AssertionError("cache contains nonfinite values or unwritten poison")
    if not torch.equal(cache[~write_mask], before[~write_mask]):
        raise AssertionError("cache changed outside the physical suffix rows")
    if not torch.equal(gather(cache, SEQ_LEN)[-NUM_TOKENS:], expected_suffix):
        raise AssertionError("new suffix rows disagree with the input")


def test_cache_validation_cpu():
    """The guard accepts exact writes and rejects unused-page or NaN damage."""
    mask = suffix_write_mask("cpu")
    assert mask.sum().item() == NUM_TOKENS
    assert mask[1, :8].all().item() and not mask[1, 8:].any().item()
    before = torch.zeros(MAX_NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
    suffix = torch.ones(NUM_TOKENS, HEAD_DIM)
    valid = before.clone()
    valid[mask] = 1
    check_cache(valid, before, suffix, mask)
    for damage in ("unused_page", "unwritten_suffix", "wrong_suffix"):
        broken = valid.clone()
        if damage == "unused_page":
            broken[0, 0] = 2
        elif damage == "unwritten_suffix":
            broken[1, 0] = float("nan")
        else:
            broken[1, 0] = 2
        try:
            check_cache(broken, before, suffix, mask)
        except AssertionError:
            continue
        raise AssertionError(f"guard missed {damage}")


def main():
    import runtime_kernel_paged_attention as rk
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    prefix = SEQ_LEN - NUM_TOKENS
    num_pages = (SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    assert num_pages == len(PAGE_TABLE)

    qo_indptr = torch.tensor([0, NUM_TOKENS], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    kv_indices = torch.tensor(PAGE_TABLE, dtype=torch.int32, device=device)
    kv_last = torch.tensor([SEQ_LEN - (num_pages - 1) * PAGE_SIZE],
                           dtype=torch.int32, device=device)
    # cos = 1, sin = 0 makes RoPE the identity; the window=0 case proves it.
    cos = torch.ones(MAX_SEQ_LEN, HEAD_DIM, dtype=dtype, device=device)
    sin = torch.zeros(MAX_SEQ_LEN, HEAD_DIM, dtype=dtype, device=device)
    norm_w = torch.ones(HEAD_DIM, dtype=dtype, device=device)

    qkv = torch.randn(NUM_TOKENS, (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM,
                      dtype=dtype, device=device)
    k_new = qkv[:, NUM_Q_HEADS * HEAD_DIM : (NUM_Q_HEADS + 1) * HEAD_DIM]
    v_new = qkv[:, (NUM_Q_HEADS + 1) * HEAD_DIM :]

    write_mask = suffix_write_mask(device)
    readonly = [qkv, qo_indptr, kv_indptr, kv_indices, kv_last, norm_w, cos, sin]
    readonly_before = [value.clone() for value in readonly]
    for window_size in WINDOWS:
        k_cache = torch.randn(MAX_NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                              dtype=dtype, device=device)
        v_cache = torch.randn_like(k_cache)
        k_before, v_before = k_cache.clone(), v_cache.clone()
        ref = reference(qkv, k_before, v_before, window_size)
        # Use the SAME cache as this window, rather than another random case.
        causal_ref = reference(qkv, k_before, v_before, 0)
        out = torch.empty(NUM_TOKENS, NUM_Q_HEADS * HEAD_DIM,
                          dtype=dtype, device=device)
        first_output = None
        for invocation in range(2):
            # Keep the actual prefix and unused pages across invocations. Force
            # the task to write every output and new KV row again.
            out.fill_(float("nan"))
            k_cache[write_mask] = float("nan")
            v_cache[write_mask] = float("nan")
            rk.paged_attention_sm100(qkv, k_cache, v_cache, out, qo_indptr,
                                     kv_indptr, kv_indices, kv_last, norm_w,
                                     norm_w, cos, sin, window_size)
            torch.cuda.synchronize()

            if not torch.isfinite(out).all().item():
                raise AssertionError(f"w={window_size}: nonfinite output")
            if not torch.isfinite(ref).all().item():
                raise AssertionError("nonfinite reference")
            diff = (out.float() - ref.float()).abs().max().item()
            n = skipped_tiles(window_size)
            print(f"[w={window_size}, call={invocation}] skips {n} tile(s), "
                  f"max |kernel - reference| = {diff:.4f}")
            if diff >= 0.05:
                raise AssertionError("output disagrees with the reference")
            if window_size:
                gap = (out.float() - causal_ref.float()).abs().max().item()
                if gap <= 0.05:
                    raise AssertionError("window output matches full causal")
            check_cache(k_cache, k_before, k_new, write_mask)
            check_cache(v_cache, v_before, v_new, write_mask)
            for value, before in zip(readonly, readonly_before):
                if not torch.equal(value, before):
                    raise AssertionError("read-only input or metadata changed")
            if first_output is None:
                first_output = out.clone()
            elif not torch.equal(out, first_output):
                raise AssertionError("identical repeated invocation changed output")

    print("\nPASSED: leading-tile skip outputs and physical cache writes "
          "remain correct across repeated calls at both skip parities")


if __name__ == "__main__":
    main()
