"""The sliding window's leading-tile skip, on a request with a cached prefix.

When a window layer decodes, seq_len is long and only the last few KV tiles
are inside the window, so the kernel starts its KV loop past the leading
tiles. Four things move with that loop start and three of them are silent if
wrong: the prologue's page index, the double-buffer phase (counted from the
first tile VISITED, not tile 0), and the "apply Q RoPE once" guard.

This needs num_tokens < seq_len, which the megakernel's test mode cannot
produce -- it resets request_ids to -1 at init, so every request is admitted
fresh and prefills whole. Hence the direct launcher, which takes the page
table verbatim.

Build first:  python setup.py build_ext --inplace
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime_kernel_paged_attention as rk

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
# indirection rather than assuming page i holds tokens [64i, 64i+64).
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


def main():
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

    ok = True
    causal_ref = None
    for window_size in WINDOWS:
        k_cache = torch.randn(MAX_NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                              dtype=dtype, device=device)
        v_cache = torch.randn_like(k_cache)
        k_before, v_before = k_cache.clone(), v_cache.clone()
        out = torch.zeros(NUM_TOKENS, NUM_Q_HEADS * HEAD_DIM,
                          dtype=dtype, device=device)

        rk.paged_attention_sm100(qkv, k_cache, v_cache, out, qo_indptr,
                                 kv_indptr, kv_indices, kv_last, norm_w,
                                 norm_w, cos, sin, window_size)
        torch.cuda.synchronize()

        n = skipped_tiles(window_size)
        ref = reference(qkv, k_before, v_before, window_size)
        diff = (out.float() - ref.float()).abs().max().item()
        print(f"[w={window_size}] skips {n} tile(s) "
              f"({'even' if n % 2 == 0 else 'odd'}), "
              f"max |kernel - reference| = {diff:.4f}")
        if diff >= 0.05:
            print(f"[w={window_size}] FAILED: disagrees with the reference")
            ok = False

        if window_size == 0:
            causal_ref = ref
        else:
            gap = (out.float() - causal_ref.float()).abs().max().item()
            print(f"[w={window_size}] max |kernel - causal reference| = {gap:.4f}")
            if gap <= 0.05:
                print(f"[w={window_size}] FAILED: matches full causal, so the "
                      f"window is being ignored")
                ok = False

        # The skip must never run past the tiles carrying the new tokens.
        if not torch.equal(gather(k_cache, SEQ_LEN)[prefix:], k_new):
            print(f"[w={window_size}] FAILED: new K rows never reached the cache")
            ok = False
        if not torch.equal(gather(v_cache, SEQ_LEN)[prefix:], v_new):
            print(f"[w={window_size}] FAILED: new V rows never reached the cache")
            ok = False

    if not ok:
        sys.exit(1)
    print("\nPASSED: the leading-tile skip agrees with the reference at both "
          "parities and still fills the KV cache")


if __name__ == "__main__":
    main()
