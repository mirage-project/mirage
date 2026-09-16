"""Paged attention (SM100) against a KV cache that already holds history.

Every other test_mode attention test runs a pure prefill: the batch is built
by prepare_next_batch() from prompt_lengths, so the first (and only) task
graph sees seq_len == num_tokens and history_len == 0. Two behaviours are
invisible under that shape:

  1. Decode, and chunked prefill, reading K/V that were written by an earlier
     iteration rather than by this one.
  2. The sliding window skipping whole leading KV tiles. The kernel starts at
     first_kv_iter = (seq_len - num_tokens - WINDOW_SIZE + 1) / KV_TILE_SIZE,
     which is 0 unless seq_len > num_tokens. This is the gap
     test_windowed_attention_testmode.py calls out of scope.

test_inject_batch=True runs the qo_indptr / paged_kv_indptr / last_page_len
built below as the batch, so the cache can be seeded with history first.

RoPE is the identity here (cos=1, sin=0), so the reference does not model it.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

NUM_KV_HEADS = 1
NUM_QO_PER_KV = 8          # GQA 8:1
NUM_Q_HEADS = NUM_KV_HEADS * NUM_QO_PER_KV
HEAD_DIM = 64
PAGE_SIZE = 64
KV_TILE_SIZE = 64          # must match the kernel's tile size
TOL = 0.05

# (label, num_tokens, history_len, window_size)
CASES = [
    ("decode over history", 1, 128, 0),
    ("chunked prefill over history", 8, 128, 0),
    # seq_len - num_tokens - window + 1 = 256 - 8 - 64 + 1 = 185 -> skips 2
    # leading KV tiles, which is only reachable with history present.
    ("window skips leading KV tiles", 8, 248, 64),
]


def reference(q_new, k_hist, v_hist, k_new, v_new, window_size):
    """Causal (optionally windowed) GQA of the new tokens over history+new."""
    num_tokens = q_new.shape[0]
    history_len = k_hist.shape[0]
    seq_len = history_len + num_tokens

    k_full = torch.cat([k_hist, k_new], dim=0).float()   # [seq_len, D]
    v_full = torch.cat([v_hist, v_new], dim=0).float()

    scores = torch.einsum("thd,sd->ths", q_new.float(), k_full)
    scores = scores / (HEAD_DIM ** 0.5)

    key_pos = torch.arange(seq_len, device=q_new.device)
    query_pos = torch.arange(history_len, seq_len, device=q_new.device)
    keep = key_pos[None, :] <= query_pos[:, None]
    if window_size > 0:
        keep &= key_pos[None, :] > query_pos[:, None] - window_size

    scores = scores.masked_fill(~keep[:, None, :], float("-inf"))
    out = torch.einsum("ths,sd->thd", torch.softmax(scores, dim=-1), v_full)
    return out.reshape(num_tokens, NUM_Q_HEADS * HEAD_DIM).to(q_new.dtype)


def run_case(num_tokens, history_len, window_size):
    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16

    seq_len = history_len + num_tokens
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    last_page_len = seq_len - (num_pages - 1) * PAGE_SIZE
    max_seq_length = max(256, seq_len)
    max_num_pages = num_pages + 1

    q_new = torch.randn(num_tokens, NUM_Q_HEADS, HEAD_DIM,
                        dtype=dtype, device=device)
    k_new = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device)
    v_new = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device)
    k_hist = torch.randn(history_len, HEAD_DIM, dtype=dtype, device=device)
    v_hist = torch.randn(history_len, HEAD_DIM, dtype=dtype, device=device)

    # Packed QKV for the new tokens only: [q heads | k | v].
    qkv = torch.empty(num_tokens, (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM,
                      dtype=dtype, device=device)
    qkv[:, : NUM_Q_HEADS * HEAD_DIM] = q_new.reshape(num_tokens, -1)
    qkv[:, NUM_Q_HEADS * HEAD_DIM : (NUM_Q_HEADS + 1) * HEAD_DIM] = k_new
    qkv[:, (NUM_Q_HEADS + 1) * HEAD_DIM :] = v_new

    # Seed the cache with history; the kernel appends the new tokens itself.
    k_cache = torch.zeros(max_num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                          dtype=dtype, device=device)
    v_cache = torch.zeros_like(k_cache)
    k_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)[:history_len, 0] = k_hist
    v_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)[:history_len, 0] = v_hist

    out = torch.zeros(num_tokens, NUM_Q_HEADS * HEAD_DIM,
                      dtype=dtype, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        test_inject_batch=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        max_seq_length=max_seq_length,
        max_num_batched_requests=1,
        max_num_batched_tokens=num_tokens,
        max_num_pages=max_num_pages,
        page_size=PAGE_SIZE,
    )
    params["meta_tensors"] = {
        "prompt_lengths": torch.tensor([seq_len], dtype=torch.int32,
                                       device=device),
        "qo_indptr_buffer": torch.tensor([0, num_tokens], dtype=torch.int32,
                                         device=device),
        "paged_kv_indptr_buffer": torch.tensor([0, num_pages],
                                               dtype=torch.int32,
                                               device=device),
        "paged_kv_indices_buffer": torch.arange(num_pages, dtype=torch.int32,
                                                device=device),
        "paged_kv_last_page_len_buffer": torch.tensor([last_page_len],
                                                      dtype=torch.int32,
                                                      device=device),
    }
    pk = PersistentKernel(**params)

    cos = torch.ones(max_seq_length, HEAD_DIM, dtype=dtype, device=device)
    sin = torch.zeros(max_seq_length, HEAD_DIM, dtype=dtype, device=device)
    norm_w = torch.ones(HEAD_DIM, dtype=dtype, device=device)
    norm_dt = pk.attach_input(norm_w, name="dummy_norm")

    pk.paged_attention_layer(
        input=pk.attach_input(qkv, name="qkv"),
        k_cache=pk.attach_input(k_cache, name="k_cache"),
        v_cache=pk.attach_input(v_cache, name="v_cache"),
        q_norm=norm_dt, k_norm=norm_dt,
        cos_pos_embed=pk.attach_input(cos, name="cos"),
        sin_pos_embed=pk.attach_input(sin, name="sin"),
        output=pk.attach_input(out, name="out"),
        grid_dim=(1, NUM_KV_HEADS, 1), block_dim=(256, 1, 1),
        enable_qk_norm=False,
        window_size=window_size,
    )

    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    pk()
    torch.cuda.synchronize()

    ref = reference(q_new, k_hist, v_hist, k_new, v_new, window_size)
    causal_ref = reference(q_new, k_hist, v_hist, k_new, v_new, 0)
    result = out.clone()
    # The new K rows must have landed after the history, not over it.
    cached_k = k_cache.view(-1, NUM_KV_HEADS, HEAD_DIM)[:seq_len, 0].clone()
    pk.finalize()
    return result, ref, causal_ref, cached_k, k_hist, k_new


def main():
    ok = True
    for label, num_tokens, history_len, window_size in CASES:
        seq_len = history_len + num_tokens
        skipped = (max(seq_len - num_tokens - window_size + 1, 0) //
                   KV_TILE_SIZE) if window_size > 0 else 0
        print(f"\n[{label}] tokens={num_tokens} history={history_len} "
              f"window={window_size} (leading KV tiles skipped: {skipped})")

        result, ref, causal_ref, cached_k, k_hist, k_new = run_case(
            num_tokens, history_len, window_size)

        diff = (result.float() - ref.float()).abs().max().item()
        print(f"  max |kernel - reference| = {diff:.4f}")
        if diff >= TOL:
            print(f"  FAILED: disagrees with the reference")
            ok = False

        if window_size > 0:
            gap = (result.float() - causal_ref.float()).abs().max().item()
            print(f"  max |kernel - full-causal reference| = {gap:.4f}")
            if gap <= TOL:
                print(f"  FAILED: matches full causal, so the window (and the "
                      f"skipped leading tiles) is being ignored")
                ok = False

        if not torch.equal(cached_k[:history_len], k_hist):
            print(f"  FAILED: the seeded history was overwritten in the cache")
            ok = False
        if not torch.equal(cached_k[history_len:], k_new):
            print(f"  FAILED: new K rows did not land after the history")
            ok = False

    if not ok:
        sys.exit(1)
    print("\nPASSED: paged attention reads a pre-seeded KV cache, appends the "
          "new tokens after it, and skips leading KV tiles under a window")


if __name__ == "__main__":
    main()
