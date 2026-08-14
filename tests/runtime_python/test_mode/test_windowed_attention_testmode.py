"""Test-mode coverage for the sliding-window mask in paged attention (SM100).

Each window is checked two ways: it matches a windowed reference, and it
differs from the plain-causal one, so an ignored WINDOW_SIZE fails.

window=0 is the no-regression control on the full-causal path, and confirms
the identity RoPE tables (cos=1, sin=0) used here are the identity.

SCOPE: the MASK only. Skipping leading KV tiles that fall outside the window
needs seq_len > num_tokens and is covered by test_windowed_attention_direct.py.
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
MAX_NUM_PAGES = 4
MAX_SEQ_LENGTH = 256
NUM_TOKENS = 8             # = max_num_batched_tokens = seq_len here
WINDOWS = (0, 4, 6)


def reference(qkv, window_size):
    """Windowed causal GQA over a pure prefill of NUM_TOKENS tokens."""
    q = qkv[:, : NUM_Q_HEADS * HEAD_DIM].view(NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM)
    k = qkv[:, NUM_Q_HEADS * HEAD_DIM : (NUM_Q_HEADS + 1) * HEAD_DIM]
    v = qkv[:, (NUM_Q_HEADS + 1) * HEAD_DIM :]

    scores = torch.einsum("thd,sd->ths", q.float(), k.float())
    scores = scores / (HEAD_DIM ** 0.5)

    pos = torch.arange(NUM_TOKENS, device=qkv.device)
    keep = pos[None, :] <= pos[:, None]
    if window_size > 0:
        keep &= pos[None, :] > pos[:, None] - window_size

    scores = scores.masked_fill(~keep[:, None, :], float("-inf"))
    out = torch.einsum("ths,sd->thd", torch.softmax(scores, dim=-1), v.float())
    return out.reshape(NUM_TOKENS, NUM_Q_HEADS * HEAD_DIM).to(qkv.dtype)


def main():
    torch.manual_seed(0)
    device = "cuda"

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        max_seq_length=MAX_SEQ_LENGTH,
        max_num_batched_requests=1,
        max_num_batched_tokens=NUM_TOKENS,
        max_num_pages=MAX_NUM_PAGES,
        page_size=PAGE_SIZE,
    )
    params["meta_tensors"] = {
        "prompt_lengths": torch.tensor([NUM_TOKENS], dtype=torch.int32,
                                       device=device),
    }
    pk = PersistentKernel(**params)

    # cos = 1, sin = 0 makes the kernel's RoPE the identity, so the reference
    # does not model it.
    cos = torch.ones(MAX_SEQ_LENGTH, HEAD_DIM, dtype=torch.bfloat16,
                     device=device)
    sin = torch.zeros(MAX_SEQ_LENGTH, HEAD_DIM, dtype=torch.bfloat16,
                      device=device)
    norm_w = torch.ones(HEAD_DIM, dtype=torch.bfloat16, device=device)
    cos_dt = pk.attach_input(cos, name="cos")
    sin_dt = pk.attach_input(sin, name="sin")
    norm_dt = pk.attach_input(norm_w, name="dummy_norm")

    cases = []
    for window_size in WINDOWS:
        tag = f"w{window_size}"
        qkv = torch.randn(NUM_TOKENS,
                          (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM,
                          dtype=torch.bfloat16, device=device)
        k_cache = torch.zeros(MAX_NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                              dtype=torch.bfloat16, device=device)
        v_cache = torch.zeros_like(k_cache)
        out = torch.zeros(NUM_TOKENS, NUM_Q_HEADS * HEAD_DIM,
                          dtype=torch.bfloat16, device=device)

        pk.paged_attention_layer(
            input=pk.attach_input(qkv, name=f"{tag}_qkv"),
            k_cache=pk.attach_input(k_cache, name=f"{tag}_k_cache"),
            v_cache=pk.attach_input(v_cache, name=f"{tag}_v_cache"),
            q_norm=norm_dt, k_norm=norm_dt,
            cos_pos_embed=cos_dt, sin_pos_embed=sin_dt,
            output=pk.attach_input(out, name=f"{tag}_out"),
            grid_dim=(1, NUM_KV_HEADS, 1), block_dim=(256, 1, 1),
            enable_qk_norm=False,
            window_size=window_size,
        )
        cases.append((window_size, qkv, k_cache, out))

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ok = True
    causal_ref = None
    for window_size, qkv, k_cache, out in cases:
        ref = reference(qkv, window_size)
        diff = (out.float() - ref.float()).abs().max().item()
        print(f"[w={window_size}] max |kernel - reference| = {diff:.4f}")
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

        # The new tokens are the whole sequence, so they land in the first
        # page, rows 0..NUM_TOKENS.
        written = k_cache[:, :NUM_TOKENS, 0].reshape(-1, HEAD_DIM)
        k_new = qkv[:, NUM_Q_HEADS * HEAD_DIM : (NUM_Q_HEADS + 1) * HEAD_DIM]
        if not any(torch.equal(k_cache[p, :NUM_TOKENS, 0], k_new)
                   for p in range(MAX_NUM_PAGES)):
            print(f"[w={window_size}] FAILED: new K rows never reached the "
                  f"paged cache (written sample {written[0, :4].tolist()})")
            ok = False

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("\nPASSED: the sliding-window mask matches the reference, differs "
          "from full causal, and the KV cache is still filled")


if __name__ == "__main__":
    main()
