"""Test-mode coverage for the sliding-window mask in paged attention (SM100).

Each window is checked two ways: it matches a windowed reference, and it
differs from the plain-causal one, so an ignored WINDOW_SIZE fails.

window=0 is the no-regression control on the full-causal path, and confirms
the identity RoPE tables (cos=1, sin=0) used here are the identity.

Each window gets its OWN KV group. That is not a detail of the harness, it is
the rule: a group IS a page table, and prepare_next_batch frees a page once it
falls out of THE GROUP's window -- so two layers masking with different windows
cannot share one, or the shorter window reclaims pages the longer one is still
reading. paged_attention_layer enforces it (_resolve_kv_block_size checks that
the layer's window equals the group's), which is what this test used to violate
by running all three windows against a single default group; it had been
failing ever since that check landed. Do not collapse them back into one group.

SCOPE: the MASK only. Skipping leading KV tiles that fall outside the window
needs seq_len > num_tokens and is covered by test_windowed_attention_direct.py.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.kv_planner import KVStream, build_kv_cache
from mirage.mpk.persistent_kernel import PersistentKernel

NUM_KV_HEADS = 1
NUM_QO_PER_KV = 8          # GQA 8:1
NUM_Q_HEADS = NUM_KV_HEADS * NUM_QO_PER_KV
HEAD_DIM = 64
PAGE_SIZE = 64
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

    entry = (NUM_KV_HEADS, HEAD_DIM)
    plan = build_kv_cache(
        [KVStream(f"w{w}", layers=(i,),
                  components=[("k", entry, torch.bfloat16),
                              ("v", entry, torch.bfloat16)],
                  window=w, preferred_block_size=PAGE_SIZE)
         for i, w in enumerate(WINDOWS)],
        max_num_pages=4 * len(WINDOWS),
        max_seq_length=MAX_SEQ_LENGTH,
        max_num_batched_requests=1,
        max_num_batched_tokens=NUM_TOKENS,
        verbose=False)
    assert len(plan.groups) == len(WINDOWS), (
        f"each window needs its own page table, got {len(plan.groups)} "
        f"group(s) for {len(WINDOWS)} windows")

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        max_seq_length=MAX_SEQ_LENGTH,
        max_num_batched_requests=1,
        max_num_batched_tokens=NUM_TOKENS,
        max_num_pages=plan.max_num_pages,
        kv_groups=plan.group_specs(),
        page_size=None,
    )
    params["meta_tensors"] = {
        "prompt_lengths": torch.tensor([NUM_TOKENS], dtype=torch.int32,
                                       device=device),
        **plan.build_meta_tensors(max_num_batched_requests=1,
                                  max_seq_length=MAX_SEQ_LENGTH),
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
    for layer_id, window_size in enumerate(WINDOWS):
        tag = f"w{window_size}"
        qkv = torch.randn(NUM_TOKENS,
                          (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM,
                          dtype=torch.bfloat16, device=device)
        out = torch.zeros(NUM_TOKENS, NUM_Q_HEADS * HEAD_DIM,
                          dtype=torch.bfloat16, device=device)

        kv = plan.attach(pk, layer_id)
        group_id, slot = plan._layer_info(layer_id)
        pk.paged_attention_layer(
            input=pk.attach_input(qkv, name=f"{tag}_qkv"),
            k_cache=kv["k_cache"], v_cache=kv["v_cache"],
            q_norm=norm_dt, k_norm=norm_dt,
            cos_pos_embed=cos_dt, sin_pos_embed=sin_dt,
            output=pk.attach_input(out, name=f"{tag}_out"),
            grid_dim=(1, NUM_KV_HEADS, 1), block_dim=(256, 1, 1),
            enable_qk_norm=False,
            window_size=window_size,
            group_id=kv["group_id"],
        )
        cases.append((window_size, qkv, plan.views(group_id)["k"][slot], out))

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
        # page of the group's page table, rows 0..NUM_TOKENS.
        k_new = qkv[:, NUM_Q_HEADS * HEAD_DIM : (NUM_Q_HEADS + 1) * HEAD_DIM]
        if not any(torch.equal(k_cache[p, :NUM_TOKENS, 0], k_new)
                   for p in range(k_cache.shape[0])):
            print(f"[w={window_size}] FAILED: new K rows never reached the "
                  f"paged cache")
            ok = False

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("\nPASSED: the sliding-window mask matches the reference, differs "
          "from full causal, and the KV cache is still filled")


if __name__ == "__main__":
    main()
