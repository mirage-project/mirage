"""Test-mode coverage for attention sinks in paged attention (SM100).

A sink is an extra softmax logit that carries no value, so it only enters the
denominator. Three cases as three layers of one task graph:

  nosink  no sinks input at all -- the reference path and no-regression control
  inert   sinks negative enough that exp(sink - m) underflows to zero, so they
          must reproduce `nosink`. Zero would not: zero is a real logit.
  real    distinct random sinks, one per query head. Must match a reference
          that concatenates the sink logit and drops its column, and must
          differ from `nosink`.

Sinks differ per head, so a wrong head index fails.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

NUM_KV_HEADS = 1
NUM_QO_PER_KV = 8          # GQA 8:1, as in GPT-OSS
NUM_Q_HEADS = NUM_KV_HEADS * NUM_QO_PER_KV
HEAD_DIM = 64
PAGE_SIZE = 64
MAX_NUM_PAGES = 4
MAX_SEQ_LENGTH = 256
NUM_TOKENS = 8


def reference(qkv, sinks):
    """Causal GQA over a pure prefill, with an optional per-head sink logit."""
    q = qkv[:, : NUM_Q_HEADS * HEAD_DIM].view(NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM)
    k = qkv[:, NUM_Q_HEADS * HEAD_DIM : (NUM_Q_HEADS + 1) * HEAD_DIM]
    v = qkv[:, (NUM_Q_HEADS + 1) * HEAD_DIM :]

    scores = torch.einsum("thd,sd->ths", q.float(), k.float())
    scores = scores / (HEAD_DIM ** 0.5)
    pos = torch.arange(NUM_TOKENS, device=qkv.device)
    scores = scores.masked_fill(~(pos[None, :] <= pos[:, None])[:, None, :],
                                float("-inf"))

    if sinks is None:
        probs = torch.softmax(scores, dim=-1)
    else:
        # Same shape as HF: concatenate the sink logit, softmax, drop it.
        sink_col = sinks.float().reshape(1, NUM_Q_HEADS, 1).expand(
            NUM_TOKENS, NUM_Q_HEADS, 1)
        probs = torch.softmax(torch.cat([scores, sink_col], dim=-1), dim=-1)
        probs = probs[..., :-1]

    out = torch.einsum("ths,sd->thd", probs, v.float())
    return out.reshape(NUM_TOKENS, NUM_Q_HEADS * HEAD_DIM).to(qkv.dtype)


def main():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

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

    # cos = 1, sin = 0 makes the kernel's RoPE the identity.
    cos = torch.ones(MAX_SEQ_LENGTH, HEAD_DIM, dtype=dtype, device=device)
    sin = torch.zeros(MAX_SEQ_LENGTH, HEAD_DIM, dtype=dtype, device=device)
    norm_w = torch.ones(HEAD_DIM, dtype=dtype, device=device)
    cos_dt = pk.attach_input(cos, name="cos")
    sin_dt = pk.attach_input(sin, name="sin")
    norm_dt = pk.attach_input(norm_w, name="dummy_norm")

    # One shared qkv so the three cases are directly comparable.
    qkv = torch.randn(NUM_TOKENS, (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM,
                      dtype=dtype, device=device)
    qkv_dt = pk.attach_input(qkv, name="qkv")

    sink_values = {
        "nosink": None,
        "inert": torch.full((NUM_KV_HEADS, NUM_QO_PER_KV), -1e4,
                            dtype=dtype, device=device),
        "real": torch.randn(NUM_KV_HEADS, NUM_QO_PER_KV,
                            dtype=dtype, device=device) * 2.0,
    }

    cases = []
    for tag, sinks in sink_values.items():
        k_cache = torch.zeros(MAX_NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM,
                              dtype=dtype, device=device)
        v_cache = torch.zeros_like(k_cache)
        out = torch.zeros(NUM_TOKENS, NUM_Q_HEADS * HEAD_DIM,
                          dtype=dtype, device=device)
        sinks_dt = (pk.attach_input(sinks, name=f"{tag}_sinks")
                    if sinks is not None else None)

        pk.paged_attention_layer(
            input=qkv_dt,
            k_cache=pk.attach_input(k_cache, name=f"{tag}_k_cache"),
            v_cache=pk.attach_input(v_cache, name=f"{tag}_v_cache"),
            q_norm=norm_dt, k_norm=norm_dt,
            cos_pos_embed=cos_dt, sin_pos_embed=sin_dt,
            output=pk.attach_input(out, name=f"{tag}_out"),
            grid_dim=(1, NUM_KV_HEADS, 1), block_dim=(256, 1, 1),
            enable_qk_norm=False,
            sinks=sinks_dt,
        )
        cases.append((tag, sinks, out))

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ok = True
    nosink_out = None
    for tag, sinks, out in cases:
        flat = None if sinks is None else sinks.reshape(-1)
        ref = reference(qkv, flat)
        diff = (out.float() - ref.float()).abs().max().item()
        print(f"[{tag}] max |kernel - reference| = {diff:.4f}")
        if diff >= 0.05:
            print(f"[{tag}] FAILED: disagrees with the reference")
            ok = False

        if tag == "nosink":
            nosink_out = out.clone()
            continue

        gap = (out.float() - nosink_out.float()).abs().max().item()
        if tag == "inert":
            print(f"[inert] max |kernel - nosink| = {gap:.4f} (want ~0)")
            if gap >= 1e-3:
                print("[inert] FAILED: an underflowing sink changed the result")
                ok = False
        else:
            print(f"[real] max |kernel - nosink| = {gap:.4f} (want > 0)")
            if gap <= 0.05:
                print("[real] FAILED: the sinks are being ignored")
                ok = False

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("\nPASSED: attention sinks enter the softmax denominator per head, "
          "an underflowing sink is inert, and no-sink is unchanged")


if __name__ == "__main__":
    main()
