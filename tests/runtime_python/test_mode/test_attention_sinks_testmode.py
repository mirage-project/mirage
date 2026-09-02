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

The three cases are three LAYERS of one KV stream, so they share a page table
and take three slots of the same page -- the shape eagle3's draft and
DeepSeek-V3's MTP layer have. That makes this the place where slot isolation
for paged_attention_layer is checked: each case's K must land in its own slot
and nowhere else.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.kv_planner import KVStream, build_kv_cache
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

    # One stream over three layers: same geometry, so one group with three
    # slots on every page.
    entry = (NUM_KV_HEADS, HEAD_DIM)
    plan = build_kv_cache(
        [KVStream("attention", layers=(0, 1, 2),
                  components=[("k", entry, dtype), ("v", entry, dtype)],
                  preferred_block_size=PAGE_SIZE)],
        max_num_pages=MAX_NUM_PAGES,
        max_seq_length=MAX_SEQ_LENGTH,
        max_num_batched_requests=1,
        max_num_batched_tokens=NUM_TOKENS,
        verbose=False)
    assert len(plan.groups) == 1 and plan.num_slots == 3, (
        f"expected one group of three slots, got {len(plan.groups)} group(s) "
        f"and {plan.num_slots} slot(s)")

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
    )
    params["meta_tensors"] = {
        "prompt_lengths": torch.tensor([NUM_TOKENS], dtype=torch.int32,
                                       device=device),
        **plan.build_meta_tensors(max_num_batched_requests=1,
                                  max_seq_length=MAX_SEQ_LENGTH),
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
    for layer_id, (tag, sinks) in enumerate(sink_values.items()):
        out = torch.zeros(NUM_TOKENS, NUM_Q_HEADS * HEAD_DIM,
                          dtype=dtype, device=device)
        sinks_dt = (pk.attach_input(sinks, name=f"{tag}_sinks")
                    if sinks is not None else None)

        kv = plan.attach(pk, layer_id)
        pk.paged_attention_layer(
            input=qkv_dt,
            k_cache=kv["k_cache"], v_cache=kv["v_cache"],
            group_id=kv["group_id"],
            q_norm=norm_dt, k_norm=norm_dt,
            cos_pos_embed=cos_dt, sin_pos_embed=sin_dt,
            output=pk.attach_input(out, name=f"{tag}_out"),
            grid_dim=(1, NUM_KV_HEADS, 1), block_dim=(256, 1, 1),
            enable_qk_norm=False,
            sinks=sinks_dt,
        )
        cases.append((tag, sinks, out, plan._layer_info(layer_id)[1]))

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ok = True
    nosink_out = None
    k_view = plan.views(0)["k"]
    k_new = qkv[:, NUM_Q_HEADS * HEAD_DIM : (NUM_Q_HEADS + 1) * HEAD_DIM]
    for tag, sinks, out, slot in cases:
        flat = None if sinks is None else sinks.reshape(-1)
        ref = reference(qkv, flat)
        diff = (out.float() - ref.float()).abs().max().item()
        print(f"[{tag}] max |kernel - reference| = {diff:.4f}")
        if diff >= 0.05:
            print(f"[{tag}] FAILED: disagrees with the reference")
            ok = False

        # Slot isolation: every case shares the page table and writes the
        # same K rows, so a slot resolved wrong would still look right in the
        # output. It shows up here: the rows must be in THIS layer's slot.
        if not torch.equal(k_view[slot, 0, :NUM_TOKENS, 0], k_new):
            print(f"[{tag}] FAILED: K rows are not in slot {slot} of page 0")
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

    # Nothing outside the three slots may hold data, and each slot holds only
    # its own layer's rows.
    for slot in range(plan.num_slots):
        rows = int((k_view[slot].abs().sum(dim=-1) > 0).sum().item())
        if rows != NUM_TOKENS:
            print(f"FAILED: slot {slot} holds {rows} K rows, expected "
                  f"{NUM_TOKENS} -- a layer wrote outside its own slot")
            ok = False

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("\nPASSED: attention sinks enter the softmax denominator per head, "
          "an underflowing sink is inert, no-sink is unchanged, and each "
          "layer's K stayed in its own slot of the shared page")


if __name__ == "__main__":
    main()
