"""Temperature / top-k / top-p sampling over an lm_head padded past the vocab.

Three sampling configurations share one compiled graph:

  greedy   temperature=0            must reproduce argmax over the real vocab
  top-k    temperature=1, top_k=3   every draw must land in torch's top-3
  top-p    temperature=1, top_p=0.6 every draw must land in torch's nucleus

Every real logit here is strictly negative while the padding region holds 0.0,
so a sampler that ignores ``vocab_size`` would draw a padding row for all three
configurations (the same hazard argmax hit in #751/#752/#755).

The three leading tokens are given near-identical logits so a working RNG
produces more than one distinct token across the batch; the rows all hold the
same logits, so the batch is a set of independent draws from one distribution.

    python tests/runtime_python/test_mode/test_sampling_topk_topp_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.core import float32
from mirage.mpk.persistent_kernel import PersistentKernel

BATCH = 64
PADDED = 153600
VOCAB = 151936
NUM_TASKS = 96  # PADDED divides evenly; chunk = 1600
TOPK_MAX = 32
SEED = 1234
# Each row draws from its own Philox offset, and the decode step (which the
# megakernel advances between iterations of a real run) is reset to 0 by
# init_kernel on every launch. Extra seeds are therefore how the test collects
# more than one batch worth of independent draws.
DIST_SEEDS = [11, 22, 33, 44]

# Scattered across chunks so the reduce stage has to merge candidates from
# several partial tasks. The first three are nearly tied.
SPECIAL_IDS = [5, 1600, 40001, 77777, 90000, 120005, 150400, 151935]
SPECIAL_LOGITS = [-1.0, -1.05, -1.1, -3.0, -3.5, -4.0, -5.0, -6.0]
BASE_LOGIT = -30.0


def build_logits(device, dtype):
    logits = torch.full((BATCH, PADDED), BASE_LOGIT, dtype=dtype, device=device)
    for tok, val in zip(SPECIAL_IDS, SPECIAL_LOGITS):
        logits[:, tok] = val
    logits[:, VOCAB:] = 0.0  # zero-padded lm_head rows beat every real logit
    return logits


def reference_sets(logits):
    """torch's greedy token, top-3 set and top-p nucleus over the real vocab."""
    probs = torch.softmax(logits[0, :VOCAB].float(), dim=-1)
    greedy = int(probs.argmax())
    topk = set(torch.topk(probs, 3).indices.tolist())
    order = torch.argsort(probs, descending=True)
    cum = torch.cumsum(probs[order], dim=0)
    n_nucleus = int((cum >= 0.6).nonzero()[0]) + 1
    nucleus = set(order[:n_nucleus].tolist())
    return greedy, topk, nucleus


def add_sampling(pk, logits_d, name, temperature, top_p, top_k, seed=SEED):
    part_val = torch.zeros(
        BATCH, NUM_TASKS * (TOPK_MAX + 2), dtype=torch.float32, device="cuda")
    part_idx = torch.zeros(
        BATCH, NUM_TASKS * TOPK_MAX, dtype=torch.int64, device="cuda")
    out = torch.zeros(BATCH, 1, dtype=torch.int64, device="cuda")
    # One DTensor per buffer: the partial->reduce dependency is inferred from
    # the shared tensor, so attaching the same buffer twice would drop the edge.
    d_val = pk.attach_input(part_val, name=f"{name}_val")
    d_idx = pk.attach_input(part_idx, name=f"{name}_idx")
    pk.sampling_partial_layer(
        input=logits_d, output=(d_val, d_idx),
        grid_dim=(NUM_TASKS, 1, 1), block_dim=(128, 1, 1),
        vocab_size=VOCAB, topk_max=TOPK_MAX, temperature=temperature)
    pk.sampling_reduce_layer(
        input=(d_val, d_idx),
        output=pk.attach_input(out, name=f"{name}_out"),
        grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
        temperature=temperature, top_p=top_p, top_k=top_k, seed=seed)
    return out


def main():
    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    qo_indptr = torch.zeros(BATCH + 1, dtype=torch.int32, device=device)
    qo_indptr[BATCH] = BATCH
    params.update(test_mode=True, num_workers=num_workers,
                  num_local_schedulers=num_schedulers,
                  max_num_batched_tokens=BATCH, max_num_batched_requests=BATCH,
                  meta_tensors={"qo_indptr_buffer": qo_indptr})
    pk = PersistentKernel(**params)

    logits = build_logits(device, dtype)
    logits_d = pk.attach_input(logits, name="logits")

    greedy_out = add_sampling(pk, logits_d, "greedy", 0.0, 1.0, 0)
    topk_out = add_sampling(pk, logits_d, "topk", 1.0, 1.0, 3)
    topp_out = add_sampling(pk, logits_d, "topp", 1.0, 0.6, 0)
    dist_outs = [add_sampling(pk, logits_d, f"dist{i}", 1.0, 1.0, 3, seed=s)
                 for i, s in enumerate(DIST_SEEDS)]

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ref_greedy, ref_topk, ref_nucleus = reference_sets(logits)
    print(f"reference greedy={ref_greedy} top3={sorted(ref_topk)} "
          f"nucleus={sorted(ref_nucleus)}")
    ok = True

    if len(ref_nucleus) < 2:
        print("FAILED: nucleus is a single token, the top-p case is not "
              "discriminating")
        ok = False

    got = greedy_out.squeeze(1).tolist()
    if any(t != ref_greedy for t in got):
        print(f"[greedy] FAILED: expected {ref_greedy} everywhere, got {got}")
        ok = False
    else:
        print(f"[greedy] ok: all rows returned {ref_greedy}")

    for name, out, allowed in (("top_k=3", topk_out, ref_topk),
                               ("top_p=0.6", topp_out, ref_nucleus)):
        got = out.squeeze(1).tolist()
        outside = [t for t in got if t not in allowed]
        distinct = set(got)
        if outside:
            print(f"[{name}] FAILED: drew {outside} outside {sorted(allowed)}")
            ok = False
        elif len(distinct) < 2:
            print(f"[{name}] FAILED: every row drew {got[0]}; the noise is "
                  f"not varying across rows")
            ok = False
        else:
            print(f"[{name}] ok: {len(distinct)} distinct tokens, all inside "
                  f"{sorted(allowed)}")

    for name, out in (("greedy", greedy_out), ("top_k=3", topk_out),
                      ("top_p=0.6", topp_out)):
        pad = [t for t in out.squeeze(1).tolist() if t >= VOCAB or t < 0]
        if pad:
            print(f"[{name}] FAILED: returned padding-row token ids {pad}")
            ok = False

    # Compare the empirical top-k distribution against the softmax over the
    # kept logits, pooling the rows of every seeded copy of the same draw.
    counts = {}
    for out in dist_outs:
        for tok in out.squeeze(1).tolist():
            counts[tok] = counts.get(tok, 0) + 1
    draws = len(dist_outs) * BATCH
    probs = torch.softmax(logits[0, :VOCAB].float(), dim=-1)
    kept = sorted(ref_topk, key=lambda t: -probs[t])
    kept_mass = float(sum(probs[t] for t in kept))
    print(f"[top_k=3] {draws} draws pooled over {len(DIST_SEEDS)} seeds")
    for tok in kept:
        expected = float(probs[tok]) / kept_mass
        empirical = counts.get(tok, 0) / draws
        sigma = (expected * (1 - expected) / draws) ** 0.5
        deviation = abs(empirical - expected) / sigma
        status = "ok" if deviation < 4 else "FAILED"
        if status != "ok":
            ok = False
        print(f"  token {tok}: expected {expected:.4f} empirical "
              f"{empirical:.4f} ({deviation:.2f} sigma) {status}")
    stray = [t for t in counts if t not in ref_topk]
    if stray:
        print(f"[top_k=3] FAILED: drew {stray} outside the top-3")
        ok = False

    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
