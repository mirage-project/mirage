"""Extra sampling edge cases: top_k=1 == argmax, and top_k budget reject.

Companion to test_sampling_topk_topp_testmode.py. Uses a smaller padded vocab
so the megakernel compiles quickly enough to keep these as cheap regression
guards.

    python tests/runtime_python/test_mode/test_sampling_edges_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

BATCH = 8
NUM_TASKS = 8
CHUNK = 256
PADDED = NUM_TASKS * CHUNK  # 2048
VOCAB = 1500
TOPK_MAX = 16
SEED = 7


def build_pk(batch=BATCH):
    device = "cuda"
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    # Force the partial grid to NUM_TASKS so the padded vocab divides evenly.
    params = PersistentKernel.get_default_init_parameters()
    qo = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    qo[batch] = batch
    params.update(
        test_mode=True,
        num_workers=NUM_TASKS,
        num_local_schedulers=num_schedulers,
        max_num_batched_tokens=batch,
        max_num_batched_requests=batch,
        meta_tensors={"qo_indptr_buffer": qo},
    )
    return PersistentKernel(**params)


def main():
    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    ok = True

    # --- API guard: top_k above the candidate budget must fail early ---
    try:
        PersistentKernel(
            **{**PersistentKernel.get_default_init_parameters(),
               "test_mode": True,
               "do_sample": True,
               "temperature": 1.0,
               "top_k": 64,
               "sampling_topk_max": 32,
               "meta_tensors": {
                   "qo_indptr_buffer": torch.zeros(2, dtype=torch.int32,
                                                   device=device)}})
        print("[api] FAILED: top_k > sampling_topk_max was accepted")
        ok = False
    except ValueError as e:
        print(f"[api] ok: rejected oversized top_k ({e})")

    pk = build_pk()
    logits = -torch.rand(BATCH, PADDED, dtype=dtype, device=device) - 0.5
    logits[:, VOCAB:] = 0.0
    winners = [3, 200, 500, 999, 1200, 1499, 77, 888]
    for row, idx in enumerate(winners):
        logits[row, idx] = -0.1

    d_logits = pk.attach_input(logits, name="logits")

    # top_k=1 must reproduce argmax over the real vocab
    part_val = torch.zeros(BATCH, NUM_TASKS * (TOPK_MAX + 2),
                           dtype=torch.float32, device=device)
    part_idx = torch.zeros(BATCH, NUM_TASKS * TOPK_MAX,
                           dtype=torch.int64, device=device)
    out_k1 = torch.zeros(BATCH, 1, dtype=torch.int64, device=device)
    d_val = pk.attach_input(part_val, name="k1_val")
    d_idx = pk.attach_input(part_idx, name="k1_idx")
    pk.sampling_partial_layer(
        input=d_logits, output=(d_val, d_idx),
        grid_dim=(NUM_TASKS, 1, 1), block_dim=(128, 1, 1),
        vocab_size=VOCAB, topk_max=TOPK_MAX, temperature=1.0)
    pk.sampling_reduce_layer(
        input=(d_val, d_idx),
        output=pk.attach_input(out_k1, name="k1_out"),
        grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
        temperature=1.0, top_p=1.0, top_k=1, seed=SEED)

    # T=0 greedy path must also match
    part_val_g = torch.zeros_like(part_val)
    part_idx_g = torch.zeros_like(part_idx)
    out_g = torch.zeros_like(out_k1)
    d_valg = pk.attach_input(part_val_g, name="g_val")
    d_idxg = pk.attach_input(part_idx_g, name="g_idx")
    pk.sampling_partial_layer(
        input=d_logits, output=(d_valg, d_idxg),
        grid_dim=(NUM_TASKS, 1, 1), block_dim=(128, 1, 1),
        vocab_size=VOCAB, topk_max=TOPK_MAX, temperature=0.0)
    pk.sampling_reduce_layer(
        input=(d_valg, d_idxg),
        output=pk.attach_input(out_g, name="g_out"),
        grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
        temperature=0.0, top_p=1.0, top_k=0, seed=SEED)

    print("Compiling...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running...")
    pk()
    torch.cuda.synchronize()

    ref = logits[:, :VOCAB].float().argmax(dim=-1).tolist()
    got_k1 = out_k1.squeeze(1).tolist()
    got_g = out_g.squeeze(1).tolist()
    if got_k1 != ref:
        print(f"[top_k=1] FAILED: expected {ref} got {got_k1}")
        ok = False
    else:
        print(f"[top_k=1] ok: matched argmax {ref}")
    if got_g != ref:
        print(f"[T=0] FAILED: expected {ref} got {got_g}")
        ok = False
    else:
        print(f"[T=0] ok: matched argmax {ref}")
    if any(t >= VOCAB for t in got_k1 + got_g):
        print("[pad] FAILED: drew a padding-row id")
        ok = False
    else:
        print("[pad] ok: no padding-row ids")

    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
