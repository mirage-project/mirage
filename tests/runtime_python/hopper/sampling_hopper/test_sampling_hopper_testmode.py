"""sampling_partial_layer + sampling_reduce_layer through MPK on Hopper.

On sm_90 the layers register sampling_partial_hopper / sampling_reduce_hopper.
One test_mode PersistentKernel holds three samplers over an lm_head padded
past the vocab (the padding logits beat every real one, so ignoring
vocab_size would be caught):

  greedy   temperature=0            must return torch's argmax
  top-k    temperature=1, top_k=3   draws inside torch's top-3, not all equal
  top-p    temperature=1, top_p=0.6 draws inside torch's nucleus, not all equal

    python tests/runtime_python/hopper/sampling_hopper/test_sampling_hopper_testmode.py
"""

import os
import sys
import tempfile

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import allowed_sets  # noqa: E402

BATCH = 32
PADDED = 153600
VOCAB = 151936
NUM_TASKS = 128
TOPK_MAX = 32
SPECIAL_IDS = [5, 1600, 40001, 77777, 120005, 151935]
SPECIAL_LOGITS = [-1.0, -1.05, -1.1, -3.0, -4.0, -6.0]


def add_sampler(pk, logits_d, name, temperature, top_p, top_k):
    val = torch.zeros(BATCH, NUM_TASKS * (TOPK_MAX + 2), device="cuda")
    idx = torch.zeros(BATCH, NUM_TASKS * TOPK_MAX, dtype=torch.int64,
                      device="cuda")
    out = torch.full((BATCH, 1), -7, dtype=torch.int64, device="cuda")
    val_d = pk.attach_input(val, name=f"{name}_val")
    idx_d = pk.attach_input(idx, name=f"{name}_idx")
    pk.sampling_partial_layer(
        input=logits_d, output=(val_d, idx_d), grid_dim=(NUM_TASKS, 1, 1),
        block_dim=(256, 1, 1), vocab_size=VOCAB, topk_max=TOPK_MAX,
        temperature=temperature)
    pk.sampling_reduce_layer(
        input=(val_d, idx_d), output=pk.attach_input(out, name=f"{name}_out"),
        grid_dim=(1, 1, 1), block_dim=(256, 1, 1), temperature=temperature,
        top_p=top_p, top_k=top_k, seed=1234)
    return out


def main():
    assert torch.cuda.get_device_capability() == (9, 0), "needs an sm_90 GPU"
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    qo_indptr = torch.zeros(BATCH + 1, dtype=torch.int32, device="cuda")
    qo_indptr[BATCH] = BATCH
    params.update(test_mode=True, num_workers=num_workers,
                  num_local_schedulers=num_schedulers, mpi_rank=0,
                  world_size=1, max_num_batched_tokens=BATCH,
                  max_num_batched_requests=BATCH, max_num_pages=BATCH,
                  meta_tensors={"qo_indptr_buffer": qo_indptr})
    pk = PersistentKernel(**params)

    logits = torch.full((BATCH, PADDED), -30.0, device="cuda",
                        dtype=torch.bfloat16)
    for tok, v in zip(SPECIAL_IDS, SPECIAL_LOGITS):
        logits[:, tok] = v
    logits[:, VOCAB:] = 0.0
    logits_d = pk.attach_input(logits, name="logits")
    outs = {
        "greedy": (add_sampler(pk, logits_d, "greedy", 0.0, 1.0, 0), 0, 1.0),
        "top_k=3": (add_sampler(pk, logits_d, "topk", 1.0, 1.0, 3), 3, 1.0),
        "top_p=0.6": (add_sampler(pk, logits_d, "topp", 1.0, 0.6, 0), 0, 0.6),
    }

    print("Compiling test kernel...")
    # Generated sources go to a temp dir; *.cu here is the tracked wrapper.
    pk.compile(output_dir=tempfile.mkdtemp(prefix="mpk_hopper_testmode_"))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ok = True
    argmax = int(logits[0, :VOCAB].float().argmax())
    for name, (out, top_k, top_p) in outs.items():
        got = out.squeeze(1).tolist()
        if name == "greedy":
            passed = all(t == argmax for t in got)
            detail = f"all rows -> {argmax}" if passed else f"got {got}"
        else:
            allowed = allowed_sets(logits[:1], VOCAB, top_k, top_p)[0]
            outside = [t for t in got if t not in allowed]
            passed = not outside and len(set(got)) > 1
            detail = (f"{len(set(got))} distinct tokens within {sorted(allowed)}"
                      if passed else f"outside={outside[:4]} distinct={set(got)}")
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    pk.finalize()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
