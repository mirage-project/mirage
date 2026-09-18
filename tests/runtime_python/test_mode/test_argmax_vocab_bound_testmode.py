"""Argmax over an lm_head padded past the real vocab.

Qwen3 demos pad the lm_head from 151936 to 153600 rows so the partial-task
grid divides evenly. The pad rows produce logits the model never trained
(0 for zero-padded weights), so argmax_partial must ignore positions at or
after ``vocab_size`` — otherwise a pad row wins whenever every real logit is
negative (#751/#752), and any non-zero weight padding is worse (#755).

Every real logit here is strictly negative while the pad region holds 0.0,
so an unbounded argmax would return a pad index (>= VOCAB) for every row.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

BATCH = 4
PADDED = 153600
VOCAB = 151936
NUM_TASKS = 96  # PADDED divides evenly; chunk = 1600
CHUNK = PADDED // NUM_TASKS
# One winner per row: an in-chunk position, the chunk-94 boundary cases
# (150400 starts the chunk that straddles VOCAB; 151935 is the last real
# position), and a mid-vocab position.
WINNERS = [5, 150400, 151935, 77777]


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

    logits = -torch.rand(BATCH, PADDED, dtype=dtype, device=device) - 0.5
    logits[:, VOCAB:] = 0.0  # zero-padded lm_head rows
    for row, idx in enumerate(WINNERS):
        logits[row, idx] = -0.25  # unique real max, still below the padding

    part_val = torch.zeros(BATCH, NUM_TASKS, dtype=dtype, device=device)
    part_idx = torch.zeros(BATCH, NUM_TASKS, dtype=torch.int64, device=device)
    out = torch.zeros(BATCH, 1, dtype=torch.int64, device=device)

    d_val = pk.attach_input(part_val, name="part_val")
    d_idx = pk.attach_input(part_idx, name="part_idx")
    pk.argmax_partial_layer(
        input=pk.attach_input(logits, name="logits"),
        output=(d_val, d_idx),
        grid_dim=(NUM_TASKS, 1, 1), block_dim=(128, 1, 1),
        vocab_size=VOCAB)
    pk.argmax_reduce_layer(
        input=(d_val, d_idx),
        output=pk.attach_input(out, name="argmax_out"),
        grid_dim=(1, 1, 1), block_dim=(128, 1, 1))

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ref = logits[:, :VOCAB].float().argmax(dim=-1).cpu()
    got = out.squeeze(1).cpu()
    ok = True
    for row in range(BATCH):
        pad_beats_real = logits[row, VOCAB:].max() > logits[row, :VOCAB].max()
        status = "ok" if got[row] == ref[row] else "FAILED"
        print(f"[row {row}] expected {ref[row].item()} got {got[row].item()} "
              f"(pad region max beats real max: {pad_beats_real}) {status}")
        if not pad_beats_real:
            print(f"[row {row}] FAILED: case is not adversarial")
            ok = False
        if got[row].item() >= VOCAB:
            print(f"[row {row}] FAILED: returned a padding-row token id")
            ok = False
        if got[row] != ref[row] or ref[row].item() != WINNERS[row]:
            ok = False

    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
