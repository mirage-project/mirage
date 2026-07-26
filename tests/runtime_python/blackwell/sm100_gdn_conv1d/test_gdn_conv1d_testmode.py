"""Test mode: `gdn_conv1d_layer` through the full MPK pipeline.

Exercises the parts the kernel-wrapper test cannot: the Python layer API, task
registration, C++ code generation (including the emitted `qo_indptr` slicing,
the per-slot conv-state offset and the `step == 0` predicate), nvcc compilation
and runtime dispatch.

Two requests prefill in the same iteration, so one compile covers:

  * per-slot token windows taken from the runtime's own `qo_indptr_buffer`
    (the two prompts have DIFFERENT lengths, so a hardcoded stride would fail)
  * per-slot conv-state pool indexing
  * both sides of the `step == 0` predicate: request 0 is seeded with step 0
    (fresh -> stored state ignored) and request 1 with a non-zero step
    (-> stored state consumed). Seeding `step` is the only way to reach the
    carried-state branch in test mode, which runs the task graph exactly once.

Run:
    python tests/runtime_python/blackwell/sm100_gdn_conv1d/\
test_gdn_conv1d_testmode.py
"""

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

CONV_DIM = 4096
KERNEL_SIZE = 4
NUM_CHANNEL_BLOCKS = 4
PROMPT_LENS = [5, 3]              # request 0, request 1
SEEDED_STEPS = [0, 7]             # step==0 -> zero state; step!=0 -> carried
MAX_SEQ_LENGTH = 64
BF16 = torch.bfloat16


def ref_conv(x, w, state, zero_state):
    """Inline reference: fp32 FIR, round the accumulator to bf16, then SiLU.

    That rounding order is HF's (`torch_causal_conv1d_update` runs F.conv1d in
    the weight dtype) and HF is the pinned numeric target
    (v1-architecture.md 1); `test_gdn_conv1d_oracle.py` shows it reproduces the
    real checkpoint's `gdn.conv_out` bit-exactly.

    x [L, D] bf16, w [D, K] bf16, state [K-1, D] bf16.
    """
    L, D = x.shape
    K = w.shape[1]
    s = (torch.zeros(K - 1, D, dtype=torch.float32, device=x.device)
         if zero_state else state.float().clone())
    seq = torch.cat([s, x.float()], dim=0)
    wf = w.float()
    out = torch.zeros(L, D, dtype=torch.float32, device=x.device)
    for t in range(L):
        acc = torch.zeros(D, dtype=torch.float32, device=x.device)
        for j in range(K):
            acc = acc + wf[:, j] * seq[t + j]
        acc = acc.to(BF16).float()
        out[t] = acc * torch.sigmoid(acc)
    return out.to(BF16), seq[-(K - 1):].to(BF16)


def main():
    device = "cuda"
    torch.manual_seed(20260726)

    num_requests = len(PROMPT_LENS)
    total_tokens = sum(PROMPT_LENS)

    x = torch.randn(total_tokens, CONV_DIM, dtype=BF16, device=device)
    w = (torch.randn(CONV_DIM, KERNEL_SIZE, dtype=torch.float32, device=device)
         * 0.05).to(BF16)
    # Non-zero garbage in every slot: slot 0 must ignore it (step == 0), slot 1
    # must consume it (step != 0).
    conv_state = (torch.randn(num_requests, KERNEL_SIZE - 1, CONV_DIM,
                              dtype=torch.float32, device=device)).to(BF16)
    conv_state_in = conv_state.clone()
    out = torch.zeros(total_tokens, CONV_DIM, dtype=BF16, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=total_tokens,
        max_num_batched_requests=num_requests,
        max_num_pages=num_requests * 4,
        page_size=MAX_SEQ_LENGTH,
        max_seq_length=MAX_SEQ_LENGTH,
        meta_tensors={
            "tokens": torch.zeros(
                (num_requests, MAX_SEQ_LENGTH), dtype=torch.int64, device=device
            ),
            "prompt_lengths": torch.tensor(
                PROMPT_LENS, dtype=torch.int32, device=device
            ),
            "step": torch.tensor(SEEDED_STEPS, dtype=torch.int32, device=device),
        },
    )
    pk = PersistentKernel(**params)
    assert pk.target_cc >= 100, "gdn_conv1d_sm100 requires Blackwell"
    block_dim = (256, 1, 1)

    x_dt = pk.attach_input(x, name="conv_in")
    w_dt = pk.attach_input(w, name="conv_weight")
    state_dt = pk.attach_input(conv_state, name="conv_state")
    out_dt = pk.attach_input(out, name="conv_out")

    # grid.y = 4 channel blocks: exercises the prefill scaling axis through the
    # real codegen path (kv_idx -> channel offset), not just grid.y == 1.
    pk.gdn_conv1d_layer(
        input=x_dt,
        weight=w_dt,
        conv_state=state_dt,
        output=out_dt,
        grid_dim=(num_requests, NUM_CHANNEL_BLOCKS, 1),
        block_dim=block_dim,
    )

    pk.compile(output_dir="./test_output_gdn_conv1d")

    # `init_kernel` runs inside compile() and ZEROES `step`
    # (persistent_kernel.cuh:150-153), so the seeded value has to be written
    # back afterwards. New-prefill admission never reads `step`, so this only
    # changes which branch of `gdn_slot_is_first_chunk` each slot takes.
    pk.meta_tensors["step"].copy_(
        torch.tensor(SEEDED_STEPS, dtype=torch.int32, device=device)
    )
    pk()
    torch.cuda.synchronize()

    # `qo_indptr_buffer` is re-filled by prepare_next_batch's SECOND call (the
    # test-mode finalize), so it reads back empty; `step` is the durable
    # evidence that both prefills were scheduled and how many tokens each got.
    print(f"  runtime qo_indptr_buffer (post-finalize) = "
          f"{pk.meta_tensors['qo_indptr_buffer'].tolist()}")
    steps = pk.meta_tensors["step"].tolist()
    want_steps = [s + n for s, n in zip(SEEDED_STEPS, PROMPT_LENS)]
    print(f"  runtime step after finalize = {steps} (expected {want_steps})")
    assert steps == want_steps, (
        "prepare_next_batch did not schedule both prefills in one iteration"
    )

    worst = 0.0
    off = 0
    for slot, n in enumerate(PROMPT_LENS):
        zero_state = SEEDED_STEPS[slot] == 0
        ref, ref_state = ref_conv(
            x[off:off + n], w, conv_state_in[slot], zero_state
        )
        got = out[off:off + n]
        err = (got.float() - ref.float()).abs().max().item()
        worst = max(worst, err)
        print(f"  slot {slot}: Q_LEN={n} step={SEEDED_STEPS[slot]} "
              f"zero_state={zero_state} max_abs_diff={err:.3e}")
        torch.testing.assert_close(got, ref, atol=1e-2, rtol=1e-2)
        assert torch.equal(conv_state[slot], ref_state), (
            f"slot {slot}: conv state written back incorrectly"
        )
        off += n

    # The predicate must actually be live: recompute slot 0 WITH its stale state
    # and confirm the kernel did not produce that.
    stale, _ = ref_conv(x[: PROMPT_LENS[0]], w, conv_state_in[0], False)
    assert not torch.allclose(out[: PROMPT_LENS[0]], stale, atol=1e-3), (
        "slot 0 consumed its stale state - the step==0 predicate is dead"
    )

    pk.finalize()
    print(f"GDN_CONV1D TEST-MODE PIPELINE PASSED (worst max_abs_diff "
          f"{worst:.3e}, budget 1e-2)")


if __name__ == "__main__":
    main()
