"""Smoke tests for layers.mtp.verify.MTPVerify (3 modes) and MTPAcceptCommit.

Forward() of every variant raises NotImplementedError. Smoke test only:
instantiate → compile → run → no crash, no NaN/Inf in output buffers.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.mtp.verify import (
    MTPVerifyStrict, MTPVerifyProbabilistic, MTPVerifyTargetGreedy,
    MTPAcceptCommit,
)


def _make_pk(batch_size):
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    return PersistentKernel(**params)


def _run_and_check(pk, *buffers):
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()
    for b in buffers:
        if b.dtype.is_floating_point:
            if b.isnan().any() or b.isinf().any():
                print(f"FAILED: buffer contains NaN/Inf")
                pk.finalize()
                sys.exit(1)
    pk.finalize()


def test_verify_strict_smoke():
    device = "cuda"
    torch.manual_seed(0)
    batch_size = 1
    num_draft_tokens = 2

    draft_tok = torch.zeros(
        batch_size, num_draft_tokens, dtype=torch.int64, device=device,
    )
    target_tok = torch.zeros(
        batch_size, num_draft_tokens + 1, dtype=torch.int64, device=device,
    )
    accepted_count = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)
    output_tokens = torch.zeros(
        batch_size, num_draft_tokens + 1, dtype=torch.int64, device=device,
    )

    m = MTPVerifyStrict(num_draft_tokens=num_draft_tokens, prefix="v_")
    pk = _make_pk(batch_size)

    d_dt = pk.attach_input(draft_tok, name="draft_strict")
    t_dt = pk.attach_input(target_tok, name="target_strict")
    a_dt = pk.attach_input(accepted_count, name="accepted_strict")
    o_dt = pk.attach_input(output_tokens, name="output_strict")

    with pk.compile_scope():
        _ = m.compile(d_dt, t_dt, a_dt, o_dt)

    print("Compiling MTPVerify(strict)...")
    _run_and_check(pk, accepted_count, output_tokens)
    print(f"accepted_count: {accepted_count.tolist()}")
    print("PASSED: MTPVerify(strict) smoke")


def test_verify_target_greedy_smoke():
    # Header include for kernel::target_verify_greedy_kernel is now in
    # task_header.cuh (added 2026-05-16). Smoke test exercises full
    # compile + run path.
    device = "cuda"
    torch.manual_seed(0)
    batch_size = 1
    num_draft_tokens = 2

    draft_tok = torch.zeros(
        batch_size, num_draft_tokens, dtype=torch.int64, device=device,
    )
    target_tok = torch.zeros(
        batch_size, num_draft_tokens + 1, dtype=torch.int64, device=device,
    )
    accepted_count = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)
    output_tokens = torch.zeros(
        batch_size, num_draft_tokens + 1, dtype=torch.int64, device=device,
    )

    m = MTPVerifyTargetGreedy(
        num_draft_tokens=num_draft_tokens, prefix="vg_",
    )
    pk = _make_pk(batch_size)
    d_dt = pk.attach_input(draft_tok, name="draft_greedy")
    t_dt = pk.attach_input(target_tok, name="target_greedy")
    a_dt = pk.attach_input(accepted_count, name="accepted_greedy")
    o_dt = pk.attach_input(output_tokens, name="output_greedy")

    with pk.compile_scope():
        _ = m.compile(d_dt, t_dt, a_dt)

    print("Compiling MTPVerify(target_greedy)...")
    _run_and_check(pk, accepted_count, output_tokens)
    print(f"accepted_count: {accepted_count.tolist()}")
    print("PASSED: MTPVerify(target_greedy) smoke")


def test_verify_probabilistic_smoke():
    device = "cuda"
    torch.manual_seed(0)
    batch_size = 1
    num_draft_tokens = 2

    draft_tok = torch.zeros(
        batch_size, num_draft_tokens, dtype=torch.int64, device=device,
    )
    target_tok = torch.zeros(
        batch_size, num_draft_tokens + 1, dtype=torch.int64, device=device,
    )
    accepted_count = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)
    output_tokens = torch.zeros(
        batch_size, num_draft_tokens + 1, dtype=torch.int64, device=device,
    )
    target_probs = torch.full(
        (batch_size, num_draft_tokens + 1), 0.5,
        dtype=torch.float32, device=device,
    )
    draft_probs = torch.full(
        (batch_size, num_draft_tokens), 0.5,
        dtype=torch.float32, device=device,
    )
    seed = torch.zeros(batch_size, dtype=torch.int32, device=device)

    m = MTPVerifyProbabilistic(
        num_draft_tokens=num_draft_tokens, prefix="vp_",
    )
    pk = _make_pk(batch_size)

    d_dt = pk.attach_input(draft_tok, name="draft_prob")
    t_dt = pk.attach_input(target_tok, name="target_prob")
    a_dt = pk.attach_input(accepted_count, name="accepted_prob")
    o_dt = pk.attach_input(output_tokens, name="output_prob")
    tp_dt = pk.attach_input(target_probs, name="t_probs")
    dp_dt = pk.attach_input(draft_probs, name="d_probs")
    sd_dt = pk.attach_input(seed, name="seed_prob")

    with pk.compile_scope():
        # New positional order: draft, target, target_probs, draft_probs, seed, accepted_count, output_tokens
        _ = m.compile(d_dt, t_dt, tp_dt, dp_dt, sd_dt, a_dt, o_dt)

    print("Compiling MTPVerify(probabilistic)...")
    _run_and_check(pk, accepted_count, output_tokens, target_probs, draft_probs)
    print(f"accepted_count: {accepted_count.tolist()}")
    print("PASSED: MTPVerify(probabilistic) smoke")


def test_accept_commit_smoke():
    device = "cuda"
    torch.manual_seed(0)
    batch_size = 1
    num_draft_tokens = 2

    accepted_count = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)
    output_tokens = torch.zeros(
        batch_size, num_draft_tokens + 1, dtype=torch.int64, device=device,
    )
    current_position = torch.zeros(batch_size, dtype=torch.int32, device=device)
    new_position = torch.zeros(batch_size, dtype=torch.int32, device=device)
    final_output = torch.zeros(batch_size, 32, dtype=torch.int64, device=device)
    num_new_tokens = torch.zeros(1, dtype=torch.int32, device=device)

    m = MTPAcceptCommit(num_draft_tokens=num_draft_tokens, prefix="ac_")
    pk = _make_pk(batch_size)

    a_dt = pk.attach_input(accepted_count, name="ac_accepted")
    o_dt = pk.attach_input(output_tokens, name="ac_output")
    cp_dt = pk.attach_input(current_position, name="ac_curpos")
    np_dt = pk.attach_input(new_position, name="ac_newpos")
    fo_dt = pk.attach_input(final_output, name="ac_final")
    nn_dt = pk.attach_input(num_new_tokens, name="ac_nnt")

    with pk.compile_scope():
        _ = m.compile(a_dt, o_dt, cp_dt, np_dt, fo_dt, nn_dt)

    print("Compiling MTPAcceptCommit (smoke)...")
    _run_and_check(pk, final_output)
    print("PASSED: MTPAcceptCommit smoke")


if __name__ == "__main__":
    test_verify_strict_smoke()
    test_verify_target_greedy_smoke()
    test_verify_probabilistic_smoke()
    test_accept_commit_smoke()
