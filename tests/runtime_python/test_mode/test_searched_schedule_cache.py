"""The searched-schedule cache: correct where it hits, inert where it misses.

generated_linear_layer consults task_search.lookup_schedule and, on a hit,
registers a schedule that search found and a whole-model measurement chose
instead of the one written in the function. That is a silent behaviour change
for anyone calling it at a cached shape, so it needs to be pinned:

  - a cache hit still computes the right thing
  - a cache miss changes nothing (the hand-written path runs)
  - the cached entry is the schedule it claims to be
  - MPK_SEARCHED_SCHEDULES=0 disables it

The whole-model throughput claim behind the entry is NOT retested here -- it
takes ~40 minutes of model builds. See experiments/searched_tasks/
rank_by_model.py and the provenance recorded alongside the entry.
"""
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from mirage.mpk.lowering import task_search

# The shape the cached linear entry was measured at (Qwen3-0.6B gate/up
# projection, batch 8).
M, K, N = 8, 1024, 3072
GRID = (N // 64, 1, 1)


def _skip_reason():
    if not torch.cuda.is_available():
        return "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        return "generated task bodies are only emitted for the sm_100 backend"
    return None


def test_cache_hit_is_the_recorded_schedule():
    sched = task_search.lookup_schedule("linear", [(M, K), (K, N)], GRID)
    assert sched is not None, "the measured linear schedule is missing"
    # forloop_range=4 is the whole point: the hand-written layer uses
    # hidden_size // 64 == 16, and 4 measured 1.072x faster end to end.
    assert sched.forloop_range == 4, sched.describe()
    assert sched.block_dim == task_search.MPK_BLOCK_DIM
    assert [o["op_type"] for o in sched.ops] == [
        "tb_input_op", "tb_input_op", "tb_matmul_op",
        "tb_forloop_accum_no_red_op", "tb_output_op",
    ], sched.describe()


def test_cache_misses_on_a_different_shape():
    # A schedule is only known good for what it was measured on, so anything
    # else must fall through to the hand-written path untouched.
    assert task_search.lookup_schedule("linear", [(M, 2048), (2048, N)], GRID) is None
    assert task_search.lookup_schedule("linear", [(M, K), (K, N)], (1, 1, 1)) is None
    assert task_search.lookup_schedule("nonexistent", [(M, K), (K, N)], GRID) is None


def test_env_var_disables_the_cache(monkeypatch):
    monkeypatch.setenv("MPK_SEARCHED_SCHEDULES", "0")
    assert task_search.lookup_schedule("linear", [(M, K), (K, N)], GRID) is None


_LINEAR_SRC = textwrap.dedent(
    """
    import sys, torch, mirage
    from mirage.mpk.persistent_kernel import PersistentKernel
    M, K, N = {m}, {k}, {n}
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nw, ns = mirage.get_configurations_from_gpu(0)
    p = PersistentKernel.get_default_init_parameters()
    p.update(test_mode=True, num_workers=nw, num_local_schedulers=ns,
             mpi_rank=0, world_size=1, max_num_batched_tokens=max(M, 8),
             max_num_batched_requests=max(M, 8))
    pk = PersistentKernel(**p)
    xd = pk.attach_input(x, name="x")
    wd = pk.attach_input(w, name="w")
    od = pk.attach_input(o, name="o")
    pk.generated_linear_layer(input=xd, weight_t=wd, output=od,
                              grid_dim=(N // 64, 1, 1), block_dim=(256, 1, 1),
                              forloop_range=K // 64)
    pk.compile(output_dir=None)
    pk(); torch.cuda.synchronize()
    ref = x.float() @ w.float()
    rel = ((o.float() - ref).abs().max() / ref.abs().max()).item()
    print("REL", rel)
    sys.exit(0 if rel < 0.02 else 1)
    """
)


def _run(env_extra, label):
    env = dict(os.environ)
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, "-c", _LINEAR_SRC.format(m=M, k=K, n=N)],
        capture_output=True, text=True, timeout=1800, env=env)
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-20:])
    assert proc.returncode == 0, f"{label} failed:\n{tail}"


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_linear_correct_with_cached_schedule():
    """The cache hit path computes x @ w correctly."""
    _run({"MPK_SEARCHED_SCHEDULES": "1"}, "cached (searched fl=4)")


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_linear_correct_without_cache():
    """And so does the hand-written path it replaces, at the same shape --
    so a failure above is the searched schedule, not the shape."""
    _run({"MPK_SEARCHED_SCHEDULES": "0"}, "hand-written (fl=16)")
