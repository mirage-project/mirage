"""A searched schedule runs as an MPK task, and matches the hand-written one.

test_generated_task_testmode.py covers the same SwiGLU segment with the
schedule written by hand (generated_swiglu_layer). This runs the identical
computation and tolerance with the schedule DISCOVERED by search() instead,
so a pass means the superoptimizer produced something MPK can execute and
that is numerically right -- on a task where the correct answer is already
known.

Search is randomized and slow, so each case runs in its own subprocess with a
generous timeout, the same isolation test_generated_task_testmode.py uses for
its deadlock-prone cases.
"""
import subprocess
import sys
import textwrap

import pytest
import torch


def _skip_reason():
    if not torch.cuda.is_available():
        return "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        return "generated task bodies are only emitted for the sm_100 backend"
    return None


_SEARCHED_SWIGLU_SRC = textwrap.dedent(
    """
    import sys, torch, mirage
    from mirage.mpk.persistent_kernel import PersistentKernel
    from mirage.mpk.lowering.task_search import (
        TaskSpec, TensorSpec, search_task_schedule, register_searched_task)

    M, K, N, FL = {m}, {k}, {n}, {fl}
    torch.manual_seed(0)

    # WHAT the task computes -- no tiling or loop decisions here.
    spec = TaskSpec(
        name="swiglu",
        build=lambda kn, t: kn.mul(kn.silu(kn.matmul(t[0], t[1])),
                                   kn.matmul(t[0], t[2])),
        inputs=[TensorSpec((M, K)), TensorSpec((K, N)), TensorSpec((K, N))],
    )
    sched = search_task_schedule(spec, grid_dim=(1, 1, 1), forloop_range=FL)
    print("SCHEDULE", sched.describe(), flush=True)

    x  = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    wg = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    wu = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    o  = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    nw, ns = mirage.get_configurations_from_gpu(0)
    p = PersistentKernel.get_default_init_parameters()
    p.update(test_mode=True, num_workers=nw, num_local_schedulers=ns,
             mpi_rank=0, world_size=1, max_num_batched_tokens=max(M, 8),
             max_num_batched_requests=max(M, 8))
    pk = PersistentKernel(**p)
    xd = pk.attach_input(x, name="x")
    gd = pk.attach_input(wg, name="wg")
    ud = pk.attach_input(wu, name="wu")
    od = pk.attach_input(o, name="o")

    register_searched_task(pk, sched, inputs=[xd, gd, ud], output=od)
    pk.compile(output_dir=None)
    pk(); torch.cuda.synchronize()

    g = (x.float() @ wg.float()).to(torch.bfloat16).float()
    u = (x.float() @ wu.float()).to(torch.bfloat16).float()
    ref = torch.nn.functional.silu(g) * u
    rel = ((o.float() - ref).abs().max() / ref.abs().max()).item()
    print("REL", rel, flush=True)
    sys.exit(0 if rel < 0.02 else 1)
    """
)


def _run(src, label, timeout=2400):
    proc = subprocess.run([sys.executable, "-c", src], timeout=timeout,
                           capture_output=True, text=True)
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    assert proc.returncode == 0, f"{label} failed (rc={proc.returncode}):\n{tail}"
    return proc.stdout


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
# Same shapes as test_generated_swiglu_segment, so a failure here against a
# pass there isolates the searched schedule as the difference.
@pytest.mark.parametrize("m,k,fl", [(128, 128, 2), (8, 256, 4)])
def test_searched_swiglu_matches_torch(m, k, fl):
    src = _SEARCHED_SWIGLU_SRC.format(m=m, k=k, n=64, fl=fl)
    out = _run(src, f"searched SwiGLU M={m} K={k} FL={fl}")
    assert "SCHEDULE" in out, "no schedule was reported"


if __name__ == "__main__":
    for m, k, fl in [(128, 128, 2), (8, 256, 4)]:
        print(f"===== M={m} K={k} FL={fl} =====", flush=True)
        try:
            print(_run(_SEARCHED_SWIGLU_SRC.format(m=m, k=k, n=64, fl=fl),
                       f"M={m}"), flush=True)
        except AssertionError as e:
            print(f"FAILED: {e}", flush=True)
    print("ALLDONE", flush=True)
