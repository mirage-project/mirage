"""How much of attention can live in the graph, and what stops the rest."""
import subprocess
import sys
import textwrap

import pytest
import torch

F, M, D, S = 8, 8, 128, 128


def _skip_reason():
    if not torch.cuda.is_available():
        return "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        return "generated task bodies are only emitted for the sm_100 backend"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None,
                                reason=_skip_reason() or "")

_SRC = textwrap.dedent(
    """
    import sys
    from mirage.mpk.lowering.task_search import (TaskSpec, TensorSpec,
                                        search_task_schedules)
    F, M, D, S = {f}, {m}, {d}, {s}

    def numerator(kn, t):
        return kn.matmul(kn.exp(kn.add(kn.matmul(t[0], t[1]), t[3])), t[2])

    def core(kn, t):
        e = kn.exp(kn.add(kn.matmul(t[0], t[1]), t[3]))
        return kn.div(kn.matmul(e, t[2]), kn.reduction(e, 2))

    CASES = {{
        "numerator": (numerator, [(F,M,D), (F,D,S), (F,S,D), (F,M,S)]),
        "reduction": (lambda kn, t: kn.reduction(t[0], 2), [(F,M,S)]),
        "core":      (core, [(F,M,D), (F,D,S), (F,S,D), (F,M,S)]),
    }}
    which = sys.argv[1]
    build, dims = CASES[which]
    try:
        got = search_task_schedules(
            TaskSpec("p", build, [TensorSpec(x) for x in dims]),
            grid_dim=(F, 1, 1), wide_inputs=len(dims) > 3)
        print("VERDICT ok " + str(len(got)))
    except Exception as e:
        print("VERDICT no " + type(e).__name__ + ": "
              + str(e)[:150].replace(chr(10), " "))
    """
)


def _verdict(which):
    proc = _run(which)
    idx = proc.stdout.rfind("VERDICT ")
    assert idx >= 0, (proc.stdout + proc.stderr)[-800:]
    return proc.stdout[idx + len("VERDICT "):].split("\n", 1)[0].strip()


def _run(which):
    """One search in its own process, so a crash cannot take pytest with it."""
    return subprocess.run(
        [sys.executable, "-c", _SRC.format(f=F, m=M, d=D, s=S), which],
        capture_output=True, text=True, timeout=2400)


def test_the_softmax_numerator_is_searchable():
    """Four inputs, a batched matmul and a chained matmul -- all of which were
    believed impossible earlier in this work. This is the largest piece of
    attention a searched task can compute."""
    v = _verdict("numerator")
    assert v.startswith("ok"), v


def test_a_reduction_is_not_usable_as_a_task():
    """Which is what keeps the softmax out of the graph."""
    v = _verdict("reduction")
    assert v.startswith("no"), f"reductions became usable: {v}"


def test_a_one_input_spec_does_not_corrupt_memory():
    """The regression for the all_tensors[-1] read."""
    proc = _run("reduction")
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"search aborted ({proc.returncode}): {out[-800:]}"
    assert "stack smashing" not in out, out[-800:]
    assert "no kernel-graph tensor with guid 0" not in out, out[-800:]
    assert "VERDICT " in proc.stdout, out[-800:]


def test_the_full_core_leaks_ops_outside_the_task():
    """Given the denominator, search does return candidates for the whole
    softmax core -- but they leave ops at kernel level, which MPK would
    silently drop, so task_search rejects them."""
    v = _verdict("core")
    assert v.startswith("no"), v
    assert "outside the fused task" in v or "TaskSearchError" in v, v
