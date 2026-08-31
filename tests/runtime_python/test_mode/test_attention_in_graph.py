"""How much of attention can live in the graph, and what stops the rest.

The other three opaque nodes -- embedding, the KV-cache append, argmax -- have
no muGraph op at all. Attention is different: most of it is expressible, and
this pins exactly where that stops, so the boundary is a measured fact rather
than an assumption.

    exp(Q@K^T + mask) @ V     the softmax NUMERATOR -- searches, and is
                              numerically correct (rel 3.4e-3, see
                              test_searched_batched_matmul.py)

    reduction(E, dim)         the DENOMINATOR -- search now RETURNS candidates,
                              but every one leaves the reduction at kernel
                              level, outside the fused task

So a searched task can compute the numerator but not normalise it, and
attention stays opaque until that changes.

What the denominator used to do instead was crash. A one-input spec fell into
get_customized_input_cand_idx's {num_inputs - 2, num_inputs - 1} candidate --
{-1, 0} -- and the caller then read all_tensors[-1], out of bounds on a
vector. The symptom depended on what sat before the buffer: a blank DTensor
(guid 0) that failed deserialization, or "*** stack smashing detected ***".
KN_REDUCTION_2_OP was blamed for it because a bare reduction was the only
one-input spec anyone had tried; the crash reproduces with that op DISABLED.
test_a_one_input_spec_does_not_corrupt_memory below is the regression.
"""
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
    """Which is what keeps the softmax out of the graph.

    KN_REDUCTION_2_OP is still commented out of knop_to_explore in
    src/search/config.cc, so search cannot build the reduction itself. It
    stays out because uncommenting it does not help: the candidates then leave
    the reduction at KERNEL level, outside the fused task, and task_search
    rejects them -- an MPK task is exactly one customized op.

    What changed: this used to fail by CORRUPTING MEMORY rather than by
    returning nothing. See the module docstring.

    If this ever starts passing, attention can stop being opaque.
    """
    v = _verdict("reduction")
    assert v.startswith("no"), f"reductions became usable: {v}"


def test_a_one_input_spec_does_not_corrupt_memory():
    """The regression for the all_tensors[-1] read.

    Any spec with fewer than two inputs used to hit it; a bare reduction is
    the smallest. What matters is not the verdict -- a reduction is still not
    usable as a task -- but that the search completes and says so, rather than
    aborting the walk on a blank guid-0 tensor or smashing the stack.
    """
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
