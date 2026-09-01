"""Search can schedule a BATCHED (3D) matmul as an MPK task."""
import subprocess
import sys
import textwrap

import pytest
import torch

from mirage.mpk.lowering.task_search import MMA_K_ATOM


def _skip_reason():
    if not torch.cuda.is_available():
        return "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        return "generated task bodies are only emitted for the sm_100 backend"
    return None


def _run(src, label, timeout=2400):
    proc = subprocess.run([sys.executable, "-c", src], timeout=timeout,
                           capture_output=True, text=True)
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    assert proc.returncode == 0, f"{label} failed (rc={proc.returncode}):\n{tail}"
    return proc.stdout


# ---------------------------------------------------------------- validators

def _matmul_op(m, k, n):
    """A tb_matmul_op as get_graph_structure reports it, batch dim included."""
    return {
        "op_type": "tb_matmul_op",
        "input_tensors": [{"dim": [1, m, k], "guid": 1},
                          {"dim": [1, k, n], "guid": 2}],
        "output_tensors": [{"dim": [1, m, n], "guid": 3}],
    }


def test_wide_n_on_a_1d_grid_is_rejected():
    """The whole of N in one threadblock: swapAB makes mma_m = N = 256."""
    from mirage.mpk.lowering.task_search import _check_matmul_tiles, TaskSearchError
    with pytest.raises(TaskSearchError, match="mma_m=256"):
        _check_matmul_tiles([_matmul_op(8, 128, 256)])


def test_n_split_across_a_second_grid_dim_is_accepted():
    """The same matmul with N split 4 ways: mma_m = 64. This is the fix."""
    from mirage.mpk.lowering.task_search import _check_matmul_tiles
    _check_matmul_tiles([_matmul_op(8, 128, 64)])


@pytest.mark.parametrize("k,ok", [(16, True), (32, True), (128, True),
                                  (8, False), (2, False)])
def test_k_tile_must_be_whole_mma_atoms(k, ok):
    from mirage.mpk.lowering.task_search import _check_matmul_tiles, TaskSearchError
    ops = [_matmul_op(8, k, 64)]
    if ok:
        _check_matmul_tiles(ops)
    else:
        with pytest.raises(TaskSearchError, match="K tile"):
            _check_matmul_tiles(ops)


def test_mma_k_atom_matches_the_transpiler():
    assert MMA_K_ATOM == 32 // 2


# ------------------------------------------------------------------ on-GPU


_ENUM_SRC = textwrap.dedent(
    """
    import json, sys
    from mirage.mpk.lowering.task_search import (TaskSpec, TensorSpec,
                                        search_task_schedules, TaskSearchError)
    {spec}
    try:
        scheds = search_task_schedules(spec, grid_dim={grid})
    except TaskSearchError as e:
        # An empty draw is not a failure -- see _ENUM_ATTEMPTS.
        print("NSCHED 0", e, flush=True)
        scheds = []
    json.dump([s.to_dict() for s in scheds], open({out!r}, "w"))
    print("NSCHED", len(scheds), flush=True)
    for i, s in enumerate(scheds):
        print("CAND", i, s.describe(), flush=True)
    sys.exit(0)
    """
)

_ENUM_ATTEMPTS = 3


def _walk_candidates(spec_src, grid, required, run_src, label):
    """Enumerate, then try each candidate in its own process."""
    import json
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        cand_path = f.name
    for attempt in range(_ENUM_ATTEMPTS):
        out = _run(_ENUM_SRC.format(spec=spec_src, grid=grid, out=cand_path),
                   f"{label}: enumerate")
        cands = json.load(open(cand_path))
        if cands:
            break
    n = len(cands)
    assert n, (f"{label}: search found no usable schedule in "
               f"{_ENUM_ATTEMPTS} draws")
    assert "CAND 0" in out
    for i, c in enumerate(cands):
        kinds = [o["op_type"] for o in c["ops"]]
        for req in required:
            assert req in kinds, f"{label}: candidate {i} lacks {req}: {kinds}"

    failures = []
    for i in range(n):
        proc = subprocess.run(
            [sys.executable, "-c", run_src.format(path=cand_path, index=i)],
            capture_output=True, text=True, timeout=2400)
        if proc.returncode == 0:
            m = [l for l in proc.stdout.splitlines() if l.startswith("REL ")]
            assert m, f"{label}: candidate {i} passed without reporting REL"
            return n, i, float(m[0].split()[1])
        failures.append(
            f"  candidate {i}: "
            + "\n".join((proc.stdout + proc.stderr).splitlines()[-3:]))
    raise AssertionError(
        f"{label}: all {n} candidates failed:\n" + "\n".join(failures))


_BMM_RUN = textwrap.dedent(
    """
    import json, sys, torch, mirage
    from mirage.mpk.persistent_kernel import PersistentKernel
    from mirage.mpk.lowering.task_search import Schedule, register_searched_task

    B, M, K, N = 8, 8, 128, 256
    sched = Schedule.from_dict(json.load(open({path!r}))[{index}])
    torch.manual_seed(0)
    a = torch.randn(B, M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(B, K, N, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros(B, M, N, dtype=torch.bfloat16, device="cuda")

    nw, ns = mirage.get_configurations_from_gpu(0)
    p = PersistentKernel.get_default_init_parameters()
    p.update(test_mode=True, num_workers=nw, num_local_schedulers=ns,
             mpi_rank=0, world_size=1, max_num_batched_tokens=8,
             max_num_batched_requests=8)
    pk = PersistentKernel(**p)
    ad = pk.attach_input(a, name="a")
    bd = pk.attach_input(b, name="b")
    od = pk.attach_input(o, name="o")
    register_searched_task(pk, sched, inputs=[ad, bd], output=od)
    pk.compile(output_dir=None)
    pk(); torch.cuda.synchronize()

    ref = torch.bmm(a.float(), b.float())
    rel = ((o.float() - ref).abs().max() / ref.abs().max()).item()
    print("REL", rel, flush=True)
    sys.exit(0 if rel < 0.02 else 1)
    """
)


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_searched_batched_matmul_matches_torch():
    """A discovered batched-matmul schedule computes torch.bmm."""
    spec = ('spec = TaskSpec("bmm", lambda kn, t: kn.matmul(t[0], t[1]),\n'
            '                [TensorSpec((8, 8, 128)), TensorSpec((8, 128, 256))])')
    n, i, rel = _walk_candidates(spec, (8, 4, 1),
                                 ["tb_matmul_op"], _BMM_RUN, "bmm")
    print(f"bmm: {n} candidate(s), #{i} ran, rel={rel:.5f}")


_SCORES_RUN = textwrap.dedent(
    """
    import json, sys, torch, mirage
    from mirage.mpk.persistent_kernel import PersistentKernel
    from mirage.mpk.lowering.task_search import Schedule, register_searched_task

    FOLD, H, D, S = 8, 8, 128, 128
    sched = Schedule.from_dict(json.load(open({path!r}))[{index}])
    torch.manual_seed(0)
    # Scaled so exp() stays well inside bf16 range.
    q = torch.randn(FOLD, H, D, dtype=torch.bfloat16, device="cuda") * 0.05
    k = torch.randn(FOLD, D, S, dtype=torch.bfloat16, device="cuda") * 0.05
    m = torch.randn(FOLD, H, S, dtype=torch.bfloat16, device="cuda") * 0.05
    o = torch.zeros(FOLD, H, S, dtype=torch.bfloat16, device="cuda")

    nw, ns = mirage.get_configurations_from_gpu(0)
    p = PersistentKernel.get_default_init_parameters()
    p.update(test_mode=True, num_workers=nw, num_local_schedulers=ns,
             mpi_rank=0, world_size=1, max_num_batched_tokens=8,
             max_num_batched_requests=8)
    pk = PersistentKernel(**p)
    qd = pk.attach_input(q, name="q")
    kd = pk.attach_input(k, name="k")
    md = pk.attach_input(m, name="m")
    od = pk.attach_input(o, name="o")
    register_searched_task(pk, sched, inputs=[qd, kd, md], output=od)
    pk.compile(output_dir=None)
    pk(); torch.cuda.synchronize()

    ref = torch.exp(torch.bmm(q.float(), k.float()) + m.float())
    rel = ((o.float() - ref).abs().max() / ref.abs().max()).item()
    print("REL", rel, flush=True)
    sys.exit(0 if rel < 0.02 else 1)
    """
)


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_searched_attention_scores_matches_torch():
    """exp(Q@K^T + mask) -- a batched matmul with a fused epilogue."""
    spec = ('spec = TaskSpec("scores",\n'
            '                lambda kn, t: kn.exp(kn.add(kn.matmul(t[0], t[1]), t[2])),\n'
            '                [TensorSpec((8, 8, 128)), TensorSpec((8, 128, 128)),\n'
            '                 TensorSpec((8, 8, 128))])')
    n, i, rel = _walk_candidates(spec, (8, 1, 1),
                                 ["tb_matmul_op", "tb_exp_op"],
                                 _SCORES_RUN, "attention scores")
    print(f"scores: {n} candidate(s), #{i} ran, rel={rel:.5f}")



# ------------------------------------------- all of the task must be fused

def test_a_leaked_kernel_op_is_rejected():
    """search() may leave part of the spec as a plain kernel-level op beside
    the customized one. The kernel graph is still equivalent to the spec, but
    MPK registers only the customized op as the task body, so the leftover
    would be silently dropped -- exactly the failure that has no symptom.
    """
    from mirage.mpk.lowering.task_search import _KN_STRUCTURAL_OPS, TaskSearchError

    class _FakeCy:
        def __init__(self, structure):
            self._s = structure

        def get_graph_structure(self):
            return self._s

    class _FakeCand:
        def __init__(self, structure):
            self.cygraph = _FakeCy(structure)

    from mirage.mpk.lowering import task_search as ts
    spec = ts.TaskSpec("x", lambda kn, t: t[0], [ts.TensorSpec((8, 64))])
    leaky = [{"op_type": "kn_input_op"},
             {"op_type": "kn_customized_op", "bgraph": {}},
             {"op_type": "kn_exp_op"},
             {"op_type": "kn_output_op"}]
    with pytest.raises(TaskSearchError, match="outside the fused task"):
        ts._as_schedule(spec, _FakeCand(leaky))
    assert "kn_exp_op" not in _KN_STRUCTURAL_OPS


def test_matmul_operand_from_an_accumulator_is_rejected():
    """Search proposed accumulating all three inputs BEFORE the matmul on the
    attention-scores spec. At forloop_range=1 that is an identity, and the
    accumulator's layout is not a legal MMA operand (transpiler error 2).
    Measured: that candidate failed registration while its three siblings,
    which accumulate after the matmul, all matched torch to rel 3.7e-3."""
    from mirage.mpk.lowering.task_search import _check_matmul_operands, TaskSearchError
    accum_then_matmul = [
        {"op_type": "tb_forloop_accum_no_red_op", "input_tensors": [{"guid": 1}],
         "output_tensors": [{"guid": 10}]},
        {"op_type": "tb_matmul_op",
         "input_tensors": [{"guid": 10}, {"guid": 2}],
         "output_tensors": [{"guid": 11}]},
    ]
    with pytest.raises(TaskSearchError, match="legal MMA operand"):
        _check_matmul_operands(accum_then_matmul)

    matmul_then_accum = [
        {"op_type": "tb_matmul_op",
         "input_tensors": [{"guid": 1}, {"guid": 2}],
         "output_tensors": [{"guid": 10}]},
        {"op_type": "tb_forloop_accum_no_red_op",
         "input_tensors": [{"guid": 10}], "output_tensors": [{"guid": 11}]},
    ]
    _check_matmul_operands(matmul_then_accum)


# --------------------------------------- batched matmuls cannot K-split yet

def _inp(guid, forloop_dim):
    return {"op_type": "tb_input_op", "forloop_dim": forloop_dim,
            "output_tensors": [{"guid": guid}]}


def _mm(a_dim, b_dim):
    return {"op_type": "tb_matmul_op",
            "input_tensors": [{"guid": 1, "dim": a_dim},
                              {"guid": 2, "dim": b_dim}]}


@pytest.mark.parametrize("ops,ok,why", [
    ([_inp(1, 2), _inp(2, -1), _mm([1, 8, 32], [1, 32, 64])], False,
     "batched, A is a gmem load split on its K dim"),
    ([_inp(1, -1), _inp(2, 1), _mm([1, 8, 32], [1, 32, 64])], True,
     "batched, only B split on K -- A is not a gmem-loaded K-split"),
    ([_inp(1, -1), _inp(2, -1), _mm([1, 8, 128], [1, 128, 64])], True,
     "batched at forloop_range=1 -- the working case"),
    ([_inp(1, -1), _inp(2, 2), _mm([1, 8, 128], [1, 128, 64])], True,
     "batched, B split on N -- generated_attention_layer's K^T"),
    ([_inp(3, 1), _mm([1, 8, 64], [1, 64, 128])], True,
     "attention's second matmul: A is exp(...), not an input op at all"),
    ([_inp(1, 1), _inp(2, 0), _mm([8, 256], [256, 64])], True,
     "2D K-split -- the cached linear schedule at forloop_range=4"),
])
def test_batched_matmul_k_split_is_rejected(ops, ok, why):
    """Splitting K on a batched operand that is TMA-loaded breaks the
    descriptor. Measured: grid=(8,4,1) fl=4 fails a device static_assert in
    cute::tma_partition, and grid=(8,1,1) fl=2 trips a HOST assert
    ("Majorness of smem doesn't match majorness of gmem") that aborts the
    process -- neither is a returnable error code, which is why search must
    not propose it.
    """
    from mirage.mpk.lowering.task_search import (_check_batched_matmul_forloop,
                                        TaskSearchError)
    if ok:
        _check_batched_matmul_forloop(ops)
    else:
        with pytest.raises(TaskSearchError, match="forloop-split on its K"):
            _check_batched_matmul_forloop(ops)


# ------------------------- search's verifier false-accepts; catch it here

def test_a_candidate_missing_part_of_the_spec_is_rejected():
    """search() verifies equivalence with probabilistic fingerprints, and that
    check false-accepts.
    """
    from mirage.mpk.lowering.task_search import (_check_computes_the_spec,
                                        TaskSearchError)
    body = [{"op_type": "tb_input_op"}, {"op_type": "tb_matmul_op"},
            {"op_type": "tb_add_op"},
            {"op_type": "tb_forloop_accum_no_red_op"}]
    with pytest.raises(TaskSearchError, match="does not compute the spec"):
        _check_computes_the_spec(body, {"tb_matmul_op", "tb_exp_op"})
    # The same body WITH the exp is fine.
    _check_computes_the_spec(body + [{"op_type": "tb_exp_op"}],
                             {"tb_matmul_op", "tb_exp_op"})


def test_only_irreducible_ops_are_required():
    """Binary arithmetic is deliberately not required: search reassociates and
    refactors it legitimately, so demanding it back would reject correct
    rewrites. An exp or a matmul cannot be synthesized from the rest, which is
    what makes those safe to insist on."""
    from mirage.mpk.lowering.task_search import _IRREDUCIBLE_OPS
    for kn_op in ("kn_add_op", "kn_mul_op", "kn_div_op", "kn_sub_op"):
        assert kn_op not in _IRREDUCIBLE_OPS
    for kn_op in ("kn_matmul_op", "kn_exp_op", "kn_silu_op"):
        assert kn_op in _IRREDUCIBLE_OPS


# ------------------------------- specs with more than three inputs

def test_more_than_three_inputs_fails_loudly():
    """It used to fail SILENTLY: search returned zero candidates with no
    diagnostic, which is what made the attention core (Q, K^T, V, mask) look
    like a shape problem when it is an input-COUNT problem."""
    from mirage.mpk.lowering import task_search as ts
    spec = ts.TaskSpec("wide", lambda kn, t: kn.add(kn.add(kn.add(
        t[0], t[1]), t[2]), t[3]), [ts.TensorSpec((8, 64))] * 4)
    with pytest.raises(ts.TaskSearchError, match="wide_inputs=True"):
        ts.search_task_schedules(spec)


def test_three_input_specs_do_not_opt_in():
    """<=3 inputs must pass -1 for both caps, i.e. search exactly as before.
    The wide path explodes (see _WIDE_INPUTS_NOTE), so nothing may reach it by
    accident."""
    import inspect
    from mirage.mpk.lowering import task_search as ts
    src = inspect.getsource(ts.search_task_schedules)
    assert "max_tb_graph_inputs=(len(spec.inputs) if wide_inputs else -1)" in src
    assert "max_kn_graph_ops=(len(spec.inputs) + 4 if wide_inputs else -1)" in src


def test_wide_inputs_note_records_the_real_limits():
    """The note carries the two findings that are easy to get wrong: which
    cap actually governs cost (the threadblock op limit, not the input count)
    and what still limits the attention core (forloop_range, a backend
    defect). It also records the hardcoded input-combination stub that had to
    be fixed, so nobody re-derives it."""
    from mirage.mpk.lowering.task_search import _WIDE_INPUTS_NOTE as note
    assert "get_customized_input_cand_idx" in note
    assert "max_tb_graph_ops" in note
    assert "forloop_range" in note


# ------------------------------------ the attention core, and its one limit

def _mm(guid, ins):
    return {"op_type": "tb_matmul_op",
            "input_tensors": [{"guid": i} for i in ins],
            "output_tensors": [{"guid": guid}]}


def _un(op_type, guid, src):
    return {"op_type": op_type, "input_tensors": [{"guid": src}],
            "output_tensors": [{"guid": guid}]}


_ATTN_BODY = [
    _mm(10, [1, 2]),                      # Q @ K^T
    _un("tb_add_op", 11, 10),             # + mask
    _un("tb_exp_op", 12, 11),             # exp
    _mm(13, [12, 3]),                     # @ V   <- chained
    _un("tb_forloop_accum_no_red_op", 14, 13),
]


_ATTN_SRC = textwrap.dedent(
    """
    import sys, torch, mirage
    from mirage.mpk.persistent_kernel import PersistentKernel
    from mirage.mpk.lowering.task_search import (
        TaskSpec, TensorSpec, search_task_schedules, register_searched_task)

    FOLD, M, D, S = 8, 8, 128, {s}

    spec = TaskSpec("attn",
                    lambda kn, t: kn.matmul(
                        kn.exp(kn.add(kn.matmul(t[0], t[1]), t[3])), t[2]),
                    [TensorSpec((FOLD, M, D)), TensorSpec((FOLD, D, S)),
                     TensorSpec((FOLD, S, D)), TensorSpec((FOLD, M, S))])
    scheds = search_task_schedules(spec, grid_dim=(FOLD, 1, 1),
                                   wide_inputs=True, forloop_range={fl})
    print("NSCHED", len(scheds), flush=True)
    assert scheds
    tiled = [x for x in scheds if x.forloop_range == {fl}]
    assert tiled, [x.describe() for x in scheds]
    sched = tiled[0]
    kinds = [o["op_type"] for o in sched.ops]
    assert kinds.count("tb_matmul_op") == 2, kinds
    assert "tb_exp_op" in kinds and "tb_add_op" in kinds, kinds
    print("SCHEDULE", sched.describe(), flush=True)

    torch.manual_seed(0)
    q = torch.randn(FOLD, M, D, dtype=torch.bfloat16, device="cuda") * 0.05
    k = torch.randn(FOLD, D, S, dtype=torch.bfloat16, device="cuda") * 0.05
    v = torch.randn(FOLD, S, D, dtype=torch.bfloat16, device="cuda") * 0.05
    m = torch.randn(FOLD, M, S, dtype=torch.bfloat16, device="cuda") * 0.05
    o = torch.zeros(FOLD, M, D, dtype=torch.bfloat16, device="cuda")

    nw, ns = mirage.get_configurations_from_gpu(0)
    p = PersistentKernel.get_default_init_parameters()
    p.update(test_mode=True, num_workers=nw, num_local_schedulers=ns,
             mpi_rank=0, world_size=1, max_num_batched_tokens=8,
             max_num_batched_requests=8)
    pk = PersistentKernel(**p)
    qd = pk.attach_input(q, name="q")
    kd = pk.attach_input(k, name="k")
    vd = pk.attach_input(v, name="v")
    md = pk.attach_input(m, name="m")
    od = pk.attach_input(o, name="o")
    register_searched_task(pk, sched, inputs=[qd, kd, vd, md], output=od)
    pk.compile(output_dir=None)
    pk(); torch.cuda.synchronize()

    ref = torch.bmm(torch.exp(torch.bmm(q.float(), k.float()) + m.float()),
                    v.float())
    rel = ((o.float() - ref).abs().max() / ref.abs().max()).item()
    print("REL", rel, flush=True)
    sys.exit(0 if rel < 0.02 else 1)
    """
)


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
@pytest.mark.parametrize("s,fl", [(128, 2), (256, 4)])
def test_searched_attention_core_matches_torch(s, fl):
    """The FULL attention core -- exp(Q@K^T + mask) @ V: four inputs, a
    batched matmul, a chained matmul, and a multi-iteration K loop --
    discovered by search, registered as an MPK task, and numerically right.
    """
    out = _run(_ATTN_SRC.format(s=s, fl=fl),
               f"searched attention core S={s} fl={fl}", timeout=3600)
    assert "SCHEDULE" in out, out[-500:]
