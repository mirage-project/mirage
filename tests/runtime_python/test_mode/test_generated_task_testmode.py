"""Compare a COMPILER-GENERATED MPK task against the PyTorch reference.

This is the end-to-end check for the segment->task pipeline: the task body is
transpiled from a threadblock graph by the muGraph backend
(TranspilerConfig::emit_device_body) and registered as TASK_GENERATED, instead
of dispatching to a handwritten .cuh kernel.

Run:
    python -m pytest tests/runtime_python/test_mode/test_generated_task_testmode.py
"""

import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402


def _skip_reason():
    if not torch.cuda.is_available():
        return "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        return "generated task bodies are only emitted for the sm_100 backend"
    return None


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_generated_silu_mul_matches_torch():
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)

    batch_size = 8
    intermediate = 2048

    gate = torch.randn(batch_size, intermediate, dtype=dtype, device=device)
    up = torch.randn(batch_size, intermediate, dtype=dtype, device=device)
    out = torch.zeros(batch_size, intermediate, dtype=dtype, device=device)

    ref = (torch.nn.functional.silu(gate.float()) * up.float()).to(dtype)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    gate_dt = pk.attach_input(gate, name="gate")
    up_dt = pk.attach_input(up, name="up")
    out_dt = pk.attach_input(out, name="out")

    pk.generated_silu_mul_layer(
        gate=gate_dt,
        up=up_dt,
        output=out_dt,
        grid_dim=(intermediate // 64, 1, 1),
        block_dim=(256, 1, 1),
    )

    # output_dir=None -> a temp dir. Writing MPK artifacts into the source
    # tree leaves task_graph_rank0.json behind, which trips
    # test_segmented_mugraph.py's assert_no_task_graph_artifacts guard
    # (it checks that the segmented path emits no task graph, against a
    # baseline snapshot of this directory).
    pk.compile(output_dir=None)
    pk()
    torch.cuda.synchronize()

    rel = ((out.float() - ref.float()).abs().max()
           / ref.float().abs().max()).item()
    assert rel < 0.02, f"generated task rel error {rel}; out={out[0, :6].tolist()}"


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_generated_gate_up_multi_output_matches_torch():
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    m, k, n = 8, 256, 64
    x = torch.randn(m, k, dtype=dtype, device=device)
    wg = torch.randn(k, n, dtype=dtype, device=device)
    wu = torch.randn(k, n, dtype=dtype, device=device)
    gate = torch.zeros(m, n, dtype=dtype, device=device)
    up = torch.zeros_like(gate)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=m,
        max_num_batched_requests=m,
    )
    pk = PersistentKernel(**params)
    xd = pk.attach_input(x, name="x")
    gd = pk.attach_input(wg, name="wg")
    ud = pk.attach_input(wu, name="wu")
    gate_d = pk.attach_input(gate, name="gate")
    up_d = pk.attach_input(up, name="up")
    pk.generated_gate_up_layer(
        input=xd,
        gate_weight_t=gd,
        up_weight_t=ud,
        gate_output=gate_d,
        up_output=up_d,
        grid_dim=(1, 1, 1),
        block_dim=(256, 1, 1),
        forloop_range=k // 64,
    )
    pk.compile(output_dir=None)
    pk()
    torch.cuda.synchronize()

    gate_ref = x.float() @ wg.float()
    up_ref = x.float() @ wu.float()
    gate_rel = ((gate.float() - gate_ref).abs().max()
                / gate_ref.abs().max()).item()
    up_rel = ((up.float() - up_ref).abs().max()
              / up_ref.abs().max()).item()
    assert max(gate_rel, up_rel) < 0.02


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_elementwise_add_hidden_grid_matches_torch():
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    m, hidden = 8, 1024
    a = torch.randn(m, hidden, dtype=dtype, device=device)
    b = torch.randn_like(a)
    out = torch.zeros_like(a)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=m,
        max_num_batched_requests=m,
    )
    pk = PersistentKernel(**params)
    ad = pk.attach_input(a, name="a")
    bd = pk.attach_input(b, name="b")
    od = pk.attach_input(out, name="out")
    pk.elementwise_add_layer(
        input_a=ad,
        input_b=bd,
        output=od,
        grid_dim=(hidden // 64, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.compile(output_dir=None)
    pk()
    torch.cuda.synchronize()

    assert torch.equal(out, (a.float() + b.float()).to(dtype))


# M covers both 1-SM MMA tiles (64, 128) and a decode shape that goes through
# swapAB (8). K = N = 64 keeps every operand tile at a 128B pitch, which is what
# the Blackwell backend supports without panel tiling.
@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
@pytest.mark.parametrize("M", [8, 64, 128])
def test_generated_linear_matches_torch(M):
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    K = N = 64

    x = torch.randn(M, K, dtype=dtype, device=device)
    # already transposed: the threadblock matmul consumes A(M,K) @ B(K,N)
    weight_t = torch.randn(K, N, dtype=dtype, device=device)
    out = torch.zeros(M, N, dtype=dtype, device=device)

    ref = x.float() @ weight_t.float()

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = max(M, 8)
    params["max_num_batched_requests"] = max(M, 8)
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x, name="x")
    w_dt = pk.attach_input(weight_t, name="weight_t")
    out_dt = pk.attach_input(out, name="out")

    pk.generated_linear_layer(
        input=x_dt,
        weight_t=w_dt,
        output=out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(256, 1, 1),
    )

    # output_dir=None -> temp dir (see the note above).
    pk.compile(output_dir=None)
    pk()
    torch.cuda.synchronize()

    rel = ((out.float() - ref).abs().max() / ref.abs().max()).item()
    # The bug this test exists to catch is a transposed write: the tile was
    # numerically correct but stored column-major, `out == ref.T` exactly,
    # which shows up here as rel ~1.5.
    assert rel < 0.02, (
        f"generated matmul task rel error {rel} at M={M}; "
        f"out[0,:4]={out[0, :4].tolist()} ref[0,:4]={ref[0, :4].tolist()}")


# The K-loop runs in a subprocess with a hard timeout. Its failure mode is a
# deadlock inside the megakernel, not an exception: an operand pipeline whose
# consumer count or transaction-byte count is wrong blocks forever. In-process
# that hangs pytest instead of failing it.
_KLOOP_SRC = textwrap.dedent(
    """
    import sys, torch, mirage
    from mirage.mpk.persistent_kernel import PersistentKernel
    M, K, N, FL = {m}, {k}, {n}, {fl}
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    wt = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nw, ns = mirage.get_configurations_from_gpu(0)
    p = PersistentKernel.get_default_init_parameters()
    p.update(test_mode=True, num_workers=nw, num_local_schedulers=ns,
             mpi_rank=0, world_size=1, max_num_batched_tokens=max(M, 8),
             max_num_batched_requests=max(M, 8))
    pk = PersistentKernel(**p)
    xd = pk.attach_input(x, name="x")
    wd = pk.attach_input(wt, name="wt")
    od = pk.attach_input(o, name="o")
    pk.generated_linear_layer(input=xd, weight_t=wd, output=od,
                              grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
                              forloop_range=FL, activation="{act}")
    pk.compile(output_dir=None)
    pk(); torch.cuda.synchronize()
    mm = (x.float() @ wt.float()).to(torch.bfloat16).float()
    ref = {{"none": mm,
           "silu": torch.nn.functional.silu(mm),
           "gelu": torch.nn.functional.gelu(mm),
           "relu": torch.nn.functional.relu(mm)}}["{act}"]
    rel = ((o.float() - ref).abs().max()
           / ref.abs().max().clamp(min=1e-3)).item()
    print("REL", rel)
    sys.exit(0 if rel < 0.02 else 1)
    """
)


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
# FL=16 at K=1024 is the Qwen3-0.6B hidden dim -- the shape a real fused segment
# has. M=8 pairs the K-loop with swapAB, which is the decode case.
@pytest.mark.parametrize("m,k,fl", [(128, 128, 2), (128, 256, 4), (8, 256, 4),
                                    (128, 1024, 16)])
def test_generated_linear_k_loop(m, k, fl):
    """K split across forloop iterations inside a megakernel task.

    This is what needs the TMA descriptors: a pipelined operand is a TMA load,
    and a task body has no kernel parameters to receive one through. The muGraph
    backend emits a host builder alongside the body, the task loader calls it,
    and the body reinterprets the device-resident atom it uploaded.
    """
    _run_kloop(m, k, fl, "none", f"K-loop M={m} K={k} FL={fl}")


def _run_generated(src, label, timeout=900):
    """Run a generated-task probe in a subprocess so a hang surfaces as a
    test failure instead of wedging the whole session (the failure mode of a
    bad barrier/pipeline emission is a deadlocked persistent kernel)."""
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    try:
        proc = subprocess.run([sys.executable, "-c", src], timeout=timeout,
                              capture_output=True, text=True, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail(f"generated {label} deadlocked")
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr[-2000:]}"


def _run_kloop(m, k, fl, act, label):
    _run_generated(_KLOOP_SRC.format(m=m, k=k, n=64, fl=fl, act=act), label)


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
# A FUSED SEGMENT: the K-looped matmul and its activation are ONE task, so the
# matmul result never round-trips through global memory. This is the shape the
# segment->task pipeline exists to produce.
@pytest.mark.parametrize("m,k,fl,act", [(128, 128, 2, "silu"),
                                        (128, 256, 4, "gelu"),
                                        (128, 256, 4, "relu"),
                                        (8, 256, 4, "silu"),
                                        (128, 1024, 16, "silu")])
def test_generated_linear_fused_activation(m, k, fl, act):
    _run_kloop(m, k, fl, act, f"fused {act} M={m} K={k} FL={fl}")


_SWIGLU_SRC = textwrap.dedent(
    """
    import sys, torch, mirage
    from mirage.mpk.persistent_kernel import PersistentKernel
    M, K, N, FL = {m}, {k}, {n}, {fl}
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    wg = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    wu = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
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
    pk.generated_swiglu_layer(input=xd, gate_weight_t=gd, up_weight_t=ud,
                              output=od, grid_dim=(1, 1, 1),
                              block_dim=(256, 1, 1), forloop_range=FL)
    pk.compile(output_dir=None)
    pk(); torch.cuda.synchronize()
    g = (x.float() @ wg.float()).to(torch.bfloat16).float()
    u = (x.float() @ wu.float()).to(torch.bfloat16).float()
    ref = torch.nn.functional.silu(g) * u
    rel = ((o.float() - ref).abs().max() / ref.abs().max()).item()
    print("REL", rel)
    sys.exit(0 if rel < 0.02 else 1)
    """
)


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
# The full gated-MLP segment: TWO K-looped matmuls plus silu and a multiply, all
# in one task. Two matmuls in one body regressed twice before this worked --
# aliased TMEM accumulators (rel ~2.4) and an MMA-mbarrier arrival count that
# only counted iterations, which deadlocked. Both failure modes are shape-
# independent, so any of these cases catches a reintroduction.
@pytest.mark.parametrize("m,k,fl", [(128, 128, 2), (8, 256, 4), (128, 1024, 16)])
def test_generated_swiglu_segment(m, k, fl):
    src = _SWIGLU_SRC.format(m=m, k=k, n=64, fl=fl)
    _run_generated(src, f"SwiGLU M={m} K={k} FL={fl}", timeout=900)


_SILU_DOWN_SRC = textwrap.dedent(
    """
    import sys, torch, mirage
    from mirage.mpk.persistent_kernel import PersistentKernel
    M, K, N, FL = {m}, {k}, {n}, {fl}
    torch.manual_seed(0)
    gate = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    up = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nw, ns = mirage.get_configurations_from_gpu(0)
    p = PersistentKernel.get_default_init_parameters()
    p.update(test_mode=True, num_workers=nw, num_local_schedulers=ns,
             mpi_rank=0, world_size=1, max_num_batched_tokens=max(M, 8),
             max_num_batched_requests=max(M, 8))
    pk = PersistentKernel(**p)
    gd = pk.attach_input(gate, name="gate")
    ud = pk.attach_input(up, name="up")
    wd = pk.attach_input(weight, name="weight")
    od = pk.attach_input(out, name="out")
    pk.generated_silu_mul_linear_layer(
        gate=gd, up=ud, weight_t=wd, output=od,
        grid_dim=(N // 64, 1, 1), block_dim=(256, 1, 1),
        forloop_range=FL)
    pk.compile(output_dir=None)
    pk(); torch.cuda.synchronize()
    mid = (torch.nn.functional.silu(gate.float()) * up.float())
    ref = mid.to(torch.bfloat16).float() @ weight.float()
    rel = ((out.float() - ref).abs().max() / ref.abs().max()).item()
    print("REL", rel)
    sys.exit(0 if rel < 0.02 else 1)
    """
)


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
@pytest.mark.parametrize("m,k,fl,n", [(8, 256, 4, 64),
                                       (128, 256, 4, 128)])
def test_generated_silu_mul_down_segment(m, k, fl, n):
    """SwiGLU and its consuming down projection stay in one task."""
    src = _SILU_DOWN_SRC.format(m=m, k=k, n=n, fl=fl)
    _run_generated(src, f"SiLU+down M={m} K={k} FL={fl} N={n}", timeout=900)


_WORKER_SRC = textwrap.dedent(
    """
    import sys, torch, mirage
    from mirage.mpk.persistent_kernel import PersistentKernel
    G, NW = {g}, {nw}
    M, K, N = 128, 64, 64 * G
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    wt = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    _, ns = mirage.get_configurations_from_gpu(0)
    p = PersistentKernel.get_default_init_parameters()
    p.update(test_mode=True, num_workers=NW, num_local_schedulers=ns, mpi_rank=0,
             world_size=1, max_num_batched_tokens=M, max_num_batched_requests=M)
    pk = PersistentKernel(**p)
    xd = pk.attach_input(x, name="x")
    wd = pk.attach_input(wt, name="wt")
    od = pk.attach_input(o, name="o")
    pk.generated_linear_layer(input=xd, weight_t=wd, output=od,
                              grid_dim=(G, 1, 1), block_dim=(256, 1, 1))
    pk.compile(output_dir=None)
    pk(); torch.cuda.synchronize()
    ref = x.float() @ wt.float()
    rel = ((o.float() - ref).abs().max() / ref.abs().max()).item()
    print("REL", rel)
    sys.exit(0 if rel < 0.02 else 1)
    """
)


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
# MORE TASKS THAN WORKERS -- the case every other test here misses, because the
# GPU has ~128 workers and these graphs have far fewer tasks. A generated matmul
# body used to relinquish its TMEM allocation permit
# (tcgen05.relinquish_alloc_permit) on the way out, which is correct for a
# standalone kernel but tells the hardware the CTA will never allocate again:
# the SECOND generated matmul on a given worker then died with "unspecified
# launch failure". Keep num_workers small so workers really do repeat.
@pytest.mark.parametrize("g,nw", [(4, 1), (16, 2), (48, 4)])
def test_generated_linear_multiple_tasks_per_worker(g, nw):
    src = _WORKER_SRC.format(g=g, nw=nw)
    _run_generated(src, f"tasks={g} workers={nw}", timeout=900)


_TILED_SRC = textwrap.dedent(
    """
    import sys, torch, mirage
    from mirage.mpk.persistent_kernel import PersistentKernel
    M, K, FL, G = {m}, {k}, {fl}, {g}
    N = 64 * G
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    wt = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    o = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    nw, ns = mirage.get_configurations_from_gpu(0)
    p = PersistentKernel.get_default_init_parameters()
    p.update(test_mode=True, num_workers=nw, num_local_schedulers=ns, mpi_rank=0,
             world_size=1, max_num_batched_tokens=max(M, 8),
             max_num_batched_requests=max(M, 8))
    pk = PersistentKernel(**p)
    xd = pk.attach_input(x, name="x")
    wd = pk.attach_input(wt, name="wt")
    od = pk.attach_input(o, name="o")
    pk.generated_linear_layer(input=xd, weight_t=wd, output=od,
                              grid_dim=(G, 1, 1), block_dim=(256, 1, 1),
                              forloop_range=FL)
    pk.compile(output_dir=None)
    pk(); torch.cuda.synchronize()
    ref = x.float() @ wt.float()
    rel = ((o.float() - ref).abs().max() / ref.abs().max()).item()
    print("REL", rel)
    sys.exit(0 if rel < 0.02 else 1)
    """
)


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
# swapAB (M < 64) COMBINED with N tiled across tasks and a K-loop. Every other
# swapAB case here uses grid_dim=1, which is exactly why this got through:
# InputTMAAsyncCopy_Blackwell derived its gmem tile coordinate from blockIdx,
# which in a task body is the WORKER id. That coordinate reaches local_tile as
# the M coord for an A operand and the N coord for a B operand -- and a
# megakernel grid is 1-D, so blockIdx.y is always 0 and a B operand was
# accidentally harmless. swapAB makes the TILED weight the A operand, so its
# coordinate became the worker id and each task read whichever tile its worker
# happened to be (rel ~1.3). M=128 is the non-swapAB control.
@pytest.mark.parametrize("m,k,fl,g", [(8, 128, 2, 2), (8, 256, 4, 4),
                                      (16, 128, 2, 4), (128, 128, 2, 2)])
def test_generated_linear_tiled_swapab(m, k, fl, g):
    src = _TILED_SRC.format(m=m, k=k, fl=fl, g=g)
    _run_generated(src, f"tiled M={m} K={k} FL={fl} grid={g}", timeout=900)


_QWEN_MLP_SRC = textwrap.dedent(
    """
    import sys, torch, mirage
    from mirage.mpk.persistent_kernel import PersistentKernel
    M, H, I = {m}, 1024, 3072            # Qwen3-0.6B hidden / intermediate
    torch.manual_seed(0)
    x  = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
    wg = torch.randn(H, I, dtype=torch.bfloat16, device="cuda")
    wu = torch.randn(H, I, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(I, H, dtype=torch.bfloat16, device="cuda")
    mid = torch.zeros(M, I, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
    nw, ns = mirage.get_configurations_from_gpu(0)
    p = PersistentKernel.get_default_init_parameters()
    p.update(test_mode=True, num_workers=nw, num_local_schedulers=ns, mpi_rank=0,
             world_size=1, max_num_batched_tokens=max(M, 8),
             max_num_batched_requests=max(M, 8))
    pk = PersistentKernel(**p)
    xd = pk.attach_input(x, name="x")
    gd = pk.attach_input(wg, name="wg")
    ud = pk.attach_input(wu, name="wu")
    w2d = pk.attach_input(w2, name="w2")
    md = pk.attach_input(mid, name="mid")
    od = pk.attach_input(out, name="out")
    pk.generated_swiglu_layer(input=xd, gate_weight_t=gd, up_weight_t=ud,
                              output=md, grid_dim=(I // 64, 1, 1),
                              block_dim=(256, 1, 1), forloop_range=H // 64)
    pk.generated_linear_layer(input=md, weight_t=w2d, output=od,
                              grid_dim=(H // 64, 1, 1), block_dim=(256, 1, 1),
                              forloop_range=I // 64)
    pk.compile(output_dir=None)
    pk(); torch.cuda.synchronize()
    g = (x.float() @ wg.float()).to(torch.bfloat16).float()
    u = (x.float() @ wu.float()).to(torch.bfloat16).float()
    mid_ref = (torch.nn.functional.silu(g) * u).to(torch.bfloat16)
    ref = mid_ref.float() @ w2.float()
    rel = ((out.float() - ref).abs().max() / ref.abs().max()).item()
    print("REL", rel)
    sys.exit(0 if rel < 0.02 else 1)
    """
)


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
# A whole Qwen3-0.6B MLP block on generated tasks only: swiglu -> down-proj,
# TWO generated tasks in ONE graph. That combination is what caught
# TMAParams::input_id (index among the GRAPH's inputs) being used where a task
# needs the OP's operand position -- identical for a single-op graph, and
# .at() range-checked out on the second op. Real model shapes on purpose.
@pytest.mark.parametrize("m", [8, 128])
def test_generated_qwen3_mlp_block(m):
    src = _QWEN_MLP_SRC.format(m=m)
    _run_generated(src, f"Qwen3 MLP M={m}", timeout=1200)


_LAYER_CHAIN_SRC = textwrap.dedent(
    """
    import sys, torch, mirage
    from mirage.mpk.persistent_kernel import PersistentKernel
    M, H, I = {m}, 1024, 3072
    torch.manual_seed(0)
    x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(1, H, dtype=torch.bfloat16, device="cuda")
    e = torch.full((M, 1), 1e-6, dtype=torch.bfloat16, device="cuda")
    wg = torch.randn(H, I, dtype=torch.bfloat16, device="cuda")
    wu = torch.randn(H, I, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(I, H, dtype=torch.bfloat16, device="cuda")
    normed = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
    mid = torch.zeros(M, I, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(M, H, dtype=torch.bfloat16, device="cuda")
    nw, ns = mirage.get_configurations_from_gpu(0)
    p = PersistentKernel.get_default_init_parameters()
    p.update(test_mode=True, num_workers=nw, num_local_schedulers=ns,
             mpi_rank=0, world_size=1, max_num_batched_tokens=max(M, 8),
             max_num_batched_requests=max(M, 8))
    pk = PersistentKernel(**p)
    xd = pk.attach_input(x, name="x")
    wd = pk.attach_input(w, name="w")
    ed = pk.attach_input(e, name="eps")
    nd = pk.attach_input(normed, name="normed")
    gd = pk.attach_input(wg, name="wg")
    ud = pk.attach_input(wu, name="wu")
    w2d = pk.attach_input(w2, name="w2")
    md = pk.attach_input(mid, name="mid")
    od = pk.attach_input(out, name="out")
    pk.generated_rmsnorm_layer(input=xd, weight=wd, eps_tensor=ed, output=nd,
                               grid_dim=(1, 1, 1), block_dim=(256, 1, 1))
    pk.generated_swiglu_layer(input=nd, gate_weight_t=gd, up_weight_t=ud,
                              output=md, grid_dim=(I // 64, 1, 1),
                              block_dim=(256, 1, 1), forloop_range=H // 64)
    pk.generated_linear_layer(input=md, weight_t=w2d, output=od,
                              grid_dim=(H // 64, 1, 1), block_dim=(256, 1, 1),
                              forloop_range=I // 64)
    pk.compile(output_dir=None)
    pk(); torch.cuda.synchronize()
    xf = x.float()
    n_ref = (xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
             * w.float()).to(torch.bfloat16)
    g_ref = (n_ref.float() @ wg.float()).to(torch.bfloat16).float()
    u_ref = (n_ref.float() @ wu.float()).to(torch.bfloat16).float()
    mid_ref = (torch.nn.functional.silu(g_ref) * u_ref).to(torch.bfloat16)
    ref = mid_ref.float() @ w2.float()
    rel = ((out.float() - ref).abs().max() / ref.abs().max()).item()
    print("REL", rel)
    sys.exit(0 if rel < 0.05 else 1)
    """
)


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
# The LINEAR PATH of a Qwen3-0.6B decode layer on generated tasks only:
# rmsnorm -> SwiGLU -> down-proj, three generated task types chained through
# gmem in one megakernel. The rmsnorm exercises the four fixes its
# decomposition required (reduction reconstruction, (M,1) broadcast, in-loop
# SYNCTHREADS, fused epilogue scalars). Tolerance 0.05: four chained bf16
# stages compound; each stage is individually ~3e-3.
#
# M=8, not M=1: swapAB puts the token dim in the MMA's N, and tcgen05 needs N
# to be a multiple of 8, so a generated MATMUL at M=1 is (correctly) rejected.
# M=1 decode therefore needs the caller to pad rows to 8 -- which MPK's
# max_num_batched_tokens=max(M, 8) already implies -- and is a builder
# concern, not a compiler one. generated_rmsnorm works at M=1 as-is.
@pytest.mark.parametrize("m", [8])
def test_generated_qwen3_layer_linear_path(m):
    src = _LAYER_CHAIN_SRC.format(m=m)
    _run_generated(src, f"layer chain M={m}", timeout=1200)
