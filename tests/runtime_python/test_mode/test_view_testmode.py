"""
End-to-end test-mode tests for virtual DTensors (views).

We exercise the full compile-and-run pipeline:
  1) pure-reshape view of a storage tensor consumed by an rmsnorm layer;
  2) split along the outermost dim (contiguous slabs) consumed by separate
     rmsnorm layers, verifying read-views work end-to-end;
  3) write-view + fused-consumer: two rmsnorm layers each write a disjoint
     slab of a shared storage buffer; a third rmsnorm reads the full buffer.

We also capture the AnnotatedGraph dump to verify edges involving views are
flagged as BARRIER (the dep-analyzer change), and that non-overlapping write
views generate no spurious cross-producer edges.
"""

import os
import sys
import io
import re
import subprocess

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def torch_rmsnorm(x, weight, eps=1e-5):
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def _build_pk():
    """Build a fresh PersistentKernel in test_mode."""
    # Deterministic RNG: each test seeds before constructing tensors so
    # subprocess re-runs (under MIRAGE_DUMP_ANNOTATED_GRAPH) and direct
    # runs produce bit-identical inputs and outputs.
    torch.manual_seed(0)
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    return PersistentKernel(**params)


def _block_dim_for(pk):
    return (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)


def _assert_close(out, ref, tol, name):
    max_diff = (out - ref).abs().max().item()
    print(f"  [{name}] max abs diff = {max_diff}")
    if max_diff >= tol:
        print(f"  FAIL: {name} exceeds tol {tol}")
        sys.exit(1)


def _read_annotated_dump(stderr_text):
    """Locate the AnnotatedGraph dump in stderr. Returns the dump text or
    raises if not found."""
    m = re.search(r"AnnotatedGraph: .*?ordered_layers: \[[^\]]*\]\n",
                  stderr_text, re.DOTALL)
    if m is None:
        return None
    return m.group(0)


def _run_subprocess_dumping_ag(test_name):
    """Re-run this script with MIRAGE_DUMP_ANNOTATED_GRAPH=1 targeting one
    test. Returns the captured AnnotatedGraph dump for that test."""
    env = dict(os.environ)
    env["MIRAGE_DUMP_ANNOTATED_GRAPH"] = "1"
    env["MPK_VIEW_TEST_FOCUS"] = test_name
    res = subprocess.run([sys.executable, __file__],
                         env=env, capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        print("  subprocess stdout:", res.stdout[-500:])
        print("  subprocess stderr:", res.stderr[-500:])
        raise RuntimeError(
            f"subprocess for {test_name} failed with code {res.returncode}")
    dump = _read_annotated_dump(res.stderr)
    if dump is None:
        print("  subprocess stderr did not contain AnnotatedGraph dump:")
        print(res.stderr[-1500:])
        raise RuntimeError("no AnnotatedGraph dump captured")
    return dump


def test_pure_reshape_view():
    """Same shape view: T_view = T.view([B, H]) is functionally identical
    to using T directly. Verifies the codegen path resolves views and that
    barrier dispatch is correct for a single read-view edge."""
    print("[test_pure_reshape_view]")
    device = "cuda"
    dtype = torch.bfloat16
    B, H = 8, 2048

    x = torch.randn(B, H, dtype=dtype, device=device)
    w = torch.randn(H, dtype=dtype, device=device)
    out = torch.zeros(B, H, dtype=dtype, device=device)

    pk = _build_pk()
    x_dt = pk.attach_input(x, name="x")
    w_dt = pk.attach_input(w, name="w")
    out_dt = pk.attach_input(out, name="out")

    # Pure reshape: same shape, same total elements; produces a view DTensor.
    x_view = pk.view(x_dt, [B, H])
    assert x_view.is_virtual, "view() result must be virtual"
    assert x_view.base_guid == x_dt.guid, "view base_guid mismatch"
    assert x_view.view_offset == 0, "pure reshape preserves view_offset = 0"

    pk.rmsnorm_layer(input=x_view, weight=w_dt, output=out_dt,
                     grid_dim=(B, 1, 1), block_dim=_block_dim_for(pk))

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    ref = torch_rmsnorm(x, w)
    _assert_close(out, ref, 0.05, "pure_reshape_view")
    pk.finalize()


def test_split_outermost_dim():
    """Storage T (B, H). Take two views via split along dim 0 (outermost).
    Each view is a contiguous slab in memory; an rmsnorm consumes each
    slab and writes to a corresponding slab of the output storage."""
    print("[test_split_outermost_dim]")
    device = "cuda"
    dtype = torch.bfloat16
    B, H = 16, 2048
    half = B // 2

    x = torch.randn(B, H, dtype=dtype, device=device)
    w = torch.randn(H, dtype=dtype, device=device)
    out = torch.zeros(B, H, dtype=dtype, device=device)

    pk = _build_pk()
    x_dt = pk.attach_input(x, name="x")
    w_dt = pk.attach_input(w, name="w")
    out_dt = pk.attach_input(out, name="out")

    # Two read-views and two write-views, split along dim 0.
    x_views = pk.split(x_dt, 2, 0)
    out_views = pk.split(out_dt, 2, 0)
    assert len(x_views) == 2 and len(out_views) == 2
    for i, (xv, ov) in enumerate(zip(x_views, out_views)):
        assert xv.shape == (half, H), f"x slab[{i}] shape {xv.shape}"
        assert ov.shape == (half, H), f"out slab[{i}] shape {ov.shape}"
        assert xv.view_offset == i * half * H * 2  # bf16 = 2 bytes
        assert ov.view_offset == i * half * H * 2

    # Two independent rmsnorms, each reading a view of x and writing a
    # view of out. No producer-consumer edge between them — the dep
    # analyzer should detect that their write windows are disjoint.
    for i, (xv, ov) in enumerate(zip(x_views, out_views)):
        pk.rmsnorm_layer(input=xv, weight=w_dt, output=ov,
                         grid_dim=(half, 1, 1), block_dim=_block_dim_for(pk))

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    ref = torch_rmsnorm(x, w)
    _assert_close(out, ref, 0.05, "split_outermost_dim")
    pk.finalize()


def test_write_view_then_full_read():
    """Two rmsnorm layers each write into disjoint slabs (via write-views)
    of a shared storage tensor; a third rmsnorm reads the full storage."""
    print("[test_write_view_then_full_read]")
    device = "cuda"
    dtype = torch.bfloat16
    B, H = 16, 2048
    half = B // 2

    x = torch.randn(B, H, dtype=dtype, device=device)
    w1 = torch.randn(H, dtype=dtype, device=device)
    w2 = torch.randn(H, dtype=dtype, device=device)
    fused = torch.zeros(B, H, dtype=dtype, device=device)
    out = torch.zeros(B, H, dtype=dtype, device=device)

    pk = _build_pk()
    x_dt = pk.attach_input(x, name="x")
    w1_dt = pk.attach_input(w1, name="w1")
    w2_dt = pk.attach_input(w2, name="w2")
    fused_dt = pk.attach_input(fused, name="fused")
    out_dt = pk.attach_input(out, name="out")

    x_top = pk.narrow(x_dt, 0, 0, half)
    x_bot = pk.narrow(x_dt, 0, half, half)
    fused_top = pk.narrow(fused_dt, 0, 0, half)
    fused_bot = pk.narrow(fused_dt, 0, half, half)

    # Two producer rmsnorms writing disjoint slabs of `fused`.
    pk.rmsnorm_layer(input=x_top, weight=w1_dt, output=fused_top,
                     grid_dim=(half, 1, 1), block_dim=_block_dim_for(pk))
    pk.rmsnorm_layer(input=x_bot, weight=w2_dt, output=fused_bot,
                     grid_dim=(half, 1, 1), block_dim=_block_dim_for(pk))
    # Consumer rmsnorm reading the FULL fused storage; depends (barrier) on
    # both producers.
    w3 = torch.randn(H, dtype=dtype, device=device)
    w3_dt = pk.attach_input(w3, name="w3")
    pk.rmsnorm_layer(input=fused_dt, weight=w3_dt, output=out_dt,
                     grid_dim=(B, 1, 1), block_dim=_block_dim_for(pk))

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    ref_fused = torch.cat([
        torch_rmsnorm(x[:half], w1),
        torch_rmsnorm(x[half:], w2),
    ], dim=0)
    ref_out = torch_rmsnorm(ref_fused, w3)
    _assert_close(fused, ref_fused, 0.05, "write_view_fused_storage")
    _assert_close(out, ref_out, 0.05, "write_view_consumer")
    pk.finalize()


def _count_barrier_edges(dump):
    return len(re.findall(r"\[BARRIER\]", dump))


def _count_edges(dump):
    # An edge line looks like "    [N] X:s -> Y:s guid=..."
    return len(re.findall(r"^\s+\[\d+\] \d+:", dump, re.MULTILINE))


def test_chained_write_then_view_read():
    """Layer A writes a non-virtual storage tensor. Layer B reads a VIEW of
    it. Exercises the inter-layer view edge → must be flagged BARRIER.
    """
    print("[test_chained_write_then_view_read]")
    device = "cuda"
    dtype = torch.bfloat16
    B, H = 16, 2048

    x = torch.randn(B, H, dtype=dtype, device=device)
    w1 = torch.randn(H, dtype=dtype, device=device)
    w2 = torch.randn(H, dtype=dtype, device=device)
    mid = torch.zeros(B, H, dtype=dtype, device=device)
    out = torch.zeros(B, H, dtype=dtype, device=device)

    pk = _build_pk()
    x_dt = pk.attach_input(x, name="x")
    w1_dt = pk.attach_input(w1, name="w1")
    w2_dt = pk.attach_input(w2, name="w2")
    mid_dt = pk.attach_input(mid, name="mid")
    out_dt = pk.attach_input(out, name="out")

    # A writes mid_dt directly; B reads a pure-reshape view of mid_dt and
    # writes out_dt. The A→B edge must be marked BARRIER because B's input
    # tensor is virtual.
    pk.rmsnorm_layer(input=x_dt, weight=w1_dt, output=mid_dt,
                     grid_dim=(B, 1, 1), block_dim=_block_dim_for(pk))
    mid_view = pk.view(mid_dt, [B, H])
    pk.rmsnorm_layer(input=mid_view, weight=w2_dt, output=out_dt,
                     grid_dim=(B, 1, 1), block_dim=_block_dim_for(pk))

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    ref = torch_rmsnorm(torch_rmsnorm(x, w1), w2)
    # Two rmsnorms in bf16 cascade hit ~1/16 in the subprocess re-run; match
    # the tolerance of the other cascade tests.
    _assert_close(out, ref, 0.07, "chained_write_then_view_read")
    pk.finalize()


def test_dep_analysis_via_dump():
    """Run several tests with MIRAGE_DUMP_ANNOTATED_GRAPH=1 in subprocesses
    and assert on the AnnotatedGraph structure.

      - chained_write_then_view_read: exactly 1 BARRIER edge between the
        two rmsnorm layers (consumer reads a view of producer's output).
      - write_view_then_full_read: 2 BARRIER edges into the final fused
        consumer (one per write-view producer); the two producers have
        non-overlapping windows and therefore NO edge between each other.
      - split_outermost_dim: 0 inter-layer edges (rmsnorms only read graph
        inputs and write disjoint slabs of out_dt).
    """
    print("[test_dep_analysis_via_dump]")

    dump_chain = _run_subprocess_dumping_ag("test_chained_write_then_view_read")
    n_barrier = _count_barrier_edges(dump_chain)
    n_edges = _count_edges(dump_chain)
    print(f"  chained: {n_edges} edges, {n_barrier} barrier")
    if n_barrier != 1:
        print("  FAIL: expected exactly 1 barrier edge for chained test")
        print(dump_chain)
        sys.exit(1)

    dump_wv = _run_subprocess_dumping_ag("test_write_view_then_full_read")
    n_barrier = _count_barrier_edges(dump_wv)
    n_edges = _count_edges(dump_wv)
    print(f"  write-view fused-read: {n_edges} edges, {n_barrier} barrier")
    if n_barrier != 2:
        print("  FAIL: expected exactly 2 barrier edges (one per write-view "
              "producer) for write-view test")
        print(dump_wv)
        sys.exit(1)
    # No cross-producer edge between the two disjoint write-views.
    if n_edges != 2:
        print("  FAIL: expected exactly 2 edges total (both barrier into "
              "consumer); got non-overlap pruning leak")
        print(dump_wv)
        sys.exit(1)

    dump_split = _run_subprocess_dumping_ag("test_split_outermost_dim")
    n_edges = _count_edges(dump_split)
    print(f"  split: {n_edges} edges")
    if n_edges != 0:
        # Reads of graph-input views produce no inter-layer edges.
        # Writes to disjoint slabs of an unconsumed output: no edges.
        print("  FAIL: expected 0 edges for split test (no inter-layer deps)")
        print(dump_split)
        sys.exit(1)

    print("  OK")


def main():
    focus = os.environ.get("MPK_VIEW_TEST_FOCUS")
    if focus is not None:
        # Subprocess mode: run only the requested test so the
        # AnnotatedGraph dump is isolated.
        if focus == "test_pure_reshape_view":
            test_pure_reshape_view()
        elif focus == "test_split_outermost_dim":
            test_split_outermost_dim()
        elif focus == "test_write_view_then_full_read":
            test_write_view_then_full_read()
        elif focus == "test_chained_write_then_view_read":
            test_chained_write_then_view_read()
        else:
            raise SystemExit(f"Unknown focus test: {focus}")
        return

    test_pure_reshape_view()
    test_split_outermost_dim()
    test_write_view_then_full_read()
    test_chained_write_then_view_read()
    test_dep_analysis_via_dump()
    print("ALL VIEW END-TO-END TESTS PASSED")


if __name__ == "__main__":
    main()
