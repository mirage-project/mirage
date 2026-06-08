"""
Corner-case end-to-end tests for virtual DTensors (views).

These cover patterns where views appear at fork points, join points, in
cascades, at multi-level views, and with overlapping write-views — i.e.
every structural shape we can think of that interacts with the dep
analyzer differently from the basic read/write-view cases in
test_view_testmode.py.

All tests use a small rmsnorm layer as the unit because (a) it accepts
one input, one weight, one output; (b) it's stride-respecting; and (c) a
single PyTorch reference matches the kernel output to ~bf16 ulp.

Each test sets torch.manual_seed(0) inside _build_pk() for deterministic
output and reproducible subprocess re-runs.
"""

import os
import re
import subprocess
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def torch_rmsnorm(x, weight, eps=1e-5):
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def _build_pk():
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
        print(f"    out[:2,:8] =\n{out[:2, :8]}")
        print(f"    ref[:2,:8] =\n{ref[:2, :8]}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Test 1. Fork point with two sibling read-views.
#
#    P (writes T) ──> A (reads T_left = T.narrow(0, 0, half))
#                 └─> B (reads T_right = T.narrow(0, half, half))
#
# After our refactor, P is a fork-producer (2 distinct consumers); both
# outgoing edges are barrier. The fork bundle's LCM degrades to a single
# event spanning all of P's tasks, launching both A and B together. This
# is the QKV-style read-view pattern.
# ---------------------------------------------------------------------------
def test_fork_with_sibling_read_views():
    print("[test_fork_with_sibling_read_views]")
    device = "cuda"
    dtype = torch.bfloat16
    B, H = 16, 2048
    half = B // 2

    x = torch.randn(B, H, dtype=dtype, device=device)
    w_p = torch.randn(H, dtype=dtype, device=device)
    w_a = torch.randn(H, dtype=dtype, device=device)
    w_b = torch.randn(H, dtype=dtype, device=device)
    T = torch.zeros(B, H, dtype=dtype, device=device)
    out_a = torch.zeros(half, H, dtype=dtype, device=device)
    out_b = torch.zeros(half, H, dtype=dtype, device=device)

    pk = _build_pk()
    x_dt = pk.attach_input(x, name="x")
    w_p_dt = pk.attach_input(w_p, name="w_p")
    w_a_dt = pk.attach_input(w_a, name="w_a")
    w_b_dt = pk.attach_input(w_b, name="w_b")
    T_dt = pk.attach_input(T, name="T")
    out_a_dt = pk.attach_input(out_a, name="out_a")
    out_b_dt = pk.attach_input(out_b, name="out_b")

    # Producer P writes T fully.
    pk.rmsnorm_layer(input=x_dt, weight=w_p_dt, output=T_dt,
                     grid_dim=(B, 1, 1), block_dim=_block_dim_for(pk))
    T_left = pk.narrow(T_dt, 0, 0, half)
    T_right = pk.narrow(T_dt, 0, half, half)
    # A and B each read a disjoint view of T. Both are barrier in-edges
    # from the same producer P, so P becomes a fork-producer and the
    # fork-bundle LCM collapses everything into one event.
    pk.rmsnorm_layer(input=T_left, weight=w_a_dt, output=out_a_dt,
                     grid_dim=(half, 1, 1), block_dim=_block_dim_for(pk))
    pk.rmsnorm_layer(input=T_right, weight=w_b_dt, output=out_b_dt,
                     grid_dim=(half, 1, 1), block_dim=_block_dim_for(pk))

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    ref_T = torch_rmsnorm(x, w_p)
    ref_a = torch_rmsnorm(ref_T[:half], w_a)
    ref_b = torch_rmsnorm(ref_T[half:], w_b)
    _assert_close(out_a, ref_a, 0.05, "fork_sibling_view_left")
    _assert_close(out_b, ref_b, 0.05, "fork_sibling_view_right")
    pk.finalize()


# ---------------------------------------------------------------------------
# Test 2. Mixed fork: one branch reads the parent directly, the other
# reads a view.
#
#    P ──> A (reads T directly)        non-barrier edge
#       └─> B (reads T.narrow(...))    barrier edge
#
# Fork-producer P with one fine-grained and one barrier outgoing edge.
# The fork LCM should degrade BOTH branches to event_dim=1 because the
# barrier branch's last3 equals full grid_dim.
# ---------------------------------------------------------------------------
def test_mixed_fork_view_and_full():
    print("[test_mixed_fork_view_and_full]")
    device = "cuda"
    dtype = torch.bfloat16
    B, H = 16, 2048
    half = B // 2

    x = torch.randn(B, H, dtype=dtype, device=device)
    w_p = torch.randn(H, dtype=dtype, device=device)
    w_a = torch.randn(H, dtype=dtype, device=device)
    w_b = torch.randn(H, dtype=dtype, device=device)
    T = torch.zeros(B, H, dtype=dtype, device=device)
    out_a = torch.zeros(B, H, dtype=dtype, device=device)   # full
    out_b = torch.zeros(half, H, dtype=dtype, device=device)  # narrowed

    pk = _build_pk()
    x_dt = pk.attach_input(x, name="x")
    w_p_dt = pk.attach_input(w_p, name="w_p")
    w_a_dt = pk.attach_input(w_a, name="w_a")
    w_b_dt = pk.attach_input(w_b, name="w_b")
    T_dt = pk.attach_input(T, name="T")
    out_a_dt = pk.attach_input(out_a, name="out_a")
    out_b_dt = pk.attach_input(out_b, name="out_b")

    pk.rmsnorm_layer(input=x_dt, weight=w_p_dt, output=T_dt,
                     grid_dim=(B, 1, 1), block_dim=_block_dim_for(pk))
    T_half = pk.narrow(T_dt, 0, 0, half)
    pk.rmsnorm_layer(input=T_dt, weight=w_a_dt, output=out_a_dt,
                     grid_dim=(B, 1, 1), block_dim=_block_dim_for(pk))
    pk.rmsnorm_layer(input=T_half, weight=w_b_dt, output=out_b_dt,
                     grid_dim=(half, 1, 1), block_dim=_block_dim_for(pk))

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    ref_T = torch_rmsnorm(x, w_p)
    ref_a = torch_rmsnorm(ref_T, w_a)
    ref_b = torch_rmsnorm(ref_T[:half], w_b)
    _assert_close(out_a, ref_a, 0.05, "mixed_fork_full")
    _assert_close(out_b, ref_b, 0.07, "mixed_fork_view")
    pk.finalize()


# ---------------------------------------------------------------------------
# Test 3. Non-overlap write-views with disjoint readers — verifying that
# the dep analyzer does NOT cross-connect.
#
#    A writes T_left  ─────────> C reads T_left  (C depends ONLY on A)
#    B writes T_right ─────────> D reads T_right (D depends ONLY on B)
#
# T is a shared storage buffer but each (writer, reader) pair touches a
# disjoint window. The window-overlap pruning should keep A→C and B→D
# while pruning A→D and B→C.
# ---------------------------------------------------------------------------
def test_disjoint_writers_disjoint_readers():
    print("[test_disjoint_writers_disjoint_readers]")
    device = "cuda"
    dtype = torch.bfloat16
    B, H = 16, 2048
    half = B // 2

    xa = torch.randn(half, H, dtype=dtype, device=device)
    xb = torch.randn(half, H, dtype=dtype, device=device)
    wa = torch.randn(H, dtype=dtype, device=device)
    wb = torch.randn(H, dtype=dtype, device=device)
    wc = torch.randn(H, dtype=dtype, device=device)
    wd = torch.randn(H, dtype=dtype, device=device)
    T = torch.zeros(B, H, dtype=dtype, device=device)
    out_c = torch.zeros(half, H, dtype=dtype, device=device)
    out_d = torch.zeros(half, H, dtype=dtype, device=device)

    pk = _build_pk()
    xa_dt = pk.attach_input(xa, name="xa")
    xb_dt = pk.attach_input(xb, name="xb")
    wa_dt = pk.attach_input(wa, name="wa")
    wb_dt = pk.attach_input(wb, name="wb")
    wc_dt = pk.attach_input(wc, name="wc")
    wd_dt = pk.attach_input(wd, name="wd")
    T_dt = pk.attach_input(T, name="T")
    out_c_dt = pk.attach_input(out_c, name="out_c")
    out_d_dt = pk.attach_input(out_d, name="out_d")

    T_left = pk.narrow(T_dt, 0, 0, half)
    T_right = pk.narrow(T_dt, 0, half, half)
    T_left_view = pk.narrow(T_dt, 0, 0, half)
    T_right_view = pk.narrow(T_dt, 0, half, half)

    pk.rmsnorm_layer(input=xa_dt, weight=wa_dt, output=T_left,
                     grid_dim=(half, 1, 1), block_dim=_block_dim_for(pk))
    pk.rmsnorm_layer(input=xb_dt, weight=wb_dt, output=T_right,
                     grid_dim=(half, 1, 1), block_dim=_block_dim_for(pk))
    pk.rmsnorm_layer(input=T_left_view, weight=wc_dt, output=out_c_dt,
                     grid_dim=(half, 1, 1), block_dim=_block_dim_for(pk))
    pk.rmsnorm_layer(input=T_right_view, weight=wd_dt, output=out_d_dt,
                     grid_dim=(half, 1, 1), block_dim=_block_dim_for(pk))

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    ref_c = torch_rmsnorm(torch_rmsnorm(xa, wa), wc)
    ref_d = torch_rmsnorm(torch_rmsnorm(xb, wb), wd)
    # Two-stage rmsnorm cascade through bf16 accumulates a little; use
    # the same tolerance as test_cascading_view_chain.
    _assert_close(out_c, ref_c, 0.07, "disjoint_writers_readers_c")
    _assert_close(out_d, ref_d, 0.07, "disjoint_writers_readers_d")
    pk.finalize()


# ---------------------------------------------------------------------------
# Test 4. Cascading chain of views: each layer's output is consumed
# through a view by the next layer.
#
#    A writes T1 → B reads T1.view([...]) → B writes T2 → C reads T2.view([...])
#
# Verifies that barrier edges chain correctly across multiple layers.
# ---------------------------------------------------------------------------
def test_cascading_view_chain():
    print("[test_cascading_view_chain]")
    device = "cuda"
    dtype = torch.bfloat16
    B, H = 16, 2048

    x = torch.randn(B, H, dtype=dtype, device=device)
    w_a = torch.randn(H, dtype=dtype, device=device)
    w_b = torch.randn(H, dtype=dtype, device=device)
    w_c = torch.randn(H, dtype=dtype, device=device)
    T1 = torch.zeros(B, H, dtype=dtype, device=device)
    T2 = torch.zeros(B, H, dtype=dtype, device=device)
    out = torch.zeros(B, H, dtype=dtype, device=device)

    pk = _build_pk()
    x_dt = pk.attach_input(x, name="x")
    w_a_dt = pk.attach_input(w_a, name="w_a")
    w_b_dt = pk.attach_input(w_b, name="w_b")
    w_c_dt = pk.attach_input(w_c, name="w_c")
    T1_dt = pk.attach_input(T1, name="T1")
    T2_dt = pk.attach_input(T2, name="T2")
    out_dt = pk.attach_input(out, name="out")

    pk.rmsnorm_layer(input=x_dt, weight=w_a_dt, output=T1_dt,
                     grid_dim=(B, 1, 1), block_dim=_block_dim_for(pk))
    T1_view = pk.view(T1_dt, [B, H])
    pk.rmsnorm_layer(input=T1_view, weight=w_b_dt, output=T2_dt,
                     grid_dim=(B, 1, 1), block_dim=_block_dim_for(pk))
    T2_view = pk.view(T2_dt, [B, H])
    pk.rmsnorm_layer(input=T2_view, weight=w_c_dt, output=out_dt,
                     grid_dim=(B, 1, 1), block_dim=_block_dim_for(pk))

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    ref = torch_rmsnorm(torch_rmsnorm(torch_rmsnorm(x, w_a), w_b), w_c)
    _assert_close(out, ref, 0.07, "cascading_view_chain")
    pk.finalize()


# ---------------------------------------------------------------------------
# Test 5. Multi-level view (view-of-view): producer writes storage T;
# consumer reads a view derived from a chain of views.
#
#    A writes T → V = T.narrow(...) → W = V.view(...) → B reads W
#
# W.base_guid must be T.guid (flattened to root). The codegen must
# correctly compute W's pointer as T's IODesc + accumulated view_offset.
# ---------------------------------------------------------------------------
def test_multi_level_view_codegen():
    print("[test_multi_level_view_codegen]")
    device = "cuda"
    dtype = torch.bfloat16
    B, H = 16, 2048
    half = B // 2

    x = torch.randn(B, H, dtype=dtype, device=device)
    w_a = torch.randn(H, dtype=dtype, device=device)
    w_b = torch.randn(H, dtype=dtype, device=device)
    T = torch.zeros(B, H, dtype=dtype, device=device)
    out = torch.zeros(half, H, dtype=dtype, device=device)

    pk = _build_pk()
    x_dt = pk.attach_input(x, name="x")
    w_a_dt = pk.attach_input(w_a, name="w_a")
    w_b_dt = pk.attach_input(w_b, name="w_b")
    T_dt = pk.attach_input(T, name="T")
    out_dt = pk.attach_input(out, name="out")

    pk.rmsnorm_layer(input=x_dt, weight=w_a_dt, output=T_dt,
                     grid_dim=(B, 1, 1), block_dim=_block_dim_for(pk))
    # Two levels of views over T.
    V = pk.narrow(T_dt, 0, half, half)  # rows [half, B)
    W = pk.view(V, [half, H])           # same shape, pure reshape — should
                                        # flatten to T at construction.
    # Sanity-check flattening at the Python level.
    assert W.base_guid == T_dt.guid, "view-of-view must flatten to root"
    assert W.view_offset == half * H * 2, "view-of-view offset accumulates"
    pk.rmsnorm_layer(input=W, weight=w_b_dt, output=out_dt,
                     grid_dim=(half, 1, 1), block_dim=_block_dim_for(pk))

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    ref_T = torch_rmsnorm(x, w_a)
    ref_out = torch_rmsnorm(ref_T[half:], w_b)
    _assert_close(out, ref_out, 0.05, "multi_level_view_codegen")
    pk.finalize()


# ---------------------------------------------------------------------------
# Test 6. Overlapping write-views with disjoint sources writing to the
# same window. With "trust the user" semantics, the compiler should NOT
# reject this, the dep analyzer should still create an edge to a reader
# of the parent, and the runtime should execute both producers (the
# second's write semantically "wins" but only for the overlap region).
#
# We deliberately pick views where the OVERLAP is half the buffer; the
# upper half of `fused` ends up determined by B since both writers touch
# it but B is emitted second in topo order (deterministically the later
# producer's writes are visible — but the exact race resolution depends
# on scheduling). We only assert that:
#   (a) the compile succeeds,
#   (b) the consumer reads SOMETHING and depends on both producers,
# without comparing the overlap region pointwise.
# ---------------------------------------------------------------------------
def test_overlapping_write_views_compile():
    print("[test_overlapping_write_views_compile]")
    device = "cuda"
    dtype = torch.bfloat16
    B, H = 16, 2048
    half = B // 2

    xa = torch.randn(B - 4, H, dtype=dtype, device=device)  # rows [0, B-4)
    xb = torch.randn(B - 4, H, dtype=dtype, device=device)  # rows [4, B)
    wa = torch.randn(H, dtype=dtype, device=device)
    wb = torch.randn(H, dtype=dtype, device=device)
    wc = torch.randn(H, dtype=dtype, device=device)
    fused = torch.zeros(B, H, dtype=dtype, device=device)
    out = torch.zeros(B, H, dtype=dtype, device=device)

    pk = _build_pk()
    xa_dt = pk.attach_input(xa, name="xa")
    xb_dt = pk.attach_input(xb, name="xb")
    wa_dt = pk.attach_input(wa, name="wa")
    wb_dt = pk.attach_input(wb, name="wb")
    wc_dt = pk.attach_input(wc, name="wc")
    fused_dt = pk.attach_input(fused, name="fused")
    out_dt = pk.attach_input(out, name="out")

    fused_top = pk.narrow(fused_dt, 0, 0, B - 4)
    fused_bot = pk.narrow(fused_dt, 0, 4, B - 4)

    pk.rmsnorm_layer(input=xa_dt, weight=wa_dt, output=fused_top,
                     grid_dim=(B - 4, 1, 1), block_dim=_block_dim_for(pk))
    pk.rmsnorm_layer(input=xb_dt, weight=wb_dt, output=fused_bot,
                     grid_dim=(B - 4, 1, 1), block_dim=_block_dim_for(pk))
    pk.rmsnorm_layer(input=fused_dt, weight=wc_dt, output=out_dt,
                     grid_dim=(B, 1, 1), block_dim=_block_dim_for(pk))

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()
    print("  overlapping write-views compiled and ran (race region unverified)")
    pk.finalize()


# ---------------------------------------------------------------------------
# Test 7. Outer-dim split, then writer/reader on disjoint slabs of the
# same parent — verifying the window-overlap pruner correctly drops the
# spurious edge.
#
# Storage T (2B, H) split along dim 0 into [T_top, T_bot]. Outer-dim
# slices are CONTIGUOUS in byte range — A writes [0, B*H*2) bytes, B
# reads [B*H*2, 2B*H*2) bytes — so byte-interval overlap is empty and
# the analyzer prunes the would-be A→B edge.
#
# NOTE on a Phase 1 conservativeness: split along an INNER dim produces
# slices whose byte ranges INTERLEAVE (they each take strided columns
# from every row). The Phase 1 byte-range overlap check then reports
# overlap, and the analyzer inserts a barrier edge even though the
# slices are semantically disjoint. This is a correctness-safe (more
# conservative than necessary) outcome; a Phase 2 stride-aware analysis
# could remove this.
# ---------------------------------------------------------------------------
def test_disjoint_sibling_write_and_read():
    print("[test_disjoint_sibling_write_and_read]")
    device = "cuda"
    dtype = torch.bfloat16
    B, H = 16, 2048

    x = torch.randn(B, H, dtype=dtype, device=device)
    seed_bot = torch.randn(B, H, dtype=dtype, device=device)
    wa = torch.randn(H, dtype=dtype, device=device)
    wb = torch.randn(H, dtype=dtype, device=device)
    big = torch.zeros(2 * B, H, dtype=dtype, device=device)
    # Pre-fill the bottom slab so the disjoint reader has data:
    big[B:, :] = seed_bot
    out = torch.zeros(B, H, dtype=dtype, device=device)

    pk = _build_pk()
    x_dt = pk.attach_input(x, name="x")
    wa_dt = pk.attach_input(wa, name="wa")
    wb_dt = pk.attach_input(wb, name="wb")
    big_dt = pk.attach_input(big, name="big")
    out_dt = pk.attach_input(out, name="out")

    splits = pk.split(big_dt, 2, 0)  # outer-dim split → contiguous slabs
    T_top, T_bot = splits[0], splits[1]
    # A writes T_top (no upstream view-input — x_dt is a graph input).
    pk.rmsnorm_layer(input=x_dt, weight=wa_dt, output=T_top,
                     grid_dim=(B, 1, 1), block_dim=_block_dim_for(pk))
    # B reads T_bot (a view of the SAME parent that A writes, but a
    # disjoint contiguous window). The byte-range overlap check returns
    # false; the analyzer must NOT create an A→B edge.
    pk.rmsnorm_layer(input=T_bot, weight=wb_dt, output=out_dt,
                     grid_dim=(B, 1, 1), block_dim=_block_dim_for(pk))

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    ref = torch_rmsnorm(seed_bot, wb)
    _assert_close(out, ref, 0.05, "disjoint_sibling")
    pk.finalize()


# ---------------------------------------------------------------------------
# Test 8. Inner-dim split: documents the Phase 1 conservativeness
# discussed in Test 7. Same shape (B, 2H), split along dim 1, A writes
# T_left, B reads T_right. The slices DO NOT overlap semantically but
# the byte-range check says they do → 1 BARRIER edge is created
# (correctness-safe; the kernel still produces correct output because
# A's write completes before B reads via the barrier).
# ---------------------------------------------------------------------------
def test_inner_dim_split_overlap_conservativeness():
    print("[test_inner_dim_split_overlap_conservativeness]")
    device = "cuda"
    dtype = torch.bfloat16
    B, H = 16, 2048

    x = torch.randn(B, H, dtype=dtype, device=device)
    seed_right = torch.randn(B, H, dtype=dtype, device=device)
    wa = torch.randn(H, dtype=dtype, device=device)
    wb = torch.randn(H, dtype=dtype, device=device)
    big = torch.zeros(B, 2 * H, dtype=dtype, device=device)
    big[:, H:] = seed_right
    out = torch.zeros(B, H, dtype=dtype, device=device)

    pk = _build_pk()
    x_dt = pk.attach_input(x, name="x")
    wa_dt = pk.attach_input(wa, name="wa")
    wb_dt = pk.attach_input(wb, name="wb")
    big_dt = pk.attach_input(big, name="big")
    out_dt = pk.attach_input(out, name="out")

    splits = pk.split(big_dt, 2, 1)  # inner-dim split
    T_left, T_right = splits[0], splits[1]
    pk.rmsnorm_layer(input=x_dt, weight=wa_dt, output=T_left,
                     grid_dim=(B, 1, 1), block_dim=_block_dim_for(pk))
    pk.rmsnorm_layer(input=T_right, weight=wb_dt, output=out_dt,
                     grid_dim=(B, 1, 1), block_dim=_block_dim_for(pk))

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    # Correctness still holds — B sees the pre-filled right half.
    ref = torch_rmsnorm(seed_right, wb)
    _assert_close(out, ref, 0.05, "inner_split_conservativeness")
    pk.finalize()


# ---------------------------------------------------------------------------
# Test 9. Producer writes a view; consumer reads the SAME view window.
#
#    A writes V_write = T.narrow(0, 0, half)
#    B reads  V_read  = T.narrow(0, 0, half)   (distinct DTensor, identical
#                                               base_guid / offset / shape / stride)
#
# Even though make_view() gives V_write and V_read different guids, they are
# the SAME view window, so producer and consumer tile it identically. The
# GCD per-tile events synchronize the A→B edge correctly, so it must be a
# fine-grained (NON-barrier) edge — NOT a coarse single-event barrier. This
# is the "write into a slice, then read that same slice" pattern.
# ---------------------------------------------------------------------------
def test_same_view_write_read():
    print("[test_same_view_write_read]")
    device = "cuda"
    dtype = torch.bfloat16
    B, H = 16, 2048
    half = B // 2

    x = torch.randn(half, H, dtype=dtype, device=device)
    wa = torch.randn(H, dtype=dtype, device=device)
    wb = torch.randn(H, dtype=dtype, device=device)
    T = torch.zeros(B, H, dtype=dtype, device=device)
    out = torch.zeros(half, H, dtype=dtype, device=device)

    pk = _build_pk()
    x_dt = pk.attach_input(x, name="x")
    wa_dt = pk.attach_input(wa, name="wa")
    wb_dt = pk.attach_input(wb, name="wb")
    T_dt = pk.attach_input(T, name="T")
    out_dt = pk.attach_input(out, name="out")

    V_write = pk.narrow(T_dt, 0, 0, half)
    V_read = pk.narrow(T_dt, 0, 0, half)  # distinct object, identical window
    pk.rmsnorm_layer(input=x_dt, weight=wa_dt, output=V_write,
                     grid_dim=(half, 1, 1), block_dim=_block_dim_for(pk))
    pk.rmsnorm_layer(input=V_read, weight=wb_dt, output=out_dt,
                     grid_dim=(half, 1, 1), block_dim=_block_dim_for(pk))

    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)
    pk()
    torch.cuda.synchronize()

    ref = torch_rmsnorm(torch_rmsnorm(x, wa), wb)
    _assert_close(out, ref, 0.07, "same_view_write_read")
    pk.finalize()


# ---------------------------------------------------------------------------
# Subprocess wrappers for AnnotatedGraph-dump assertions.
# ---------------------------------------------------------------------------
def _read_annotated_dump(stderr_text):
    m = re.search(r"AnnotatedGraph: .*?ordered_layers: \[[^\]]*\]\n",
                  stderr_text, re.DOTALL)
    return m.group(0) if m else None


def _run_subprocess(focus):
    env = dict(os.environ)
    env["MIRAGE_DUMP_ANNOTATED_GRAPH"] = "1"
    env["MPK_VIEW_CORNER_FOCUS"] = focus
    res = subprocess.run([sys.executable, __file__],
                         env=env, capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        print("  subprocess stdout tail:", res.stdout[-800:])
        print("  subprocess stderr tail:", res.stderr[-800:])
        raise RuntimeError(f"subprocess for {focus} failed (rc={res.returncode})")
    dump = _read_annotated_dump(res.stderr)
    if dump is None:
        print("  no dump in stderr (tail):", res.stderr[-800:])
        raise RuntimeError("AnnotatedGraph dump not captured")
    return dump


def _count_barrier(dump):
    return len(re.findall(r"\[BARRIER\]", dump))


def _count_edges(dump):
    return len(re.findall(r"^\s+\[\d+\] \d+:", dump, re.MULTILINE))


def _count_fork_groups(dump):
    m = re.search(r"(\d+) fork groups", dump)
    return int(m.group(1)) if m else 0


def _count_join_groups(dump):
    m = re.search(r"(\d+) join groups", dump)
    return int(m.group(1)) if m else 0


def test_dep_analysis_corner_cases():
    """Verify edge counts / fork/join group counts for the corner-case
    graphs above by re-running each in a subprocess under
    MIRAGE_DUMP_ANNOTATED_GRAPH=1."""
    print("[test_dep_analysis_corner_cases]")

    # 1. Fork with two sibling read-views: P has 2 distinct consumers, so
    #    1 fork group; 2 edges, both BARRIER.
    d = _run_subprocess("test_fork_with_sibling_read_views")
    nb = _count_barrier(d); ne = _count_edges(d); nf = _count_fork_groups(d)
    print(f"  fork_sibling: edges={ne} barrier={nb} fork_groups={nf}")
    if ne != 2 or nb != 2 or nf != 1:
        print("  FAIL: expected edges=2 barrier=2 fork_groups=1")
        print(d); sys.exit(1)

    # 2. Mixed fork: 1 view + 1 full edge. Both should still form a fork
    #    bundle with 2 edges, 1 barrier (the view edge).
    d = _run_subprocess("test_mixed_fork_view_and_full")
    nb = _count_barrier(d); ne = _count_edges(d); nf = _count_fork_groups(d)
    print(f"  mixed_fork: edges={ne} barrier={nb} fork_groups={nf}")
    if ne != 2 or nb != 1 or nf != 1:
        print("  FAIL: expected edges=2 barrier=1 fork_groups=1")
        print(d); sys.exit(1)

    # 3. Disjoint writers/readers: A→C, B→D (2 edges). Each (writer, reader)
    #    pair touches the SAME view window, so both edges are fine-grained
    #    (0 BARRIER). No spurious cross edges (A→D or B→C) from the window
    #    overlap pruner.
    d = _run_subprocess("test_disjoint_writers_disjoint_readers")
    nb = _count_barrier(d); ne = _count_edges(d)
    print(f"  disjoint: edges={ne} barrier={nb}")
    if ne != 2 or nb != 0:
        print("  FAIL: expected edges=2 barrier=0 (same-view fine-grained; "
              "window overlap pruning keeps cross edges out)")
        print(d); sys.exit(1)

    # 4. Cascade A→B→C with views at each step: 2 chain edges, both
    #    BARRIER (each consumer reads through a view).
    d = _run_subprocess("test_cascading_view_chain")
    nb = _count_barrier(d); ne = _count_edges(d)
    print(f"  cascade: edges={ne} barrier={nb}")
    if ne != 2 or nb != 2:
        print("  FAIL: expected edges=2 barrier=2")
        print(d); sys.exit(1)

    # 5. Multi-level view: chain of one view-of-view in B's input. 1 edge
    #    A→B (barrier).
    d = _run_subprocess("test_multi_level_view_codegen")
    nb = _count_barrier(d); ne = _count_edges(d)
    print(f"  multi_level: edges={ne} barrier={nb}")
    if ne != 1 or nb != 1:
        print("  FAIL: expected edges=1 barrier=1")
        print(d); sys.exit(1)

    # 6. Overlapping write-views: A and B each write to overlapping
    #    halves of `fused`; C reads fused (whole). Both A and B's
    #    windows overlap C's full window → 2 BARRIER edges into C.
    #    A and B themselves have no read of each other → no WAW edges.
    d = _run_subprocess("test_overlapping_write_views_compile")
    nb = _count_barrier(d); ne = _count_edges(d); nj = _count_join_groups(d)
    print(f"  overlap_write: edges={ne} barrier={nb} join_groups={nj}")
    if ne != 2 or nb != 2 or nj != 1:
        print("  FAIL: expected edges=2 barrier=2 join_groups=1 (C joins A+B)")
        print(d); sys.exit(1)

    # 7. Disjoint outer-dim sibling write+read: byte-range disjoint → 0
    #    inter-layer edges.
    d = _run_subprocess("test_disjoint_sibling_write_and_read")
    ne = _count_edges(d)
    print(f"  disjoint_sibling: edges={ne}")
    if ne != 0:
        print("  FAIL: expected 0 edges (outer-dim disjoint windows)")
        print(d); sys.exit(1)

    # 8. Inner-dim sibling write+read: byte-ranges interleave → Phase 1
    #    conservatively reports overlap → 1 BARRIER edge. Documents the
    #    correctness-safe limitation.
    d = _run_subprocess("test_inner_dim_split_overlap_conservativeness")
    ne = _count_edges(d); nb = _count_barrier(d)
    print(f"  inner_split: edges={ne} barrier={nb}")
    if ne != 1 or nb != 1:
        print("  FAIL: expected 1 conservative barrier edge for inner split")
        print(d); sys.exit(1)

    # 9. Same-view write→read: A writes view V, B reads the IDENTICAL view
    #    window (distinct narrow() call; same base_guid/offset/shape/stride).
    #    Per-tile correspondence is valid → fine-grained: 1 edge, 0 BARRIER.
    d = _run_subprocess("test_same_view_write_read")
    ne = _count_edges(d); nb = _count_barrier(d)
    print(f"  same_view: edges={ne} barrier={nb}")
    if ne != 1 or nb != 0:
        print("  FAIL: expected edges=1 barrier=0 (same view → fine-grained)")
        print(d); sys.exit(1)

    print("  OK")


def main():
    focus = os.environ.get("MPK_VIEW_CORNER_FOCUS")
    if focus is not None:
        # Subprocess path: run only the requested test so the dump in
        # stderr corresponds to one graph in isolation.
        globals()[focus]()
        return

    test_fork_with_sibling_read_views()
    test_mixed_fork_view_and_full()
    test_disjoint_writers_disjoint_readers()
    test_cascading_view_chain()
    test_multi_level_view_codegen()
    test_overlapping_write_views_compile()
    test_disjoint_sibling_write_and_read()
    test_inner_dim_split_overlap_conservativeness()
    test_same_view_write_read()
    test_dep_analysis_corner_cases()
    print("ALL VIEW CORNER-CASE TESTS PASSED")


if __name__ == "__main__":
    main()
