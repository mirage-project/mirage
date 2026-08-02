"""Regression net for the Blackwell (sm_100) threadblock codegen backend.

Before these fixes, *no* graph containing a `KNCustomizedOp` produced a working
kernel on sm_90/sm_100 -- the backend emitted references to identifiers it never
declared, launched with an illegal cluster/grid combination, and mis-sized the
cooperative kernels' thread counts.

What is covered here is the case the fused-MPK-task work depends on: a
single-tile custom threadblock op (`grid=(1,1,1)`, `forloop_range=1`), which is
the shape an MPK task body has -- one CTA processing one tile from pointers the
task descriptor supplies.

The pipelined/TMA and matmul paths are still broken (guid-suffixed `tiled_mma`
symbols and CUTLASS 4.2.1 signature mismatches); they are marked xfail so the
suite records the boundary instead of hiding it.

Run:
    PYTHONPATH=. pytest tests/experiments/test_blackwell_codegen.py -v
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

torch = pytest.importorskip("torch", reason="PyTorch is required")
import mirage as mi  # noqa: E402


def _skip_reason():
    if not torch.cuda.is_available():
        return "CUDA is not available"
    if shutil.which("nvcc") is None:
        return "nvcc not found; muGraph compilation needs the CUDA toolkit"
    return None


requires_gpu = pytest.mark.skipif(
    _skip_reason() is not None, reason=_skip_reason() or ""
)

TILE_M, TILE_N = 8, 64


def _build_silu_mul(grid, block, forloop_range, tile=(TILE_M, TILE_N), forloop_dim=-1):
    """silu(a) * b over one tile -- the shape of a fused MPK task body.

    ``forloop_dim=1`` makes the inputs *pipelined*, which is what turns on the
    TMA producer/consumer warp-specialized path.
    """
    m, n = tile
    imap = (-1, -1, -1) if grid == (1, 1, 1) else (1, -1, -1)
    g = mi.new_kernel_graph()
    a = g.new_input(dims=(m, n), dtype=mi.bfloat16)
    b = g.new_input(dims=(m, n), dtype=mi.bfloat16)
    tb = mi.new_threadblock_graph(
        grid_dim=grid, block_dim=block, forloop_range=forloop_range, reduction_dimx=64
    )
    ta = tb.new_input(dtensor=a, input_map=imap, forloop_dim=forloop_dim)
    tbb = tb.new_input(dtensor=b, input_map=imap, forloop_dim=forloop_dim)
    # forloop_accum is required: Graph::create_customized_op segfaults when a
    # threadblock graph's output does not pass through an accumulator.
    out = tb.mul(tb.silu(tb.forloop_accum(ta, None)), tb.forloop_accum(tbb, None))
    tb.new_output(stensor=out, output_map=imap)
    outs = g.customized([a, b], tb)
    g.mark_output(outs[0])
    return g


def _compile_and_run(graph, target_cc, num_warp_groups, pipeline_stages=2):
    ins = [
        torch.randn(TILE_M, TILE_N, dtype=torch.bfloat16, device="cuda")
        for _ in range(2)
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        result = graph.compile(
            inputs=ins,
            target_cc=target_cc,
            num_warp_groups=num_warp_groups,
            pipeline_stages=pipeline_stages,
        )
    assert result is not None and graph._valid_cuda_kernels, graph.get_error_message()
    out = graph(inputs=ins)
    torch.cuda.synchronize()
    ref = torch.nn.functional.silu(ins[0].float()) * ins[1].float()
    return (out[0].float() - ref).abs().max().item()


class TestTranspilerDiagnostics:
    """Phase 0: a rejected graph must say why, without needing a GPU."""

    def test_generate_cuda_program_reports_error_type(self):
        g = _build_silu_mul((32, 1, 1), (128, 1, 1), forloop_range=4, tile=(8, 2048))
        strides = [
            g.cygraph.get_input_dtensor_shape_and_stride(t)[1]
            for t in g.cygraph.get_input_dtensors()
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            # block_dim 128 with 2 warp groups is an illegal Blackwell config
            res = mi.generate_cuda_program(
                g.cygraph, target_cc=100, input_strides=strides,
                num_warp_groups=2, pipeline_stages=2,
            )
        assert "error_type" in res
        assert res["error_type"] != 0
        assert res["code"] == ""

    def test_transpiler_config_defaults_are_deterministic(self):
        """Omitting the warp-group args must not read uninitialized memory."""
        g = _build_silu_mul((32, 1, 1), (128, 1, 1), forloop_range=4, tile=(8, 2048))
        strides = [
            g.cygraph.get_input_dtensor_shape_and_stride(t)[1]
            for t in g.cygraph.get_input_dtensors()
        ]
        seen = set()
        for _ in range(3):
            with contextlib.redirect_stdout(io.StringIO()):
                res = mi.generate_cuda_program(
                    g.cygraph, target_cc=100, input_strides=strides
                )
            seen.add(res["error_type"])
        assert len(seen) == 1, f"non-deterministic transpiler config: {seen}"


@requires_gpu
class TestSingleTileCustomOp:
    """The shape a fused MPK task body has: one CTA, one tile."""

    @pytest.mark.parametrize("block", [128, 256, 384])
    def test_blackwell_compiles_and_is_correct(self, block):
        num_warp_groups = block // 128
        g = _build_silu_mul((1, 1, 1), (block, 1, 1), forloop_range=1)
        max_abs = _compile_and_run(g, 100, num_warp_groups)
        # bf16 elementwise rounding only
        assert max_abs < 0.1, f"block={block}: max abs error {max_abs}"

    def test_ampere_still_works(self):
        """Guards against regressing the one backend that already worked."""
        g = _build_silu_mul((1, 1, 1), (128, 1, 1), forloop_range=1)
        with contextlib.redirect_stdout(io.StringIO()):
            ins = [
                torch.randn(TILE_M, TILE_N, dtype=torch.bfloat16, device="cuda")
                for _ in range(2)
            ]
            res = g.compile(inputs=ins, target_cc=80, num_warp_groups=1,
                            pipeline_stages=2)
        assert res is not None and g._valid_cuda_kernels

    def test_unpipelined_multi_tile_grid(self):
        """A multi-CTA grid must get a cluster that divides it.

        cluster_dim was hardcoded {4,4,1}; for grids not divisible by it the
        cluster launch failed silently and the output was never written, so this
        asserts on the numerics rather than on the (unexported) cluster_dim.
        """
        g = _build_silu_mul((32, 1, 1), (128, 1, 1), forloop_range=1, tile=(8, 2048))
        ins = [
            torch.randn(8, 2048, dtype=torch.bfloat16, device="cuda") for _ in range(2)
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            res = g.compile(inputs=ins, target_cc=100, num_warp_groups=1,
                            pipeline_stages=2)
        assert res is not None and g._valid_cuda_kernels, g.get_error_message()
        out = g(inputs=ins)
        torch.cuda.synchronize()
        ref = torch.nn.functional.silu(ins[0].float()) * ins[1].float()
        assert (out[0].float() - ref).abs().max().item() < 0.1


@requires_gpu
class TestKnownBroken:
    """Boundary of what works today -- see DESIGN.md for the remaining defects."""

    @pytest.mark.xfail(
        reason="pipelined/TMA path: input atoms reference guid-suffixed "
               "tiled_mma/mma_tiler that the TB backend never declares (A2/A3)",
        strict=True,
    )
    def test_pipelined_elementwise(self):
        g = _build_silu_mul(
            (32, 1, 1), (256, 1, 1), forloop_range=4, tile=(8, 2048), forloop_dim=1
        )
        ins = [
            torch.randn(8, 2048, dtype=torch.bfloat16, device="cuda") for _ in range(2)
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            res = g.compile(inputs=ins, target_cc=100, num_warp_groups=2,
                            pipeline_stages=2)
        assert res is not None and g._valid_cuda_kernels

    @pytest.mark.xfail(
        reason="Hopper TB backend has the same class of defects as Blackwell "
               "had; not yet fixed and no sm_90 hardware here to validate on",
        strict=True,
    )
    def test_hopper_single_tile(self):
        g = _build_silu_mul((1, 1, 1), (128, 1, 1), forloop_range=1)
        ins = [
            torch.randn(TILE_M, TILE_N, dtype=torch.bfloat16, device="cuda")
            for _ in range(2)
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            res = g.compile(inputs=ins, target_cc=90, num_warp_groups=1,
                            pipeline_stages=2)
        assert res is not None and g._valid_cuda_kernels


class TestBlackwellMatmul:
    """A generated 1-SM UMMA matmul must agree with PyTorch.

    The matmul now compiles and runs (it previously emitted no MMA at all and
    deadlocked), and its reduction is correct -- all-ones operands give exactly
    K in every output element. What remains is a layout defect in the A operand.

    The G->S copy and the MMA no longer disagree about the swizzle: the copy is
    now pointed at Blackwell_Matmul::SmemLayout{A,B}_MMA_*, which is derived from
    the same DstPipeLayout the UMMA reads, and the operand base is 1024B aligned
    so the swizzle pattern is anchored correctly. Both were real defects and both
    are fixed. B (MN-major) is correct as a result.

    The layout defect that made this return right values at wrong positions was
    get_stensor_layout() hardcoding Swizzle<3,3,4>. The UMMA reads its operands
    through a 128B swizzle over 16B chunks, which in raw element units is
    Swizzle<3,3,3> -- one bit of chunk granularity apart. Sweeping <B,M,S>
    against a PyTorch reference, only <3,3,3> matches exactly, and that is what
    get_threadblock_swizzle_plan_blackwell already computed; its result was just
    being discarded. Operands also need a 1024B base (the swizzle period), which
    is why the alignment is raised for blackwell_arch.
    """

    def _build_matmul(self, m, k, n):
        g = mi.new_kernel_graph()
        X = g.new_input(dims=(m, k), dtype=mi.bfloat16)
        W = g.new_input(dims=(k, n), dtype=mi.bfloat16)
        tb = mi.new_threadblock_graph(
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
            forloop_range=1, reduction_dimx=64,
        )
        tX = tb.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=-1)
        tW = tb.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=-1)
        tb.new_output(
            stensor=tb.forloop_accum(tb.matmul(tX, tW), None),
            output_map=(1, -1, -1),
        )
        O = g.customized([X, W], tb)
        g.mark_output(O[0])
        return g

    def test_matmul_reduction_length(self):
        """All-ones operands: every element must equal K exactly."""
        m, k, n = 128, 64, 64
        g = self._build_matmul(m, k, n)
        a = torch.ones(m, k, dtype=torch.bfloat16, device="cuda")
        b = torch.ones(k, n, dtype=torch.bfloat16, device="cuda")
        with contextlib.redirect_stdout(io.StringIO()):
            res = g.compile(inputs=[a, b], target_cc=100, num_warp_groups=1,
                            pipeline_stages=2)
        assert res is not None and g._valid_cuda_kernels, g.get_error_message()
        out = g(inputs=[a, b])[0].float()
        torch.cuda.synchronize()
        assert out.unique().tolist() == [float(k)]

    def test_matmul_b_operand_layout(self):
        """A=I selects B's rows: verifies the MN-major operand end to end."""
        m, k, n = 128, 64, 64
        g = self._build_matmul(m, k, n)
        a = torch.eye(m, k, dtype=torch.bfloat16, device="cuda")
        b = (torch.arange(k * n, device="cuda").reshape(k, n) % 13).to(
            torch.bfloat16)
        with contextlib.redirect_stdout(io.StringIO()):
            res = g.compile(inputs=[a, b], target_cc=100, num_warp_groups=1,
                            pipeline_stages=2)
        assert res is not None and g._valid_cuda_kernels, g.get_error_message()
        out = g(inputs=[a, b])[0].float()
        torch.cuda.synchronize()
        assert torch.equal(out[0], b[0].float())

    # M in {8, 16} exercises swapAB: 1-SM tcgen05 needs an M-tile of 64/128, so a
    # decode-shaped token count is moved into N by computing C^T = B^T * A^T.
    @pytest.mark.parametrize(
        "m,k",
        [(128, 16), (128, 32), (128, 64), (64, 64), (8, 64), (16, 64)],
    )
    def test_matmul_matches_torch(self, m, k):
        n = 64
        g = self._build_matmul(m, k, n)
        a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
        with contextlib.redirect_stdout(io.StringIO()):
            res = g.compile(inputs=[a, b], target_cc=100, num_warp_groups=1,
                            pipeline_stages=2)
        assert res is not None and g._valid_cuda_kernels, g.get_error_message()
        out = g(inputs=[a, b])[0].float()
        torch.cuda.synchronize()
        ref = a.float() @ b.float()
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        assert rel < 0.02, f"relative error {rel}"

    @pytest.mark.parametrize("m,k,n", [(128, 128, 64), (128, 64, 128)])
    def test_oversized_operand_pitch_is_rejected(self, m, k, n):
        """A/B row pitch >128B must fail loudly, never silently miscompute.

        This covers the NON-pipelined path only (_build_matmul uses
        forloop_range=1). There the G->S copy is InputChunkedSyncCopy, which
        indexes linearly through the transpiler's dense-stride + XOR-swizzle
        model, while CUTLASS panel-tiles wider operands (tile_to_shape at K=128
        puts the second K block at +8192 elements, not at a row pitch of 128).
        No dense-stride layout can express that, and these shapes measured ~1.6
        relative error -- silently wrong output -- before the guard existed.

        A PIPELINED operand has no such limit: TMA writes and the UMMA reads
        through the same CUTLASS DstPipeLayout, so both panel-tile identically.
        See test_matmul_wide_tile_pipelined.
        """
        g = self._build_matmul(m, k, n)
        a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
        with contextlib.redirect_stdout(io.StringIO()):
            g.compile(inputs=[a, b], target_cc=100, num_warp_groups=1,
                      pipeline_stages=2)
        assert not g._valid_cuda_kernels
        assert "layout error" in g.get_error_message()

    # Tiles wider than the 128B pitch the non-pipelined path is limited to:
    # N=128 makes the B tile 256B wide, and K=256 over 2 iterations makes the A
    # tile 128 elements (256B) along K. Both are correct here because a
    # pipelined operand is written by TMA and read by the UMMA through the same
    # CUTLASS DstPipeLayout, derived on both sides from sm100_smem_selector.
    @pytest.mark.parametrize("m,k,n,r", [(128, 128, 128, 2), (128, 256, 64, 2),
                                         (128, 256, 128, 4), (8, 256, 128, 4)])
    def test_matmul_wide_tile_pipelined(self, m, k, n, r):
        g = self._build_matmul_kloop(m, k, n, r)
        a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
        with contextlib.redirect_stdout(io.StringIO()):
            g.compile(inputs=[a, b], target_cc=100, num_warp_groups=2,
                      pipeline_stages=2)
        assert g._valid_cuda_kernels, g.get_error_message()
        out = g(inputs=[a, b])[0].float()
        torch.cuda.synchronize()
        ref = a.float() @ b.float()
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        assert rel < 0.02, f"relative error {rel}"

    def _build_matmul_kloop(self, m, k, n, r):
        g = mi.new_kernel_graph()
        X = g.new_input(dims=(m, k), dtype=mi.bfloat16)
        W = g.new_input(dims=(k, n), dtype=mi.bfloat16)
        tb = mi.new_threadblock_graph(
            grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
            forloop_range=r, reduction_dimx=64,
        )
        tX = tb.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
        tW = tb.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
        tb.new_output(
            stensor=tb.forloop_accum(tb.matmul(tX, tW), None),
            output_map=(1, -1, -1),
        )
        O = g.customized([X, W], tb)
        g.mark_output(O[0])
        return g

    # A CHAINED matmul: Q@K^T -> exp -> @V, the attention core. The first
    # matmul's result is consumed by the second INSIDE the loop, which needs
    # (a) a per-matmul count-1 barrier waited before the read, (b) the fused exp
    # actually applied in write_tC_to_sC (it was accepted and silently dropped),
    # and (c) the intermediate pinned to the gmem-operand orientation by the
    # layout solver (left free it picked the other dim; right values, wrong
    # positions). No softmax denominator on purpose -- inputs are scaled so exp
    # stays in range; online softmax is separate work.
    def test_chained_matmul_exp_matmul(self):
        NH = HD = S = 64
        g = mi.new_kernel_graph()
        Q = g.new_input(dims=(NH, HD), dtype=mi.bfloat16)
        KT = g.new_input(dims=(HD, S), dtype=mi.bfloat16)
        V = g.new_input(dims=(S, HD), dtype=mi.bfloat16)
        tb = mi.new_threadblock_graph(grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
                                      forloop_range=1, reduction_dimx=64)
        tQ = tb.new_input(dtensor=Q, input_map=(-1, -1, -1), forloop_dim=-1)
        tK = tb.new_input(dtensor=KT, input_map=(-1, -1, -1), forloop_dim=-1)
        tV = tb.new_input(dtensor=V, input_map=(-1, -1, -1), forloop_dim=-1)
        E = tb.exp(tb.matmul(tQ, tK))
        acc = tb.forloop_accum(tb.matmul(E, tV), None)
        tb.new_output(stensor=acc, output_map=(-1, -1, -1))
        out_op = g.customized([Q, KT, V], tb)
        g.mark_output(out_op[0])
        q = torch.randn(NH, HD, dtype=torch.bfloat16, device="cuda") * 0.1
        k = torch.randn(HD, S, dtype=torch.bfloat16, device="cuda") * 0.1
        v = torch.randn(S, HD, dtype=torch.bfloat16, device="cuda")
        with contextlib.redirect_stdout(io.StringIO()):
            g.compile(inputs=[q, k, v], target_cc=100, num_warp_groups=1,
                      pipeline_stages=2)
        assert g._valid_cuda_kernels, g.get_error_message()
        out = g(inputs=[q, k, v])[0].float()
        torch.cuda.synchronize()
        ref = torch.exp(q.float() @ k.float()) @ v.float()
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        assert rel < 0.02, f"chained matmul rel {rel}"

    def test_chained_matmul_k_loop(self):
        """K-LOOPED chained matmul: Q@K^T -> exp -> @V with KV tiled over the
        loop -- the flash-attention iteration. The last bug here was the input
        atom advancing local_tile along Tiles_K while attention tiles N (with
        one K_mma tile it refetched K0 into every stage; an E-accumulator
        probe read exactly 2*E0). N_LOOP in InputTMAAsyncCopy_Blackwell fixes
        the free coordinate mode."""
        NH = HD = 64
        S = 128
        g = mi.new_kernel_graph()
        Q = g.new_input(dims=(NH, HD), dtype=mi.bfloat16)
        KT = g.new_input(dims=(HD, S), dtype=mi.bfloat16)
        V = g.new_input(dims=(S, HD), dtype=mi.bfloat16)
        # 256 threads: num_warp_groups=2 below needs block_dim = 2 * 128.
        tb = mi.new_threadblock_graph(grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
                                      forloop_range=2, reduction_dimx=64)
        tQ = tb.new_input(dtensor=Q, input_map=(-1, -1, -1), forloop_dim=-1)
        tK = tb.new_input(dtensor=KT, input_map=(-1, -1, -1), forloop_dim=1)
        tV = tb.new_input(dtensor=V, input_map=(-1, -1, -1), forloop_dim=0)
        E = tb.exp(tb.matmul(tQ, tK))
        acc = tb.forloop_accum(tb.matmul(E, tV), None)
        tb.new_output(stensor=acc, output_map=(-1, -1, -1))
        out_op = g.customized([Q, KT, V], tb)
        g.mark_output(out_op[0])
        q = torch.randn(NH, HD, dtype=torch.bfloat16, device="cuda") * 0.1
        k = torch.randn(HD, S, dtype=torch.bfloat16, device="cuda") * 0.1
        v = torch.randn(S, HD, dtype=torch.bfloat16, device="cuda")
        with contextlib.redirect_stdout(io.StringIO()):
            # num_warp_groups=2: forloop_range>1 needs a consumer warp group.
            g.compile(inputs=[q, k, v], target_cc=100, num_warp_groups=2,
                      pipeline_stages=2)
        assert g._valid_cuda_kernels, g.get_error_message()
        out = g(inputs=[q, k, v])[0].float()
        torch.cuda.synchronize()
        ref = torch.exp(q.float() @ k.float()) @ v.float()
        rel = ((out - ref).abs().max() / ref.abs().max()).item()
        assert rel < 0.02, f"K-loop chained rel {rel}"

    def test_matmul_identity_is_exact(self):
        """I @ I must be exactly I: catches any operand-layout permutation."""
        m, k, n = 128, 64, 64
        g = self._build_matmul(m, k, n)
        a = torch.eye(m, k, dtype=torch.bfloat16, device="cuda")
        b = torch.eye(k, n, dtype=torch.bfloat16, device="cuda")
        with contextlib.redirect_stdout(io.StringIO()):
            res = g.compile(inputs=[a, b], target_cc=100, num_warp_groups=1,
                            pipeline_stages=2)
        assert res is not None and g._valid_cuda_kernels, g.get_error_message()
        out = g(inputs=[a, b])[0].float()
        torch.cuda.synchronize()
        assert torch.equal(out, (a.float() @ b.float()))


# The K-loop must run in a subprocess with a hard timeout. Its failure mode is a
# deadlock, not an exception: the A-operand mbarrier used to expect 2x the bytes
# a single CTA delivers (a 2-SM multicast leftover), so consumer_wait blocked
# forever. In-process that hangs pytest instead of failing it.
_KLOOP_SRC = textwrap.dedent(
    """
    import sys, torch, mirage as mi
    M, K, N, R = {m}, {k}, {n}, {r}
    g = mi.new_kernel_graph()
    X = g.new_input(dims=(M, K), dtype=mi.bfloat16)
    W = g.new_input(dims=(K, N), dtype=mi.bfloat16)
    tb = mi.new_threadblock_graph(grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
                                  forloop_range=R, reduction_dimx=64)
    tX = tb.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW = tb.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
    tb.new_output(stensor=tb.forloop_accum(tb.matmul(tX, tW), None),
                  output_map=(1, -1, -1))
    O = g.customized([X, W], tb); g.mark_output(O[0])
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    g.compile(inputs=[a, b], target_cc=100, num_warp_groups=2, pipeline_stages=2)
    out = g(inputs=[a, b])[0].float(); torch.cuda.synchronize()
    ref = a.float() @ b.float()
    rel = ((out - ref).abs().max() / ref.abs().max()).item()
    print("REL", rel)
    sys.exit(0 if rel < 0.02 else 1)
    """
)


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
# M=8 exercises swapAB together with the pipelined K-loop -- the shape a real
# fused decode segment has. Getting there required every site that asks "which
# stensor is the A operand" to agree; see the role-vs-majorness note in
# InputTMAAsyncCopy_Blackwell (input.h), which is the distinction that makes
# them consistent.
@pytest.mark.parametrize("m,r", [(128, 2), (128, 4), (8, 4), (8, 2)])
def test_matmul_k_loop(m, r):
    """K split across forloop iterations -- what a real fused MLP segment needs.

    num_warp_groups must be >= 2: core.pyx sets num_consumer_wgs =
    num_warp_groups - 1, so 1 yields zero consumer warp groups and the kernel
    deadlocks by construction rather than by any compiler defect.
    """
    src = _KLOOP_SRC.format(m=m, k=64 * r, n=64, r=r)
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    try:
        proc = subprocess.run([sys.executable, "-c", src], timeout=600,
                              capture_output=True, text=True, env=env)
    except subprocess.TimeoutExpired:
        pytest.fail(f"K-loop M={m} forloop_range={r} deadlocked (timed out)")
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr[-2000:]}"
