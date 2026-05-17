"""FP8 dense / swapAB / BMM linear layers — DeepSeek V3 fast paths.

Catalog wrappers around the SM100 FP8 GEMM family in
``python/mirage/mpk/persistent_kernel.py``:

* :meth:`PersistentKernel.linear_fp8_layer`
  → task ``linear_fp8_sm100``
* :meth:`PersistentKernel.linear_fp8_with_residual_layer`
  → task ``linear_fp8_with_residual_sm100``
* :meth:`PersistentKernel.linear_fp8_swapAB_layer`
  → task ``linear_fp8_swapAB_sm100``
* :meth:`PersistentKernel.linear_fp8_swapAB_with_residual_layer`
  → task ``linear_fp8_swapAB_with_residual_sm100``
* :meth:`PersistentKernel.linear_fp8_bmm_sm100_layer`
  → task ``linear_fp8_bmm_sm100``
* :meth:`PersistentKernel.linear_splitk_swapAB_fp8_layer`
  → task ``splitk_linear_fp8_swapAB_sm100`` (Split-K variant)

Tensor / scale layout (shared by every variant)
-----------------------------------------------

* ``input_fp8``    : FP8 E4M3 (stored as ``uint8`` on the device side).
                    Row-major; trailing dim is the K (reduction) axis.
* ``input_scale``  : ``uint32`` UE8M0-packed scales, **four scales per
                    uint32** along K. Shape is
                    ``(*, packed_K)`` where ``packed_K = K // 128``
                    (one logical scale per 128-element K block, after
                    packing). See ``quantize_fp8_layer`` with
                    ``scale_ue8m0=True`` for the producer.
* ``weight_fp8``   : FP8 E4M3. Standard ``nn.Linear`` shape
                    ``(out_features, in_features)`` for the dense
                    variants, ``(H, D_out, D_in)`` for BMM.
* ``weight_scale`` : ``uint32`` UE8M0-packed scales, **stored
                    column-major along M** (i.e., the kernel views the
                    buffer as ``[packed_K, M_aligned]``). Same packing
                    convention as ``input_scale``.

For ``forward()`` we dequantize both operands with
``input_scale.repeat_interleave(128, dim=-1)`` to recover an fp32
scale-per-element view, multiply through the fp8-as-fp32 numerics,
and run a plain ``F.linear`` (or batched matmul for BMM). UE8M0 packing
is recovered by reinterpreting the ``uint32`` storage as four
consecutive ``uint8`` per K-block (each ``uint8`` is a UE8M0 exponent
``s`` interpreted as ``2 ** (s - 127)``). The PyTorch reference here
mirrors what DeepSeek V3's eager scripts do — it's the same scheme the
backing kernels implement (``tcgen05.mma.kind::mxf8f6f4.block_scale``
on Blackwell).

Grid heuristic
--------------

The auto-grid mirrors the legacy DSv3 builder choice in
``python/mirage/mpk/models/deepseek_v3/builder.py``: split the
``out_features`` axis by 128 (the kernel's MMA-M tile) and cap at
``current_pk().num_workers``. Callers that need a non-standard tile
can pass ``grid_dim`` explicitly.

``LinearFP8BMM`` keeps the head-axis as a second grid dim (``grid.y =
num_heads``); each CTA handles one head.

``LinearSplitKFP8SwapAB`` exposes ``grid.x`` (M-shard count) and
``grid.y`` (split-K factor). The kernel uses ``tma_reduce_add_async``
and accumulates into ``output`` — the module's ``accumulate`` flag
matches the pk method's: ``accumulate=False`` prepends a tensor_init
that zeroes ``output`` first.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule


__all__ = ["LinearFP8", "LinearFP8BMM", "LinearSplitKFP8SwapAB"]


# ----------------------------------------------------------------------
# Dequant helpers — only used by the PyTorch reference path.
# ----------------------------------------------------------------------
def _ue8m0_packed_to_fp32(scale_packed: torch.Tensor, k_dim: int) -> torch.Tensor:
    """Decode UE8M0-packed uint32 scales to fp32, expanded along K.

    Each packed ``uint32`` carries 4 consecutive UE8M0 exponents along
    the K-block axis. An UE8M0 byte ``s`` decodes to ``2 ** (s - 127)``.
    The returned tensor has the same leading shape as ``scale_packed``
    with the trailing dim expanded from ``packed_K`` to
    ``packed_K * 4 * 128 = K`` (so an elementwise multiply against the
    dequantized FP8 operand works).

    This helper exists only for the PyTorch reference; the kernel reads
    UE8M0 directly.
    """
    # Interpret the packed uint32 buffer as uint8 along K. Shape:
    # (..., packed_K * 4) — one byte per logical scale block.
    leading = scale_packed.shape[:-1]
    packed_k = scale_packed.shape[-1]
    bytes_view = scale_packed.contiguous().view(torch.uint8).reshape(
        *leading, packed_k * 4
    )
    exp_f32 = bytes_view.to(torch.float32) - 127.0
    scales = torch.pow(torch.tensor(2.0), exp_f32)  # (..., packed_K * 4)
    # Expand each block to 128 K-elements.
    scales = scales.repeat_interleave(128, dim=-1)
    return scales[..., :k_dim]


def _colmajor_weight_scale_for_tma(weight_scale: torch.Tensor) -> torch.Tensor:
    """Repack the row-major ``(M, packed_K)`` weight_scale into a buffer
    whose physical storage is **M-fastest** (the layout the kernel TMA
    descriptor for SFB expects in ``linear_fp8_sm100`` /
    ``linear_fp8_with_residual_sm100``: ``stride[0] == 1``,
    ``stride[1] == aligned_M``).

    See ``include/mirage/persistent_kernel/tma.cuh`` around the
    ``TASK_LINEAR_FP8_SM100`` TMA descriptor build for SFB (param_id=3).

    The catalog keeps ``self.weight_scale`` as a row-major ``nn.Parameter``
    so HF ``load_state_dict`` mapping with the HF M-outer storage works
    unchanged. We build the col-major attached buffer at ``compile()``
    time. Caller is responsible for stashing the result on ``self`` so
    it outlives the kernel run (``attach_input`` only stores a pointer).

    M is always a multiple of 128 in the FP8 layers (MMA-M tile), which
    is already 4-aligned so no padding is needed.
    """
    assert weight_scale.dim() == 2, (
        f"_colmajor_weight_scale_for_tma: expected 2D weight_scale, got "
        f"shape {tuple(weight_scale.shape)}")
    M, packed_K = weight_scale.shape
    # M-fastest storage: physical layout = (packed_K, M) row-major.
    # ``.t().contiguous()`` produces that exactly, and then ``.t()`` again
    # returns a strided view with logical shape (M, packed_K) and strides
    # (1, M) — which is what ``attach_input``'s col-major-2D path accepts.
    return weight_scale.t().contiguous().t()


def _dequant_fp8(
    fp8_bytes: torch.Tensor,
    scales_packed: torch.Tensor,
) -> torch.Tensor:
    """Return an fp32 dequant of ``fp8_bytes`` using UE8M0-packed scales.

    ``fp8_bytes`` may either be a ``torch.float8_e4m3fn`` tensor (cast
    to fp32 with ``.float()``) or a ``uint8`` raw-byte buffer (we
    reinterpret then cast). ``scales_packed`` is UE8M0-packed uint32
    laid out per the module docstring; we expand it along the trailing
    axis to match ``fp8_bytes``'s K dim and multiply elementwise.
    """
    if fp8_bytes.dtype == torch.float8_e4m3fn:
        fp32 = fp8_bytes.float()
    else:
        # Raw uint8 — reinterpret as float8_e4m3fn, then promote.
        fp32 = fp8_bytes.view(torch.float8_e4m3fn).float()
    k_dim = fp32.shape[-1]
    scales = _ue8m0_packed_to_fp32(scales_packed, k_dim).to(fp32.device)
    return fp32 * scales


# ----------------------------------------------------------------------
# LinearFP8 — the four-way dispatch (residual × swap_ab).
# ----------------------------------------------------------------------
class LinearFP8(MPKModule):
    """FP8 dense linear projection (SM100 only).

    Dispatches to one of four backing pk methods based on
    ``(residual, swap_ab)``:

    +-----------+---------+----------------------------------------+
    | residual  | swap_ab | pk method                              |
    +===========+=========+========================================+
    | False     | False   | ``linear_fp8_layer``                   |
    | True      | False   | ``linear_fp8_with_residual_layer``     |
    | False     | True    | ``linear_fp8_swapAB_layer``            |
    | True      | True    | ``linear_fp8_swapAB_with_residual_layer`` |
    +-----------+---------+----------------------------------------+

    The swapAB variant exposes weight as MMA's A operand (transposed
    internally) — it gives better latency on small-batch decode shapes
    (``batch_size <= 16``). The residual variants fold a bf16 ``+ R``
    into the GEMM epilogue.

    Args:
        in_features: K (reduction) axis. Must be a multiple of 128
            (UE8M0 block size).
        out_features: N (output) axis. The auto-grid heuristic
            requires ``out_features % 128 == 0``.
        residual: If ``True``, dispatches to the ``_with_residual``
            variant. ``compile()`` then requires a ``residual``
            DTensor of shape ``(batch_size, out_features)`` bf16.
        swap_ab: If ``True``, uses the swapAB variant.
        bias_term: Accepted for ``nn.Linear`` parity. Must be ``False``
            (the FP8 kernels do not implement a bias add).
        scale_ue8m0: Indicates the input/weight scales are UE8M0-packed
            uint32. Required True — the FP8 linear kernels only
            consume UE8M0 (plain fp32 scales are for MoE group GEMMs).
        prefix: HF state_dict / tensor-name prefix.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        residual: bool = False,
        swap_ab: bool = False,
        bias_term: bool = False,
        scale_ue8m0: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if bias_term:
            raise NotImplementedError(
                "LinearFP8(bias_term=True) is not supported — the FP8 "
                "linear kernels have no bias add. Fold the bias into the "
                "residual stream or add a separate Add layer."
            )
        if not scale_ue8m0:
            raise NotImplementedError(
                "LinearFP8 requires UE8M0-packed scales (scale_ue8m0=True). "
                "Plain fp32 scales are only supported by the MoE group GEMM "
                "path."
            )
        if in_features % 128 != 0:
            raise ValueError(
                f"LinearFP8: in_features={in_features} must be a multiple "
                "of 128 (FP8 UE8M0 block size)."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.swap_ab = swap_ab
        self.scale_ue8m0 = scale_ue8m0

        # FP8 weight + UE8M0 scale. We use raw uint8 / uint32 storage so
        # the device pointer matches the kernel's expectation byte-for-byte.
        # (torch.float8_e4m3fn would give identical bytes but PyTorch's
        # cast / view path for fp8 is patchy across versions.)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.uint8),
            requires_grad=False,
        )
        # Stored col-major along M as the kernel expects:
        # weight_scale logical shape (M, packed_K) but registered with
        # dim0 = M-slice (grid.x splits dim0). We surface it as
        # (out_features, in_features // 128) for HF loaders; the
        # builder can reinterpret as needed.
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, in_features // 128, dtype=torch.uint32),
            requires_grad=False,
        )

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dequantize both operands to fp32 then compute ``F.linear``.

        The kernel's compute is fp32-accumulator MMA with hardware
        UE8M0 dequant; this reference matches the algebraic semantics
        but not the bit-exact output (the kernel rounds at every MMA
        partial).

        Args:
            x_fp8:    ``(B, in_features)`` FP8 (uint8 or float8_e4m3fn).
            x_scale:  ``(B, in_features // 128)`` UE8M0-packed uint32.
            residual: For ``residual=True`` builds, ``(B, out_features)``
                bf16. ``None`` for the no-residual variant.

        Returns:
            ``(B, out_features)`` bf16. swap_ab does not change the
            algebra (only the operand layout); we ignore it here.
        """
        if self.residual and residual is None:
            raise ValueError(
                "LinearFP8(residual=True).forward requires a residual tensor."
            )
        if not self.residual and residual is not None:
            raise ValueError(
                "LinearFP8(residual=False).forward got an unexpected "
                "residual tensor."
            )

        x_f32 = _dequant_fp8(x_fp8, x_scale)
        w_f32 = _dequant_fp8(self.weight, self.weight_scale)
        out_f32 = F.linear(x_f32, w_f32)
        if residual is not None:
            out_f32 = out_f32 + residual.float()
        return out_f32.to(torch.bfloat16)

    # ------------------------------------------------------------------
    # Grid heuristic — split N by 128, cap at num_workers.
    # ------------------------------------------------------------------
    def auto_grid_dim(self, x_fp8: Any = None) -> GridDim:
        """``(out_features // 128, 1, 1)``, capped at ``pk.num_workers``.

        128 is the kernel's MMA-M tile on SM100 (see
        ``linear_sm100_mpk.cuh``). The DSv3 builder picks the same tile
        for the FP8 dense linears.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        if self.out_features % 128 != 0:
            raise ValueError(
                f"LinearFP8.auto_grid_dim: out_features={self.out_features} "
                "must be a multiple of 128. Pass grid_dim explicitly if "
                "you need a different tile."
            )
        gx = max(1, min(self.out_features // 128, int(pk.num_workers)))
        return (gx, 1, 1)

    def default_block_dim(self) -> BlockDim:
        """All FP8 SM100 linears use 256 threads (4 warps for swapAB,
        8 warps for the standard variant — both fit in a 256-thread CTA)."""
        return (256, 1, 1)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    def compile(
        self,
        x_fp8: Any,
        x_scale: Any,
        *,
        residual: Optional[Any] = None,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
        gate_mode: int = 0,
    ) -> Any:
        """Register one of the four ``linear_fp8_*`` tasks on the active PK.

        Args:
            x_fp8: FP8 activations DTensor ``(B, in_features)``.
            x_scale: UE8M0-packed uint32 scale DTensor
                ``(B, in_features // 128)``.
            residual: Required when ``self.residual=True``. DTensor
                ``(B, out_features)`` bf16.
            output: ``None`` (allocate via ``pk.new_tensor``),
                ``torch.Tensor`` (attach for test readback), or a
                ``DTensor`` (use as-is).
            grid_dim / block_dim: Explicit overrides; ``None`` falls
                back to :meth:`auto_grid_dim` /
                :meth:`default_block_dim`.
            gate_mode: Passed through to the underlying pk method.
                Non-zero enables the gate-emit fast path used by the
                MoE fused-gate decode. ``0`` is the default.

        Returns:
            The output DTensor.
        """
        import torch as _torch
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if self.residual != (residual is not None):
            raise ValueError(
                f"LinearFP8(residual={self.residual}).compile: residual "
                f"must be {'provided' if self.residual else 'None'}; got "
                f"residual={'<value>' if residual is not None else 'None'}"
            )

        w_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")
        # SFB layout depends on dispatch:
        #   - linear_fp8_sm100 / linear_fp8_with_residual_sm100 use a TMA
        #     descriptor for SFB that reads col-major (M, packed_K) — see
        #     ``tma.cuh`` ``TASK_LINEAR_FP8_SM100`` SFB encode.
        #   - linear_fp8_swapAB_* read scales via raw pointers in row-major
        #     ``src[row * packed_K + packed_k_idx]`` (codegen passes
        #     ``weight_scale_row_stride = packed_K``). Row-major is correct
        #     for those.
        # ``self.weight_scale`` is row-major ``(M, packed_K)`` for HF
        # state_dict compatibility; we materialize a col-major view only
        # on the TMA path and cache it on ``self`` so the underlying
        # storage outlives the kernel run.
        if self.swap_ab:
            ws_attached = self.weight_scale
        else:
            ws_attached = _colmajor_weight_scale_for_tma(self.weight_scale)
            self._weight_scale_colmajor = ws_attached
        ws_dt = pk.attach_input(
            ws_attached, name=f"{self.prefix}weight_scale"
        )

        batch_size = x_fp8.dim(0)
        if output is None:
            out_dt = pk.new_tensor(
                dims=(batch_size, self.out_features),
                dtype=mi.bfloat16,
                name=f"{self.prefix}linear_fp8_out",
            )
        elif isinstance(output, _torch.Tensor):
            out_dt = pk.attach_input(
                output, name=f"{self.prefix}linear_fp8_out"
            )
        else:
            out_dt = output

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x_fp8)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration. The four (swap_ab, residual) variants
        # share the same TBGraph wiring (input/scale/weight/scale/[residual]/output);
        # only the task name and `params` differ.
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x_fp8,    (-1, -1, -1), -1, True)
        tb_graph.new_input(x_scale,  (-1, -1, -1), -1, True)
        tb_graph.new_input(w_dt,     (0, -1, -1),  -1, True)
        tb_graph.new_input(ws_dt,    (0, -1, -1),  -1, True)
        if self.residual:
            tb_graph.new_input(residual, (1, -1, -1), -1, True)
        tb_graph.new_input(out_dt,   (1, -1, -1),  -1, True)
        if self.residual:
            pk.kn_graph.customized(
                [x_fp8, x_scale, w_dt, ws_dt, residual, out_dt], tb_graph)
            params = [1] if gate_mode == 0 else [1, gate_mode]
        else:
            pk.kn_graph.customized(
                [x_fp8, x_scale, w_dt, ws_dt, out_dt], tb_graph)
            params = [] if gate_mode == 0 else [gate_mode]

        if self.swap_ab and self.residual:
            pk.kn_graph.register_task(
                tb_graph, "linear_fp8_swapAB_with_residual_sm100", params)
        elif self.swap_ab:
            pk.kn_graph.register_task(
                tb_graph, "linear_fp8_swapAB_sm100", params)
        elif self.residual:
            pk.kn_graph.register_task(
                tb_graph, "linear_fp8_with_residual_sm100", params)
        else:
            pk.kn_graph.register_task(
                tb_graph, "linear_fp8_sm100", params)
        return out_dt


# ----------------------------------------------------------------------
# LinearFP8BMM — per-head batched matmul (decode Q absorb path).
# ----------------------------------------------------------------------
class LinearFP8BMM(MPKModule):
    """Per-head FP8 batched matmul on SM100.

    Computes ``output[n, h, :] = input[n, h, :] @ weight[h, :, :].T``
    with FP8 E4M3 operands and UE8M0-packed scales. Used by the
    DeepSeek V3 decode Q-absorb path:
    ``q_nope_fp8 @ kv_b_k_bmm.T → q_nope_abs``.

    Layouts:

    * ``input_fp8``   : ``(N, H, D_in)`` E4M3.
    * ``input_scale`` : ``(N, H, packed_K)`` UE8M0 uint32.
    * ``weight_fp8`` (param) : ``(H, D_out, D_in)`` E4M3.
    * ``weight_scale`` (param) : ``(H, D_out, packed_K)`` UE8M0 uint32.
    * ``output``      : ``(N, H, D_out)`` bf16.

    Decode-only (``N <= 16``). Grid is ``(m_shards_per_head, H, 1)``;
    first cut requires ``grid.y == H`` (one head per CTA). Block dim is
    256 (kernel role layout fixed at 8 warps).

    Args:
        num_heads: ``H``.
        in_features_per_head: ``D_in``. Must be a multiple of 128.
        out_features_per_head: ``D_out``. Must be a multiple of MMA-M
            (128) per CTA; with the default grid.x=1 that's a multiple
            of 128 overall.
        scale_ue8m0: Required True (kernel only accepts UE8M0).
        prefix: HF state_dict / tensor-name prefix.
    """

    def __init__(
        self,
        num_heads: int,
        in_features_per_head: int,
        out_features_per_head: int,
        *,
        scale_ue8m0: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if not scale_ue8m0:
            raise NotImplementedError(
                "LinearFP8BMM requires UE8M0-packed scales."
            )
        if in_features_per_head % 128 != 0:
            raise ValueError(
                f"LinearFP8BMM: in_features_per_head={in_features_per_head} "
                "must be a multiple of 128."
            )
        if out_features_per_head % 128 != 0:
            raise ValueError(
                f"LinearFP8BMM: out_features_per_head={out_features_per_head} "
                "must be a multiple of 128 (kernel MMA-M=128)."
            )
        self.num_heads = num_heads
        self.in_features_per_head = in_features_per_head
        self.out_features_per_head = out_features_per_head
        self.scale_ue8m0 = scale_ue8m0

        packed_k = in_features_per_head // 128
        self.weight = nn.Parameter(
            torch.empty(
                num_heads, out_features_per_head, in_features_per_head,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(
                num_heads, out_features_per_head, packed_k,
                dtype=torch.uint32,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Per-head dequant + batched matmul reference.

        Args:
            x_fp8:   ``(N, H, D_in)`` E4M3.
            x_scale: ``(N, H, packed_K)`` UE8M0 uint32.

        Returns:
            ``(N, H, D_out)`` bf16.
        """
        x_f32 = _dequant_fp8(x_fp8, x_scale)  # (N, H, D_in)
        w_f32 = _dequant_fp8(self.weight, self.weight_scale)  # (H, D_out, D_in)
        # einsum: per-head matmul. x[n, h, d_in] * w[h, d_out, d_in] -> out[n, h, d_out]
        out_f32 = torch.einsum("nhi,hoi->nho", x_f32, w_f32)
        return out_f32.to(torch.bfloat16)

    def auto_grid_dim(self, x_fp8: Any) -> GridDim:
        """``(1, num_heads, 1)`` — one head per CTA, no M-shard."""
        return (1, self.num_heads, 1)

    def default_block_dim(self) -> BlockDim:
        """Kernel role layout assumes 256 threads (8 warps)."""
        return (256, 1, 1)

    def compile(
        self,
        x_fp8: Any,
        x_scale: Any,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        import torch as _torch
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x_fp8)
        if block_dim is None:
            block_dim = self.default_block_dim()

        w_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")
        # SFB layout: ``linear_fp8_bmm_sm100`` reads scales via raw
        # pointers ``src[row * packed_K + packed_k_idx]`` (codegen passes
        # ``weight_scale_row_stride = packed_K``), so the per-head slice
        # ``(D_out, packed_K)`` row-major is the correct layout. No
        # transpose needed; just attach the parameter directly.
        # (Cf. ``tma.cuh`` ``TASK_LINEAR_FP8_BMM_SM100`` — scales are NOT
        # TMA-descriptor backed for the BMM kernel.)
        ws_dt = pk.attach_input(
            self.weight_scale, name=f"{self.prefix}weight_scale"
        )

        # Resolve output. The pk method accepts both 3D and 2D output;
        # production allocates 3D (N, H, D_out).
        N = x_fp8.dim(0)
        if output is None:
            out_dt = pk.new_tensor(
                dims=(N, self.num_heads, self.out_features_per_head),
                dtype=mi.bfloat16,
                name=f"{self.prefix}linear_fp8_bmm_out",
            )
        elif isinstance(output, _torch.Tensor):
            out_dt = pk.attach_input(
                output, name=f"{self.prefix}linear_fp8_bmm_out"
            )
        else:
            out_dt = output

        # Inlined task registration (was pk.linear_fp8_bmm_sm100_layer).
        # Per-head FP8 batched matmul on SM100. Weight is 3D (H, D_out, D_in);
        # input/output may be 2D (N, H*D) or 3D (N, H, D) — same byte layout.
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert w_dt.num_dims == 3
        assert ws_dt.num_dims == 3
        assert x_fp8.num_dims in (2, 3)
        assert x_scale.num_dims in (2, 3)
        assert out_dt.num_dims in (2, 3)
        params = []
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        in_h_axis = 1 if x_fp8.num_dims == 3 else 1
        in_sc_h_axis = 1 if x_scale.num_dims == 3 else 1
        out_h_axis = 1
        out_m_axis = 2 if out_dt.num_dims == 3 else 1
        tb_graph.new_input(x_fp8,    (-1, in_h_axis, -1),    -1, True)
        tb_graph.new_input(x_scale,  (-1, in_sc_h_axis, -1), -1, True)
        tb_graph.new_input(w_dt,     (1, 0, -1),             -1, True)
        tb_graph.new_input(ws_dt,    (1, 0, -1),             -1, True)
        if out_dt.num_dims == 3:
            tb_graph.new_input(out_dt, (out_m_axis, out_h_axis, -1), -1, True)
        else:
            assert grid_dim[0] == 1, (
                "linear_fp8_bmm with 2D output requires grid.x=1 "
                "(D_out cannot be sharded across CTAs when packed flat)")
            tb_graph.new_input(out_dt, (-1, 1, -1), -1, True)
        pk.kn_graph.customized(
            [x_fp8, x_scale, w_dt, ws_dt, out_dt], tb_graph)
        pk.kn_graph.register_task(tb_graph, "linear_fp8_bmm_sm100", params)
        return out_dt


# ----------------------------------------------------------------------
# LinearSplitKFP8SwapAB — Split-K decode variant.
# ----------------------------------------------------------------------
class LinearSplitKFP8SwapAB(MPKModule):
    """Split-K FP8 swapAB linear — DSv3 decode-only.

    Wraps :meth:`PersistentKernel.linear_splitk_swapAB_fp8_layer`
    (task ``splitk_linear_fp8_swapAB_sm100``). ``grid.y`` CTAs each
    compute a K-slice partial and TMA reduce-add into the shared
    output tile.

    The kernel always *adds* onto ``output``. The ``accumulate``
    constructor flag matches the pk method:

    * ``accumulate=True`` — caller owns ``output`` (typically a
      residual stream). No pre-zero; the matmul is added on top.
    * ``accumulate=False`` — layer prepends a ``tensor_init`` to zero
      ``output`` before the GEMM. The result is a pure sum.

    Constraints:

    * ``out_features / grid.x`` must be a multiple of 128 (per-task N).
    * ``in_features / grid.y`` must be a multiple of 128 (per-task K).
    * ``batch_size <= 16`` (decode-only).

    Args:
        in_features: K (reduction) axis. Multiple of 128.
        out_features: N (output) axis. Multiple of 128.
        accumulate: See above.
        scale_ue8m0: Required True.
        prefix: HF state_dict / tensor-name prefix.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        accumulate: bool,
        scale_ue8m0: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if not scale_ue8m0:
            raise NotImplementedError(
                "LinearSplitKFP8SwapAB requires UE8M0-packed scales."
            )
        if in_features % 128 != 0:
            raise ValueError(
                f"LinearSplitKFP8SwapAB: in_features={in_features} must "
                "be a multiple of 128."
            )
        if out_features % 128 != 0:
            raise ValueError(
                f"LinearSplitKFP8SwapAB: out_features={out_features} must "
                "be a multiple of 128."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.accumulate = accumulate
        self.scale_ue8m0 = scale_ue8m0

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.uint8),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, in_features // 128, dtype=torch.uint32),
            requires_grad=False,
        )

    def forward(
        self,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dequant + linear; for ``accumulate=True`` adds onto ``output``.

        Args:
            x_fp8:   ``(B, in_features)`` E4M3.
            x_scale: ``(B, in_features // 128)`` UE8M0 uint32.
            output:  Required when ``self.accumulate=True``. Pre-existing
                ``(B, out_features)`` bf16 tensor to add onto.

        Returns:
            ``(B, out_features)`` bf16.
        """
        x_f32 = _dequant_fp8(x_fp8, x_scale)
        w_f32 = _dequant_fp8(self.weight, self.weight_scale)
        result = F.linear(x_f32, w_f32)
        if self.accumulate:
            if output is None:
                raise ValueError(
                    "LinearSplitKFP8SwapAB(accumulate=True).forward "
                    "requires the prior `output` tensor (the residual)."
                )
            result = result + output.float()
        return result.to(torch.bfloat16)

    def auto_grid_dim(self, x_fp8: Any = None) -> GridDim:
        """``(out_features // 128, split_k, 1)``.

        ``split_k`` is heuristically picked so that
        ``grid.x * grid.y <= pk.num_workers`` and ``in_features / grid.y``
        stays a multiple of 128. Default ``split_k = max(1, num_workers
        // grid.x)``.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        gx = max(1, min(self.out_features // 128, int(pk.num_workers)))
        # Pick split_k = largest divisor of (in_features // 128) that
        # fits within num_workers // gx.
        max_k_blocks = self.in_features // 128
        budget = max(1, int(pk.num_workers) // gx)
        split_k = 1
        for candidate in range(min(budget, max_k_blocks), 0, -1):
            if max_k_blocks % candidate == 0:
                split_k = candidate
                break
        return (gx, split_k, 1)

    def default_block_dim(self) -> BlockDim:
        return (256, 1, 1)

    def compile(
        self,
        x_fp8: Any,
        x_scale: Any,
        output: Any,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Register the split-K FP8 swapAB linear.

        ``output`` MUST be a caller-allocated DTensor (the split-K
        kernel reduce-adds into it). For ``accumulate=True`` callers
        pre-populate it (residual); for ``accumulate=False`` the
        layer prepends a tensor_init to zero it.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x_fp8)
        if block_dim is None:
            block_dim = self.default_block_dim()

        w_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")
        # SFB layout: ``splitk_linear_fp8_swapAB_sm100`` reads scales via
        # raw pointers in row-major ``src[row * packed_K + packed_k_idx]``
        # (codegen passes ``weight_scale_row_stride = packed_K``). Row-
        # major ``(M, packed_K)`` is the kernel-expected layout — no
        # transpose. Scales are NOT TMA-descriptor backed for the splitK
        # swapAB kernel either (see ``tma.cuh``).
        ws_dt = pk.attach_input(
            self.weight_scale, name=f"{self.prefix}weight_scale"
        )

        # Inlined task registration (was pk.linear_splitk_swapAB_fp8_layer).
        # Split-K kernel uses tma_reduce_add_async; for accumulate=False we
        # prepend a tensor_init that zeroes `output` before the GEMM.
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        if not self.accumulate:
            pk.tensor_init_layer(
                target=output,
                dummy=x_fp8,
                grid_dim=grid_dim,
                block_dim=block_dim,
                dummy_input_map=(-1, 1, -1),
                target_input_map=(1, -1, -1),
            )
        params = []
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x_fp8,    (-1, 1, -1), 1, True)
        tb_graph.new_input(x_scale,  (-1, 1, -1), 1, True)
        tb_graph.new_input(w_dt,     (0, 1, -1),  1, True)
        tb_graph.new_input(ws_dt,    (0, 1, -1),  1, True)
        tb_graph.new_input(output,   (1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [x_fp8, x_scale, w_dt, ws_dt, output], tb_graph)
        pk.kn_graph.register_task(
            tb_graph, "splitk_linear_fp8_swapAB_sm100", params)
        return output
