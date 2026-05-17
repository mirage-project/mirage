"""Grouped (per-expert) FP8 block-scaled GEMM on SM100.

Wraps :meth:`PersistentKernel.fp8_group_gemm_smallm_layer` (task
``fp8_group_gemm_smallm_sm100``, BN=64 NS=8) and
:meth:`...fp8_group_gemm_largem_layer` (task
``fp8_group_gemm_largem_sm100``, BN=128 NS=6), with an ``"auto"``
mode that mirrors :meth:`...fp8_group_gemm_layer`'s ``(K, MPE)`` based
dispatcher.

Variant dispatch
----------------

+-------------+-------------------------------------------+
| ``variant`` | underlying pk method                      |
+=============+===========================================+
| ``"smallm"``| ``fp8_group_gemm_smallm_layer``           |
| ``"largem"``| ``fp8_group_gemm_largem_layer``           |
| ``"auto"``  | ``fp8_group_gemm_layer`` (K>4096 && MPE<=8|
|             | → smallm, else largem)                    |
+-------------+-------------------------------------------+

Computes per-expert
``D[r, :] = (A[r, :] * scale_a[r]) @ (B[m_indices[r]].T * scale_b)``
with hardware UE8M0 dequant. Rows in each ``BM=128`` block must share
the same expert id (caller responsibility — typically via the
:class:`MoEPermute` companion).

Tensor layout
-------------

* ``a_fp8``      : ``(M_total, K)`` E4M3 (packed FP8). Already
                   permuted so contiguous 128-row blocks share an expert.
* ``b_fp8``      : ``(E, N, K)`` E4M3 — owned by this module as
                   ``self.weight``.
* ``sfa_packed`` : ``(num_sf_k, M_total)`` uint32, UE8M0-packed
                   **K-outermost** (note the transpose vs the
                   ``quantize_fp8`` output — see
                   :class:`TransposeScale`).
* ``sfb_packed`` : ``(num_sf_k, E * N)`` uint32 — owned as
                   ``self.weight_scale``, derived from the per-expert
                   ``[E, N/128, K/128]`` scale tensor by
                   ``repeat_interleave(N) → uint32 pack``.
* ``m_indices``  : ``(M_total,)`` int32 expert id per row.
* ``output``     : ``(M_total, N)`` bf16.

FP8 scale-layout quirk
----------------------

``sfa_packed`` arrives K-outermost (the kernel's TMA descriptor expects
this), but :meth:`PersistentKernel.quantize_fp8_layer` writes M-outermost.
The DSv3 builder therefore inserts a :class:`TransposeScale` between
quantize and group_gemm. ``sfb_packed`` is built once at load time and
stored in this module — no per-iter transpose is needed for the weight
scales.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

import torch
import torch.nn as nn

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule
from .linear_fp8 import _dequant_fp8


__all__ = ["FP8GroupGEMM"]


Variant = Literal["smallm", "largem", "auto"]


class FP8GroupGEMM(MPKModule):
    """Per-expert grouped FP8 GEMM.

    Args:
        num_experts: ``E`` — first dim of the weight tensor.
        in_features: ``K`` — reduction axis. Multiple of 128.
        out_features: ``N`` — per-expert output dim. Multiple of 128.
        variant: ``"smallm"``, ``"largem"``, or ``"auto"`` (dispatches
            based on ``K`` and per-expert M; see module docstring).
        scale_ue8m0: Required True.
        prefix: HF state_dict / tensor-name prefix.

    Owned parameters:

    * ``weight``      : ``(E, N, K)`` ``uint8`` storage (E4M3 bytes).
    * ``weight_scale`` (a.k.a. ``sfb_packed``) : ``(packed_K, E * N)``
      ``uint32`` storage. **Already in the K-outermost packed layout**
      the kernel expects — the model loader is responsible for the
      one-time pack at weight-load time. (See
      ``demo/deepseek_v3/builder.py`` for the exact transform.)
    """

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        *,
        variant: Variant = "auto",
        scale_ue8m0: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if variant not in ("smallm", "largem", "auto"):
            raise ValueError(
                f"FP8GroupGEMM.variant must be 'smallm', 'largem', or "
                f"'auto'; got {variant!r}"
            )
        if not scale_ue8m0:
            raise NotImplementedError(
                "FP8GroupGEMM requires UE8M0-packed scales."
            )
        if in_features % 128 != 0:
            raise ValueError(
                f"FP8GroupGEMM: in_features={in_features} must be a "
                "multiple of 128."
            )
        if out_features % 128 != 0:
            raise ValueError(
                f"FP8GroupGEMM: out_features={out_features} must be a "
                "multiple of 128."
            )
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.variant = variant
        self.scale_ue8m0 = scale_ue8m0

        # nk = in_features // 128 (one UE8M0 byte per 128-K-element block).
        # The packed scale stores 4 consecutive UE8M0 bytes per uint32 along
        # the K-block axis, so the K-outer dim is num_sf_k = ceil(nk / 4).
        # The kernel's SFB TMA descriptor reads this as logical
        # ``[num_sf_k, E*N]`` row-major uint32 (see ``tma.cuh``
        # ``TASK_FP8_GROUP_GEMM_*`` SFB encode — gd=[E*N, num_sf_k]).
        nk = in_features // 128
        num_sf_k = (nk + 3) // 4
        self.weight = nn.Parameter(
            torch.empty(
                num_experts, out_features, in_features, dtype=torch.uint8
            ),
            requires_grad=False,
        )
        # sfb_packed in its final K-outermost packed layout. Shape matches
        # the kernel's TMA descriptor: (num_sf_k, E*N) uint32, row-major.
        self.weight_scale = nn.Parameter(
            torch.empty(
                num_sf_k, num_experts * out_features, dtype=torch.uint32
            ),
            requires_grad=False,
        )

    def forward(
        self,
        a_fp8: torch.Tensor,
        sfa_packed: torch.Tensor,
        m_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dequant + per-row expert lookup + matmul.

        Args:
            a_fp8:      ``(M_total, K)`` E4M3. Each contiguous 128-row
                        block shares an expert (caller responsibility).
            sfa_packed: ``(packed_K, M_total)`` UE8M0-packed uint32.
            m_indices:  ``(M_total,)`` int32. Expert id per row.

        Returns:
            ``(M_total, N)`` bf16.
        """
        # Transpose sfa_packed back to (M_total, packed_K) so the
        # _dequant_fp8 helper (which expects trailing-K scales) works.
        # NOTE: ``sfa_packed`` arrives as the kernel-shaped
        # ``(num_sf_k, M_total)`` uint32 (4 UE8M0 bytes per uint32 along
        # K-block axis). ``_dequant_fp8`` reinterprets uint32 as 4 bytes
        # and then ``repeat_interleave(128)`` to expand, so we just need
        # to put M_total on the leading axis.
        sfa_m_outermost = sfa_packed.transpose(0, 1).contiguous()
        a_f32 = _dequant_fp8(a_fp8, sfa_m_outermost)  # (M_total, K)

        # Reshape sfb back from (num_sf_k, E*N) → per-expert (E, N, num_sf_k),
        # then expand 4 UE8M0 bytes per uint32 along the K-block axis and
        # finally repeat to per-element along K.
        E, N, K = self.num_experts, self.out_features, self.in_features
        nk = K // 128
        num_sf_k = (nk + 3) // 4
        # weight_scale stored as (num_sf_k, E*N) row-major.
        sfb = (
            self.weight_scale  # (num_sf_k, E*N)
            .view(num_sf_k, E, N)
            .permute(1, 2, 0)  # (E, N, num_sf_k)
            .contiguous()
        )
        w_f32 = self.weight.view(torch.float8_e4m3fn).float()  # (E, N, K)
        # Unpack uint32 → 4 UE8M0 bytes along K-block axis.
        sfb_bytes = sfb.view(torch.uint8).reshape(E, N, num_sf_k * 4)
        # Drop any padding past nk before expanding to K.
        sfb_bytes = sfb_bytes[..., :nk]
        sfb_f32 = torch.pow(torch.tensor(2.0), sfb_bytes.to(torch.float32) - 127.0)
        sfb_expanded = sfb_f32.repeat_interleave(128, dim=-1)[..., :K]
        w_dequant = w_f32 * sfb_expanded

        M_total = a_fp8.shape[0]
        out = torch.zeros(M_total, N, dtype=torch.float32, device=a_f32.device)
        for r in range(M_total):
            e = int(m_indices[r].item())
            if e < 0 or e >= E:
                continue
            out[r] = a_f32[r] @ w_dequant[e].t()
        return out.to(torch.bfloat16)

    def auto_grid_dim(self, *_: Any) -> GridDim:
        """Grid is fixed at ``(num_workers, 1, 1)`` by the pk method."""
        from ... import context as _ctx

        pk = _ctx.current_pk()
        return (int(pk.num_workers), 1, 1)

    def default_block_dim(self) -> BlockDim:
        return (256, 1, 1)

    def compile(
        self,
        a_fp8: Any,
        sfa_packed: Any,
        m_indices: Any,
        output: Any,
        *,
        num_workers: Optional[int] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Register the appropriate ``fp8_group_gemm_*`` task.

        Args:
            a_fp8: Permuted FP8 activations DTensor ``(M_total, K)``.
            sfa_packed: UE8M0-packed K-outermost scale DTensor
                ``(packed_K, M_total)``. See :class:`TransposeScale`.
            m_indices: ``(M_total,)`` int32 DTensor — expert per row.
            output: Caller-allocated ``(M_total, N)`` bf16 DTensor.
            num_workers: ``grid.x`` width. Defaults to
                ``current_pk().num_workers``.

        Returns:
            ``output``.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if num_workers is None:
            num_workers = int(pk.num_workers)

        b_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")
        sfb_dt = pk.attach_input(
            self.weight_scale, name=f"{self.prefix}weight_scale"
        )

        # Inlined task registration (was pk.fp8_group_gemm_{smallm,largem,auto}
        # via the shared _fp8_group_gemm_layer_impl). Both backing kernels
        # share identical TBGraph wiring; only the task name changes.
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        # Resolve the variant. "auto" mirrors pk.fp8_group_gemm_layer's
        # (K, M-per-expert) heuristic.
        if self.variant == "auto":
            K_dim = a_fp8.dim(1)
            M_total_local = a_fp8.dim(0)
            E_local = b_dt.dim(0)
            MPE = M_total_local // E_local
            resolved_variant = "smallm" if (K_dim > 4096 and MPE <= 8) else "largem"
        else:
            resolved_variant = self.variant
        task_name = ("fp8_group_gemm_smallm_sm100"
                     if resolved_variant == "smallm"
                     else "fp8_group_gemm_largem_sm100")

        assert a_fp8.num_dims == 2
        assert b_dt.num_dims == 3
        assert output.num_dims == 2
        M_total = a_fp8.dim(0)
        K = a_fp8.dim(1)
        E = b_dt.dim(0)
        N = b_dt.dim(1)
        assert b_dt.dim(2) == K
        assert m_indices.dim(0) == M_total
        params = [M_total, N, K, E, num_workers]
        grid_dim_local = (num_workers, 1, 1)
        block_dim_local = (256, 1, 1)  # 8 warps fixed by kernel role layout
        tb_graph = TBGraph(CyTBGraph(grid_dim_local, block_dim_local, 1, 64))
        tb_graph.new_input(a_fp8,      (-1, -1, -1), -1, True)
        tb_graph.new_input(b_dt,       (-1, -1, -1), -1, True)
        tb_graph.new_input(sfa_packed, (-1, -1, -1), -1, True)
        tb_graph.new_input(sfb_dt,     (-1, -1, -1), -1, True)
        tb_graph.new_input(m_indices,  (-1, -1, -1), -1, True)
        tb_graph.new_input(output,     (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [a_fp8, b_dt, sfa_packed, sfb_dt, m_indices, output], tb_graph)
        pk.kn_graph.register_task(tb_graph, task_name, params)
        return output
