"""Grouped (per-expert) FP8 block-scaled GEMM on SM100.

Per-arch task kernel:
* SM100 Blackwell smallm : ``tasks/blackwell/fp8_group_gemm_smallm_sm100.cuh`` (``fp8_group_gemm_smallm_sm100``, BN=64 NS=8)
* SM100 Blackwell largem : ``tasks/blackwell/fp8_group_gemm_largem_sm100.cuh`` (``fp8_group_gemm_largem_sm100``, BN=128 NS=6)

Shared body in ``fp8_group_gemm_sm100_common.cuh``.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule
from .linear_fp8 import _dequant_fp8


__all__ = ["FP8GroupGEMMSmallM", "FP8GroupGEMMLargeM", "FP8GroupGEMMAuto"]


class _FP8GroupGEMMBase(MPKModule):
    """Per-expert grouped FP8 GEMM base; subclasses set ``_TASK_NAME`` or override.

    Args:
        num_experts: ``E`` — first dim of the weight.
        in_features: ``K`` — reduction. Multiple of 128.
        out_features: ``N`` — per-expert output. Multiple of 128.
        scale_ue8m0: Required True.
        prefix: state_dict / tensor-name prefix.

    Owned parameters:
    * ``weight`` ``(E, N, K)`` uint8 (E4M3 bytes).
    * ``weight_scale`` ``(num_sf_k, E * N)`` uint32 (UE8M0-packed,
      K-outermost; ``num_sf_k = ceil((K // 128) / 4)``).
    """

    _TASK_NAME: str = ""

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        *,
        scale_ue8m0: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if not scale_ue8m0:
            raise NotImplementedError(
                f"{type(self).__name__} requires UE8M0-packed scales."
            )
        if in_features % 128 != 0:
            raise ValueError(
                f"{type(self).__name__}: in_features={in_features} must be a multiple of 128."
            )
        if out_features % 128 != 0:
            raise ValueError(
                f"{type(self).__name__}: out_features={out_features} must be a multiple of 128."
            )
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.scale_ue8m0 = scale_ue8m0

        nk = in_features // 128
        num_sf_k = (nk + 3) // 4
        self.weight = nn.Parameter(
            torch.empty(
                num_experts, out_features, in_features, dtype=torch.uint8
            ),
            requires_grad=False,
        )
        # K-outermost packed layout matching the kernel SFB TMA descriptor.
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
            a_fp8: ``(M_total, K)`` E4M3.
            sfa_packed: ``(num_sf_k, M_total)`` UE8M0-packed uint32 (K-outermost).
            m_indices: ``(M_total,)`` int32 expert id per row.
        Returns: ``(M_total, N)`` bf16.
        """
        # Put M on the leading axis so _dequant_fp8 (which expects
        # trailing-K scales) works.
        sfa_m_outermost = sfa_packed.transpose(0, 1).contiguous()
        a_f32 = _dequant_fp8(a_fp8, sfa_m_outermost)  # (M_total, K)

        E, N, K = self.num_experts, self.out_features, self.in_features
        nk = K // 128
        num_sf_k = (nk + 3) // 4
        sfb = (
            self.weight_scale
            .view(num_sf_k, E, N)
            .permute(1, 2, 0)
            .contiguous()
        )
        w_f32 = self.weight.view(torch.float8_e4m3fn).float()
        sfb_bytes = sfb.view(torch.uint8).reshape(E, N, num_sf_k * 4)
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
        """Grid fixed at ``(num_workers, 1, 1)``: each task strides over output tiles."""
        from ... import context as _ctx

        pk = _ctx.current_pk()
        return (int(pk.num_workers), 1, 1)

    def default_block_dim(self) -> BlockDim:
        return (256, 1, 1)

    def _register(
        self,
        a_fp8: Any,
        sfa_packed: Any,
        m_indices: Any,
        output: Any,
        num_workers: Optional[int],
        task_name: str,
    ) -> Any:
        """Shared compile body — wires inputs and registers ``task_name``."""
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        if num_workers is None:
            num_workers = int(pk.num_workers)

        b_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")
        sfb_dt = pk.attach_input(
            self.weight_scale, name=f"{self.prefix}weight_scale"
        )

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
        block_dim_local = (256, 1, 1)
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

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class FP8GroupGEMMSmallM(_FP8GroupGEMMBase):
    """Grouped FP8 GEMM — smallm variant (BN=64, NS=8). Best for K>4096 && MPE<=8.

    Args: see :class:`_FP8GroupGEMMBase`.
    """

    _TASK_NAME = "fp8_group_gemm_smallm_sm100"

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
        """Register ``fp8_group_gemm_smallm_sm100`` (BN=64, NS=8; K>4096 && MPE<=8).

        Tensor contract:
          a_fp8:        (M_total, K) fp8_e4m3 as uint8, row-major. A operand.
          sfa_packed:   (num_sf_k, M_total) uint32 UE8M0-packed, K-outermost (num_sf_k=ceil((K//128)/4)).
          m_indices:    (M_total,) int32 expert-id per row; -1 to skip.
          weight:       (E, N, K) fp8_e4m3 as uint8, row-major stacked along expert dim (owned). B operand.
          weight_scale: (num_sf_k, E*N) uint32 UE8M0-packed, K-outermost (owned; matches SFB TMA desc).
          output:       (M_total, N) bf16, row-major, caller-allocated.

        Notes: K and N mult of 128; TMA-aligned; grid (num_workers,1,1) — tasks stride over output tiles.
        params=[M_total, N, K, E, num_workers].
        """
        return self._register(
            a_fp8, sfa_packed, m_indices, output, num_workers, self._TASK_NAME
        )


class FP8GroupGEMMLargeM(_FP8GroupGEMMBase):
    """Grouped FP8 GEMM — largem variant (BN=128, NS=6). Default for prefill / large per-expert M.

    Args: see :class:`_FP8GroupGEMMBase`.
    """

    _TASK_NAME = "fp8_group_gemm_largem_sm100"

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
        """Register ``fp8_group_gemm_largem_sm100`` (BN=128, NS=6; default for prefill / large per-expert M).

        Tensor contract:
          a_fp8:        (M_total, K) fp8_e4m3 as uint8, row-major. A operand.
          sfa_packed:   (num_sf_k, M_total) uint32 UE8M0-packed, K-outermost (num_sf_k=ceil((K//128)/4)).
          m_indices:    (M_total,) int32 expert-id per row; -1 to skip.
          weight:       (E, N, K) fp8_e4m3 as uint8, row-major stacked along expert dim (owned). B operand.
          weight_scale: (num_sf_k, E*N) uint32 UE8M0-packed, K-outermost (owned; matches SFB TMA desc).
          output:       (M_total, N) bf16, row-major, caller-allocated.

        Notes: K and N mult of 128; TMA-aligned; grid (num_workers,1,1).
        params=[M_total, N, K, E, num_workers].
        """
        return self._register(
            a_fp8, sfa_packed, m_indices, output, num_workers, self._TASK_NAME
        )


class FP8GroupGEMMAuto(_FP8GroupGEMMBase):
    """Grouped FP8 GEMM — dispatches to smallm if ``K>4096 && MPE<=8`` else largem at compile time.

    Args: see :class:`_FP8GroupGEMMBase`. ``MPE = M_total // num_experts``.
    """

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
        """Pick variant: ``K>4096 && MPE<=8 → smallm`` else ``largem``; then register.

        Tensor contract:
          a_fp8:        (M_total, K) fp8_e4m3 as uint8, row-major. A operand.
          sfa_packed:   (num_sf_k, M_total) uint32 UE8M0-packed, K-outermost (num_sf_k=ceil((K//128)/4)).
          m_indices:    (M_total,) int32 expert-id per row; -1 to skip.
          weight:       (E, N, K) fp8_e4m3 as uint8, row-major stacked along expert dim (owned). B operand.
          weight_scale: (num_sf_k, E*N) uint32 UE8M0-packed, K-outermost (owned; matches SFB TMA desc).
          output:       (M_total, N) bf16, row-major, caller-allocated.

        Notes: K and N mult of 128; TMA-aligned; MPE = M_total // num_experts.
        """
        K_dim = a_fp8.dim(1)
        M_total_local = a_fp8.dim(0)
        E_local = self.num_experts
        MPE = M_total_local // max(1, E_local)
        task_name = (
            "fp8_group_gemm_smallm_sm100"
            if (K_dim > 4096 and MPE <= 8)
            else "fp8_group_gemm_largem_sm100"
        )
        return self._register(
            a_fp8, sfa_packed, m_indices, output, num_workers, task_name
        )
