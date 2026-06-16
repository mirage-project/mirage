"""DeepSeek-V3 task-registration helpers operating on a PersistentKernel;
moved out of the core file. Every function takes the
PersistentKernel as its first argument (``pk``) and registers one (or a
small family of) megakernel task(s) on ``pk.kn_graph``, exactly as the
former ``PersistentKernel`` methods of the same names did.
"""

from ....core import *
from ....kernel import TBGraph
from ...multigpu import allocate_nvshmem_teams



def fused_rmsnorm_quantize_fp8_layer(
    pk,
    input: DTensor,
    weight: DTensor,
    output_bf16: DTensor,
    output_fp8: DTensor,
    output_scale: DTensor,
    grid_dim: tuple,
    block_dim: tuple,
    process_dim: int = None,
    in_offset_elems: int = 0,
    out_offset_elems: int = 0,
    scale_ue8m0: bool = True,
    emit_bf16: bool = True,
    eps: float = 1e-6,  # accepted for API parity; kernel hardcodes 1e-6f
    epsilon: float = None,  # alias for `eps` to match older call sites
    group_size: int = 128,  # kernel currently asserts GROUP_SIZE == 128
):
    """Fused RMSNorm + per-token-group FP8 quantize.

    Replaces the two-task chain `rmsnorm_layer` + `quantize_fp8_layer`
    when the BF16 rmsnorm output is consumed (only) by an FP8 dense
    GEMM. Saves one dispatch wave + one BF16 HBM round-trip per layer
    (~10 μs/layer expected at TP=4 EP=2 mbt=128 decode).

    Parameters mirror the two underlying calls:
      * `process_dim` / `in_offset_elems` / `out_offset_elems` select
        a column slice of a wider parent buffer (QKV-a FuseTensor
        path). Defaults preserve legacy contiguous behaviour.
      * `scale_ue8m0=True` writes packed UE8M0 uint32 scales in the
        column-major `[packed_k, aligned_batch]` layout that the new
        FP8 dense GEMMs (`fp8_gemm_dense_smallm/mediumm_sm100`) read.
        `False` writes float32 scales in `[batch, num_groups]`
        row-major (MoE permute path).
      * `emit_bf16=False` skips writing the BF16 normalized output to
        HBM. Use when no downstream consumer needs the BF16 (e.g.,
        pre-qkv_a where only the FP8 path reads the result). Defaults
        to True so the wrapper is a strict superset of `rmsnorm_layer`.
      * `eps` / `epsilon`: RMS epsilon (kernel hard-codes 1e-6f today;
        accepted only for API parity).
      * `group_size`: FP8 quantization group size; kernel requires 128.
    """
    del eps, epsilon  # API parity only, kernel uses 1e-6f hard-coded.
    if group_size != 128:
        raise ValueError(
            f"fused_rmsnorm_quantize_fp8_layer requires group_size=128, "
            f"got {group_size}")
    assert input.num_dims == 2
    assert weight.num_dims == 1
    assert output_bf16.num_dims == 2
    assert output_fp8.num_dims == 2
    # output_scale shape is layout-dependent: packed UE8M0 is
    # (packed_k, aligned_batch) column-major; float32 is
    # (batch, num_groups) row-major. Both are 2D.
    assert output_scale.num_dims == 2
    assert input.dim(0) == output_bf16.dim(0)
    assert input.dim(1) == output_bf16.dim(1)
    assert output_fp8.dim(0) == input.dim(0)
    legacy_hidden = input.dim(1)
    if process_dim is None:
        process_dim = legacy_hidden
    assert output_fp8.dim(1) == process_dim, (
        f"output_fp8 second dim must equal process_dim "
        f"({output_fp8.dim(1)} vs {process_dim})")
    assert in_offset_elems + process_dim <= legacy_hidden
    assert out_offset_elems + process_dim <= output_bf16.dim(1)

    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    # IMPORTANT: input order MUST match the C++ task_register reader.
    # input_ptrs[0]=input, [1]=weight, [2]=output_bf16, [3]=output_fp8,
    # [4]=output_scale. We pass outputs via `store_in_dmem=True` inputs
    # so the (num_inputs, num_outputs) tuple in graph.cc is (5, 0).
    #
    # Per-CTA pointer offsetting via dim_maps:
    #   input / output_bf16 / output_fp8: row dim 0 → grid.x, so each
    #     CTA's base pointer is pre-offset to its row-block. The kernel
    #     then walks `batch_idx in [0, BATCH_SIZE)` within that block.
    #   weight: 1D, shared across all CTAs (dim_maps all -1).
    #   output_scale: 2D but BOTH UE8M0 (col-major) and float32
    #     (row-major) layouts need the GLOBAL row index, which the
    #     kernel reconstructs from task_idx = task_metadata.request_id.
    #     dim_maps stays (-1, -1, -1) so the kernel sees the buffer
    #     base pointer.
    tb_graph.new_input(input, (0, -1, -1), 1, True)
    tb_graph.new_input(weight, (-1, -1, -1), 0, True)
    tb_graph.new_input(output_bf16, (0, -1, -1), 1, True)
    tb_graph.new_input(output_fp8, (0, -1, -1), 1, True)
    tb_graph.new_input(output_scale, (-1, -1, -1), -1, True)
    pk.kn_graph.customized(
        [input, weight, output_bf16, output_fp8, output_scale], tb_graph)
    # The C++ register on this branch reads [process_dim, scale_ue8m0,
    # emit_bf16]; slice offsets are carried by mpk.narrow views, not params.
    assert in_offset_elems == 0 and out_offset_elems == 0, (
        "offset params were dropped from the upstream task ABI; pass narrow "
        "views instead")
    params = [
        process_dim,
        1 if scale_ue8m0 else 0,
        1 if emit_bf16 else 0,
    ]
    pk.kn_graph.register_task(
        tb_graph, "fused_rmsnorm_quantize_fp8_sm100", params)


def mla_kv_append_layer(
    pk,
    c_latent_new: DTensor,
    k_pe_new: DTensor,
    kv_buf: DTensor,
    mla_params: tuple,
    grid_dim: tuple,
    block_dim: tuple,
    c_latent_row_stride: int = None,
    k_pe_row_stride: int = None,
):
    """bs=1 contiguous KV append (no page table).

    Writes the new token rows' [c_latent(D_V) | k_pe(D_K-D_V)] into the
    per-layer contiguous KV buffer at row = sequence position (single
    sequence => logical position == physical row). Replaces the paged-cache
    append + page gather; the MLA decode kernels read ``kv_buf`` directly via
    their contiguous branch. ``kv_buf`` is tracked as the task output so the
    decode task gets a same-iteration dependency edge.
    """
    d_k, d_v = mla_params
    params = [
        d_k, d_v,
        c_latent_row_stride if c_latent_row_stride is not None else d_v,
        k_pe_row_stride if k_pe_row_stride is not None else 128,
    ]
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(c_latent_new, (-1, -1, -1), -1, True)
    tb_graph.new_input(k_pe_new, (-1, -1, -1), -1, True)
    tb_graph.new_input(kv_buf, (-1, -1, -1), -1, True)
    pk.kn_graph.customized([c_latent_new, k_pe_new, kv_buf], tb_graph)
    pk.kn_graph.register_task(tb_graph, "mla_kv_append_sm100", params)


def deepseek_mla_rope_q_layer(
    pk,
    q_nope_pe: DTensor,
    q_pe: DTensor,
    cos_pos_embed: DTensor,
    sin_pos_embed: DTensor,
    num_heads: int,
    has_split_q: bool,
    grid_dim: tuple,
    block_dim: tuple = (128, 1, 1),
    q_tile_size: int = 16,
):
    params = [num_heads, q_tile_size, 1 if has_split_q else 0]
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    # Duplicate Q tensors are used as task outputs. This gives downstream
    # MLA tasks a real dependency on the in-place RoPE write without
    # joining the independent K-RoPE dependency chain.
    tb_graph.new_input(q_nope_pe, (-1, -1, -1), -1, True)
    tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
    tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
    tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
    tb_graph.new_input(q_nope_pe, (-1, -1, -1), -1, True)
    tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
    pk.kn_graph.customized(
        [
            q_nope_pe,
            q_pe,
            cos_pos_embed,
            sin_pos_embed,
            q_nope_pe,
            q_pe,
        ],
        tb_graph,
    )
    pk.kn_graph.register_task(tb_graph, "deepseek_mla_rope_q_sm100", params)


def deepseek_mla_rope_q_fused_layer(
    pk,
    q_nope_pe: DTensor,
    cos_pos_embed: DTensor,
    sin_pos_embed: DTensor,
    num_heads: int,
    grid_dim: tuple,
    block_dim: tuple = (128, 1, 1),
    q_tile_size: int = 16,
    phase_gate: int = 0,
):
    # phase_gate=2 (decode-only): codegen emits a
    # `if (Q_LEN > 8) return;` gate so the kernel skips the rotation
    # on prefill iters where q_nope_pe is stale (the absorbed q_b
    # decode GEMM early-exits via gate_mode=2 on prefill iters).
    params = [num_heads, q_tile_size]
    if phase_gate != 0:
        params.append(phase_gate)
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(q_nope_pe, (-1, -1, -1), -1, True)
    tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
    tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
    tb_graph.new_input(q_nope_pe, (-1, -1, -1), -1, True)
    pk.kn_graph.customized(
        [q_nope_pe, cos_pos_embed, sin_pos_embed, q_nope_pe],
        tb_graph,
    )
    pk.kn_graph.register_task(
        tb_graph, "deepseek_mla_rope_q_fused_sm100", params)


def deepseek_mla_rope_q_split_layer(
    pk,
    q_pe: DTensor,
    cos_pos_embed: DTensor,
    sin_pos_embed: DTensor,
    num_heads: int,
    grid_dim: tuple,
    block_dim: tuple = (128, 1, 1),
    q_tile_size: int = 16,
    qfused_mode: int = 0,
    phase_gate: int = 0,
):
    # qfused_mode = 0: q_pe is a standalone (mbt, num_heads*64) tensor.
    # qfused_mode = 1: q_pe is the same DTensor as the fused q_b_prefill
    # buffer (mbt, num_heads*192) with row-swap layout. Kernel uses
    # row_stride = num_heads*192 and pe_base_in_row = num_heads*128.
    # phase_gate=1 (prefill-only): codegen emits a
    # `if (Q_LEN <= 8) return;` gate so the kernel skips the rotation
    # on decode iters where the q_b_prefill_fused buffer is stale
    # (the unabsorbed q_b prefill GEMM early-exits via gate_mode=1
    # on decode iters; chunked_prefill itself returns early too).
    params = [num_heads, q_tile_size]
    # The codegen reads phase_gate at params[3], so qfused_mode (params[2])
    # must be present when phase_gate is set, even if 0.
    if phase_gate != 0:
        params.append(qfused_mode)
        params.append(phase_gate)
    elif qfused_mode != 0:
        params.append(qfused_mode)
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
    tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
    tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
    tb_graph.new_input(q_pe, (-1, -1, -1), -1, True)
    pk.kn_graph.customized(
        [q_pe, cos_pos_embed, sin_pos_embed, q_pe],
        tb_graph,
    )
    pk.kn_graph.register_task(
        tb_graph, "deepseek_mla_rope_q_split_sm100", params)


def deepseek_mla_rope_k_layer(
    pk,
    k_pe: DTensor,
    cos_pos_embed: DTensor,
    sin_pos_embed: DTensor,
    grid_dim: tuple,
    block_dim: tuple = (128, 1, 1),
    q_tile_size: int = 16,
    k_pe_row_stride: int = None,
    k_pe_offset: int = 0,
):
    # k_pe_row_stride supports running the K_PE rotation in-place on a slice
    # of a wider buffer (e.g., qkv_a_out (mbt, 2176) where k_pe lives at cols
    # [2048:2112)). The C++ register on this branch reads [q_tile] or
    # [q_tile, k_pe_row_stride] — the slice offset comes from the narrow
    # view's base pointer, NOT a param.
    assert k_pe_offset == 0, (
        "k_pe_offset was dropped from the upstream task ABI; pass a narrow "
        "view instead")
    params = [q_tile_size]
    if k_pe_row_stride is not None:
        params = [q_tile_size, k_pe_row_stride]
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(k_pe, (-1, -1, -1), -1, True)
    tb_graph.new_input(cos_pos_embed, (-1, -1, -1), -1, True)
    tb_graph.new_input(sin_pos_embed, (-1, -1, -1), -1, True)
    tb_graph.new_input(k_pe, (-1, -1, -1), -1, True)
    pk.kn_graph.customized(
        [
            k_pe,
            cos_pos_embed,
            sin_pos_embed,
            k_pe,
        ],
        tb_graph,
    )
    pk.kn_graph.register_task(tb_graph, "deepseek_mla_rope_k_sm100", params)


def mla_mtp_decode_layer(
    pk, q_input, kv_input, output_partial, output_lse, q_len, kv_len,
    tp_size: int, num_splits_override=None,
):
    """Unified MLA MTP decode entry — dispatches on tp_size to the
    TP1/TP2/TP4/TP8 task variants (registered task names unchanged).

    tp_size: the tensor-parallel world size (1/2/4/8). Selects the
        per-rank head count (128/64/32/16) variant.
    num_splits_override: KV-split override for the TP variants. The TP1
        kernel derives its split count internally and has no override
        plumbing (callers historically never passed one at TP1).
    For tp_size == 8, q_len is the REAL (unpadded) Q_LEN; the TP8
    variant pads it to even internally.
    """
    if tp_size == 1:
        pk.mla_mtp_decode_layer(
            q_input, kv_input, output_partial, output_lse, q_len, kv_len)
    elif tp_size == 2:
        pk.mla_mtp_decode_tp2_layer(
            q_input, kv_input, output_partial, output_lse, q_len, kv_len,
            num_splits_override=num_splits_override)
    elif tp_size == 4:
        pk.mla_mtp_decode_tp4_layer(
            q_input, kv_input, output_partial, output_lse, q_len, kv_len,
            num_splits_override=num_splits_override)
    elif tp_size == 8:
        pk.mla_mtp_decode_tp8_layer(
            q_input, kv_input, output_partial, output_lse, q_len, kv_len,
            num_splits_override=num_splits_override)
    else:
        raise ValueError(
            f"mla_mtp_decode_layer: unsupported tp_size {tp_size} "
            "(expected 1, 2, 4, or 8)")


def mla_mtp_reduce_layer(
    pk, input_partial, input_lse, output, q_len, kv_len,
    tp_size: int,
):
    """Unified MLA MTP decode-reduce entry — the tp_size-dispatching
    companion of mla_mtp_decode_layer (registered task names unchanged).
    For tp_size == 8, q_len is the REAL (unpadded) Q_LEN.
    """
    if tp_size == 1:
        pk.mla_mtp_reduce_layer(
            input_partial, input_lse, output, q_len, kv_len)
    elif tp_size == 2:
        pk.mla_mtp_decode_tp2_reduce_layer(
            input_partial, input_lse, output, q_len, kv_len)
    elif tp_size == 4:
        pk.mla_mtp_decode_tp4_reduce_layer(
            input_partial, input_lse, output, q_len, kv_len)
    elif tp_size == 8:
        pk.mla_mtp_decode_tp8_reduce_layer(
            input_partial, input_lse, output, q_len, kv_len)
    else:
        raise ValueError(
            f"mla_mtp_reduce_layer: unsupported tp_size {tp_size} "
            "(expected 1, 2, 4, or 8)")


def _fp8_group_gemm_layer_impl(
    pk,
    task_name: str,
    a_fp8: DTensor,
    b_fp8: DTensor,
    sfa_packed: DTensor,
    sfb_packed: DTensor,
    m_indices: DTensor,
    output: DTensor,
    num_workers: int,
    meta: DTensor = None,
):
    """Shared registration helper for the SM100 grouped FP8 block-scaled
    GEMM tasks (`fp8_group_gemm_smallm_sm100` / `fp8_group_gemm_largem_sm100`).

    Computes  D[r, :] = (A[r, :] * scale_a[r]) @ (B[m_indices[r]].T * scale_b)
    with hardware UE8M0 dequant via `tcgen05.mma.kind::mxf8f6f4.block_scale`.
    Rows in each BM=128 block must share the same expert id.

    Shape symbols
    -------------
        M_total : total number of rows across all experts (must be a
                  multiple of BM=128; pad-rows can carry a dummy expert).
        K       : reduction dim (must be a multiple of BK=128).
        N       : per-expert output dim.
        E       : number of experts.
        nk       = ceil(K / 128)              UE8M0 scales per row.
        num_sf_k = ceil(nk / 4)               uint32-packed scale columns
                                               (4 UE8M0 per uint32 along K).

    DTensor inputs / output
    -----------------------
    a_fp8       (M_total, K)            fp8_e4m3 (attached as uint8)
                row-major, K innermost. Activations (already permuted so
                that contiguous BM=128 row-blocks share one expert).

    b_fp8       (E, N, K)               fp8_e4m3 (attached as uint8)
                row-major per expert (K innermost). The kernel flattens
                the buffer to (E*N, K) for its TMA descriptor; same memory.

    sfa_packed  (num_sf_k, M_total)     uint32, UE8M0-packed
                Row-major with M_total innermost (PyTorch shape order;
                same memory the kernel's TMA descriptor describes with
                g=(M_total, num_sf_k) in its innermost-first convention).
                Each uint32 packs 4 consecutive UE8M0 scales along the
                K-block axis (one scale per 128-K-element block per row).

    sfb_packed  (num_sf_k, E*N)         uint32, UE8M0-packed
                Same packing convention as SFA. Built by expanding the
                per-expert per-block scale [E, N/128, K/128] →
                [E*N, K/128] (repeat_interleave along N) → pack to
                [num_sf_k, E*N] uint32. One scale per output element per
                128-K-element block (after expansion).

    m_indices   (M_total,)              int32
                Expert id per A row. Rows in [bm*BM, (bm+1)*BM) for any
                bm must share the same expert (only m_indices[bm*BM] is
                read per block). For static permuted layouts this is
                typically `arange(M_total) // BM_PADDING`.

    output      (M_total, N)            bf16
                Row-major, N innermost. Written via TMA store.

    Other params
    ------------
    task_name   : "fp8_group_gemm_smallm_sm100" (BN=64, NS=8) or
                  "fp8_group_gemm_largem_sm100" (BN=128, NS=6); picks the
                  tile/stage variant. Dispatch policy lives in
                  `fp8_group_gemm_layer`.
    num_workers : grid_dim.x. Each task instance handles a stride of
                  (bm, bn) tiles `task_desc.task_metadata.request_id ::
                  num_workers`; pick `pk.num_workers` so every worker
                  gets a slice.

    Partitioning
    ------------
    All six tensors are registered with input_map (-1,-1,-1): every task
    gets the full base pointer. Tile selection is internal to the kernel
    (driven by worker_idx + num_workers), not by MPK's TBGraph slicer.
    block_dim is fixed at (256, 1, 1) — 8 warps with hard-coded roles
    (TMA-load / UTCCP-transpose / MMA-issue / epilogue+TMA-store).
    """
    assert a_fp8.num_dims == 2
    assert b_fp8.num_dims == 3
    assert output.num_dims == 2
    M_total = a_fp8.dim(0)
    K = a_fp8.dim(1)
    E = b_fp8.dim(0)
    N = b_fp8.dim(1)
    assert b_fp8.dim(2) == K
    assert m_indices.dim(0) == M_total
    if meta is None:
        active_mask_offset = -1
    else:
        assert meta.num_dims == 2
        # meta layout: row 0 = out_weights+tok_to_perm (length M_total+MBT*TOPK).
        # Row 1's first E entries hold active_expert_mask (int32).
        # Flat offset of row 1: meta.dim(1) (since row 0 occupies that).
        active_mask_offset = meta.dim(1)
    params = [M_total, N, K, E, num_workers, active_mask_offset]
    grid_dim = (num_workers, 1, 1)
    block_dim = (256, 1, 1)  # 8 warps fixed by kernel role layout
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(a_fp8, (-1, -1, -1), -1, True)
    tb_graph.new_input(b_fp8, (-1, -1, -1), -1, True)
    tb_graph.new_input(sfa_packed, (-1, -1, -1), -1, True)
    tb_graph.new_input(sfb_packed, (-1, -1, -1), -1, True)
    tb_graph.new_input(m_indices, (-1, -1, -1), -1, True)
    operators = [a_fp8, b_fp8, sfa_packed, sfb_packed, m_indices]
    # CRITICAL ORDERING: the codegen reads
    # input_ptrs[5] as the meta/active-expert-mask buffer
    # (register_fp8_group_gemm_variant: "active_mask_offset >= 0 means
    # input_ptrs[5] is the meta buffer") and output_ptrs[0] as the D
    # output; graph.cc sets the tuple to (num_inputs = 6 if meta else 5,
    # 1 output). meta MUST therefore be registered BEFORE output so the
    # positional split gives input[5]=meta, output[0]=D.
    # The earlier order [..., m_indices, output, meta] put `output` at
    # input[5] (read as the active mask -> garbage -> num_active=0 -> the
    # kernel exits writing NOTHING) and `meta` in the output slot (the D
    # TMA-store goes to the tiny meta buffer, dropped) -> the entire
    # active-skip MoE W13/W2 GEMM produced NULL output. Same bug class as
    # the moe_silu_mul "CRITICAL ORDERING" fix; the grouped-GEMM path never
    # got the analog.
    # meta=None path (non-active-skip) is unchanged (5 inputs + output).
    if meta is not None:
        tb_graph.new_input(meta, (-1, -1, -1), -1, True)
        operators.append(meta)
    tb_graph.new_input(output, (-1, -1, -1), -1, True)
    operators.append(output)
    pk.kn_graph.customized(operators, tb_graph)
    pk.kn_graph.register_task(tb_graph, task_name, params)


def _fp8_group_gemm_smallm_layer(
    pk, a_fp8, b_fp8, sfa_packed, sfb_packed, m_indices, output,
    num_workers, meta=None,
):
    # Smallm variant: BN=64, NS=8. Best for K>4096 && MPE<=8 (gate_up
    # M{1,4,8} on DSv3). MoE decode niche.
    _fp8_group_gemm_layer_impl(
        pk, "fp8_group_gemm_smallm_sm100",
        a_fp8, b_fp8, sfa_packed, sfb_packed, m_indices, output,
        num_workers, meta=meta)


def _fp8_group_gemm_largem_layer(
    pk, a_fp8, b_fp8, sfa_packed, sfb_packed, m_indices, output,
    num_workers, meta=None,
):
    # Largem variant: BN=128, NS=6. Default for everything outside the
    # smallm niche (most MoE configs incl. all prefill MPE >= 16 and any
    # K <= 4096 layer like down_proj).
    _fp8_group_gemm_layer_impl(
        pk, "fp8_group_gemm_largem_sm100",
        a_fp8, b_fp8, sfa_packed, sfb_packed, m_indices, output,
        num_workers, meta=meta)


def fp8_group_gemm_layer(
    pk, a_fp8, b_fp8, sfa_packed, sfb_packed, m_indices, output,
    num_workers, meta=None,
):
    # Public family entry. Auto-dispatcher: pick the smallm/largem tile
    # variant by (K, M_per_expert).
    K = a_fp8.dim(1)
    M_total = a_fp8.dim(0)
    E = b_fp8.dim(0)
    MPE = M_total // E
    if K > 4096 and MPE <= 8:
        _fp8_group_gemm_smallm_layer(
            pk, a_fp8, b_fp8, sfa_packed, sfb_packed, m_indices, output,
            num_workers, meta=meta)
    else:
        _fp8_group_gemm_largem_layer(
            pk, a_fp8, b_fp8, sfa_packed, sfb_packed, m_indices, output,
            num_workers, meta=meta)


def moe_permute_sm100_layer(
    pk,
    input_fp8: DTensor,
    input_scale: DTensor,
    topk_weights: DTensor,
    routing_indices: DTensor,
    permuted_fp8: DTensor,
    permuted_scale: DTensor,
    meta: DTensor,
    bm_padding: int = 128,
    e_per_cta: int = 1,
):
    """MoE expand-permute-sort task — peripheral glue for the PR-674
    grouped FP8 GEMM. See moe_permute_sm100.cuh for the exact contract.

    By default one CTA per local expert (grid_dim = (E_local, 1, 1)).
    `e_per_cta` (the DSv3 builder passes 4) lets each CTA own
    E_PER_CTA consecutive experts, shrinking the
    launch to (E_local / E_PER_CTA, 1, 1). This collapses the decode
    "permute valley" (128 CTAs vs ~8 active experts contending with the
    shared-expert GEMM). E_PER_CTA==1 is byte-identical to the legacy
    path. Scans routing_indices[expert, :], gathers matched tokens,
    and copies
    FP8 row + UE8M0-packed scale into the permuted layout. Small
    per-row metadata (permuted_weights + token_to_permuted) is packed
    into one int32 `meta` buffer so the task stays within MPK's
    3-outputs-per-task limit:

      meta[0       : M_TOTAL]            = permuted_weights (f32 bits)
      meta[M_TOTAL : M_TOTAL + MBT*TOPK] = token_to_permuted (row + 1;
                                              0 = not routed locally;
                                              caller must tensor_init
                                              zero this region each
                                              iter).

    `m_indices` is a STATIC constant the builder sets up once via
    attach_input (pattern: m_indices[r] = r / BM_PADDING). It is fed
    directly to the grouped FP8 GEMM and is NOT a per-iter output.

    IMPORTANT: input_scale must be UE8M0-PACKED uint32 (produced by
    quantize_fp8_layer with scale_ue8m0=True).
    """
    assert input_fp8.num_dims == 2
    assert input_scale.num_dims == 2
    assert topk_weights.num_dims == 2
    assert routing_indices.num_dims == 2
    assert permuted_fp8.num_dims == 2
    assert permuted_scale.num_dims == 2
    # meta is shaped (2, M_TOTAL + MBT*TOPK) int32 — see builder.py for
    # the BATCH_SIZE=2 rationale (full-byte tensor_init).
    assert meta.num_dims == 2
    assert meta.dim(0) == 2

    K = input_fp8.dim(1)
    # K_PACKED derives from K (128-wide groups, 4 UE8M0 bytes per uint32):
    # the scale buffer is K-outer [K_PACKED, round4(MBT)] memory, but callers
    # may attach it under a transposed logical shape.
    K_PACKED = ((K + 127) // 128 + 3) // 4
    MBT = input_fp8.dim(0)
    TOPK = topk_weights.dim(1)
    E_LOCAL = routing_indices.dim(0)
    M_TOTAL = E_LOCAL * bm_padding
    assert routing_indices.dim(1) == MBT
    assert topk_weights.dim(0) == MBT
    assert permuted_fp8.dim(0) == M_TOTAL
    assert permuted_fp8.dim(1) == K
    assert permuted_scale.dim(0) == K_PACKED
    assert permuted_scale.dim(1) == M_TOTAL
    assert meta.dim(1) == M_TOTAL + MBT * TOPK, (
        f"meta length must be {M_TOTAL + MBT * TOPK}, got {meta.dim(1)}")

    assert e_per_cta >= 1, "e_per_cta must be >= 1"
    assert E_LOCAL % e_per_cta == 0, (
        f"E_LOCAL ({E_LOCAL}) must be divisible by e_per_cta "
        f"({e_per_cta})")
    params = [K, K_PACKED, MBT, TOPK, E_LOCAL, bm_padding, e_per_cta]
    # E_PER_CTA experts per CTA → (E_LOCAL / E_PER_CTA) CTAs. Each CTA
    # derives its expert range from task_metadata.expert_offset (= bid.x,
    # the CTA index) inside the kernel.
    grid_dim = (E_LOCAL // e_per_cta, 1, 1)
    block_dim = (128, 1, 1)
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(input_fp8, (-1, -1, -1), -1, True)
    tb_graph.new_input(input_scale, (-1, -1, -1), -1, True)
    tb_graph.new_input(topk_weights, (-1, -1, -1), -1, True)
    # routing_indices: (-1, -1, -1) so the kernel sees the FULL (E_LOCAL, MBT)
    # buffer and computes its expert row from task_metadata.expert_offset.
    tb_graph.new_input(routing_indices, (-1, -1, -1), -1, True)
    tb_graph.new_input(permuted_fp8, (-1, -1, -1), -1, True)
    tb_graph.new_input(permuted_scale, (-1, -1, -1), -1, True)
    tb_graph.new_input(meta, (-1, -1, -1), -1, True)
    pk.kn_graph.customized(
        [input_fp8, input_scale, topk_weights, routing_indices,
         permuted_fp8, permuted_scale, meta], tb_graph)
    pk.kn_graph.register_task(tb_graph, "moe_permute_sm100", params)


def moe_unpermute_sm100_layer(
    pk,
    permuted_output: DTensor,
    meta: DTensor,
    residual: DTensor,
    output: DTensor,
    rows_per_cta: int = 8,
    hidden_split: int = 1,
):
    """MoE combine-unpermute task — inverse of moe_permute_sm100. See
    moe_unpermute_sm100.cuh for the contract. Decodes `meta` into
    permuted_weights + token_to_permuted, then writes
    `output[t] = residual[t] +
                 sum_k(permuted_output[token_to_permuted[t,k]-1]
                        * permuted_weights[same row])`.

    grid_dim = (ceil(MBT / rows_per_cta), 1, 1). The
    kernel's ROWS_PER_TASK template (moe_unpermute_sm100.cuh) loops
    `ceil(MBT / grid.x)` tokens per CTA, so each CTA handles
    rows_per_cta consecutive tokens. Default rows_per_cta=8 gives 16
    CTAs for MBT=128 (vs 128 CTAs at rows_per_cta=1), freeing 112
    worker slots per unpermute wave for concurrent tasks. For
    decode (active_rows=1) only CTA 0 does work; the rest pass the
    my_token >= num_active_rows check and exit immediately, same as
    before. Setting rows_per_cta=1 preserves the legacy 1-CTA-per-
    token shape. The codegen recomputes ROWS_PER_TASK from grid.x so
    this kwarg only affects launch fan-out, not correctness.
    """
    assert permuted_output.num_dims == 2
    # meta is shaped (2, M_TOTAL + MBT*TOPK) int32 — see
    # moe_permute_sm100_layer for the layout contract.
    assert meta.num_dims == 2
    assert meta.dim(0) == 2
    assert residual.num_dims == 2
    assert output.num_dims == 2

    MBT = residual.dim(0)
    HIDDEN = permuted_output.dim(1)
    M_TOTAL = permuted_output.dim(0)
    # meta = M_TOTAL (weights) + MBT*TOPK (token_to_permuted) entries.
    meta_len = meta.dim(1)
    TOPK = (meta_len - M_TOTAL) // MBT
    assert M_TOTAL + MBT * TOPK == meta_len
    assert residual.dim(1) == HIDDEN
    assert output.dim(0) == MBT
    assert output.dim(1) == HIDDEN

    params = [MBT, TOPK, HIDDEN, M_TOTAL]
    rows_per_cta_safe = max(1, int(rows_per_cta))
    grid_x = max(1, (MBT + rows_per_cta_safe - 1) // rows_per_cta_safe)
    # Stragglers fix: grid.y = hidden_split spreads each
    # token's HIDDEN work across hidden_split CTAs. For decode
    # (active_rows=1) only 1*hidden_split CTAs do work — bumping
    # hidden_split splits the 32 μs per-token straggler across
    # more SMs concurrently. task_register passes hidden_split as
    # the kernel's HIDDEN_SPLIT template and bid.y becomes the
    # partition index (kv_idx). HIDDEN must be divisible by
    # hidden_split for clean partitions; the kernel rounds up
    # via ceil-div and clamps the upper partition to HIDDEN.
    hidden_split_safe = max(1, int(hidden_split))
    grid_dim = (grid_x, hidden_split_safe, 1)
    block_dim = (128, 1, 1)
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    # All inputs/outputs are (-1, -1, -1) so the kernel sees the FULL
    # tensors and indexes them with task_metadata.request_id (= task_idx).
    # task_idx * ROWS_PER_TASK + r is the per-CTA token id (kernel-side).
    tb_graph.new_input(permuted_output, (-1, -1, -1), -1, True)
    tb_graph.new_input(meta, (-1, -1, -1), -1, True)
    tb_graph.new_input(residual, (-1, -1, -1), -1, True)
    tb_graph.new_input(output, (-1, -1, -1), -1, True)
    pk.kn_graph.customized(
        [permuted_output, meta, residual, output], tb_graph)
    pk.kn_graph.register_task(tb_graph, "moe_unpermute_sm100", params)


def linear_fp8_swapAB_layer(
    pk,
    input_fp8: DTensor,
    input_scale: DTensor,
    weight_fp8: DTensor,
    weight_scale: DTensor,
    output: DTensor,
    grid_dim: tuple,
    block_dim: tuple,
    gate_mode: int = 0,
):
    # MPK-native FP8 linear (swapAB inside the kernel). Same Python-layer
    # API as linear_fp8_layer; the kernel maps weight->A and input->B.
    # Constraints (asserted at registration time):
    #   per-task output size (output.dim[1] / grid_dim.x) must be a
    #   multiple of 128, and batch_size must be <= 16 (decode-only).
    params = [] if gate_mode == 0 else [gate_mode]
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(input_fp8, (-1, -1, -1), -1, True)
    tb_graph.new_input(input_scale, (-1, -1, -1), -1, True)
    tb_graph.new_input(weight_fp8, (0, -1, -1), -1, True)
    tb_graph.new_input(weight_scale, (0, -1, -1), -1, True)
    tb_graph.new_input(output, (1, -1, -1), -1, True)
    pk.kn_graph.customized(
        [input_fp8, input_scale, weight_fp8, weight_scale, output], tb_graph)
    pk.kn_graph.register_task(tb_graph, "linear_fp8_swapAB_sm100", params)


def linear_fp8_swapAB_with_residual_layer(
    pk,
    input_fp8: DTensor,
    input_scale: DTensor,
    weight_fp8: DTensor,
    weight_scale: DTensor,
    residual: DTensor,
    output: DTensor,
    grid_dim: tuple,
    block_dim: tuple,
    gate_mode: int = 0,
):
    params = [1] if gate_mode == 0 else [1, gate_mode]
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(input_fp8, (-1, -1, -1), -1, True)
    tb_graph.new_input(input_scale, (-1, -1, -1), -1, True)
    tb_graph.new_input(weight_fp8, (0, -1, -1), -1, True)
    tb_graph.new_input(weight_scale, (0, -1, -1), -1, True)
    tb_graph.new_input(residual, (1, -1, -1), -1, True)
    tb_graph.new_input(output, (1, -1, -1), -1, True)
    pk.kn_graph.customized(
        [input_fp8, input_scale, weight_fp8, weight_scale, residual, output],
        tb_graph)
    pk.kn_graph.register_task(
        tb_graph, "linear_fp8_swapAB_with_residual_sm100", params)


def assemble_q_decode_sm100_layer(
    pk,
    q_nope_abs: DTensor,
    q_pe: DTensor,
    q_nope_pe: DTensor,
    grid_dim: tuple,
    block_dim: tuple = (128, 1, 1),
    pe_only: bool = False,
):
    """Interleave the BMM-absorbed q_nope (N, H, 512) with q_pe (N, H, 64)
    into per-head [nope|pe] layout (N, H, 576) for MLA decode.

    Used by the DSv3 decode Q path:
      rmsnorm_linear(q_a, q_b_nope) → q_nope (N, H, 128)
      quantize_fp8(q_nope)         → q_nope_fp8 (N, H, 128)
      linear_fp8_bmm_sm100(q_nope_fp8, kv_b_k_bmm) → q_nope_abs (N, H, 512)
      rmsnorm_linear(q_a, q_b_pe)  → q_pe (N, H, 64)
      assemble_q_decode_sm100(q_nope_abs, q_pe) → q_nope_pe (N, H, 576)

    Replaces the load-time absorbed q_b_proj GEMM. The BMM is per-head and
    loads smaller weights ((H, 512, 128) per head) vs the absorbed (H*576, q_lora)
    monolith, which is the perf win — smaller TMA traffic per task.

    grid_dim = (N, 1, 1); each CTA processes 1 token (all H heads).
    block_dim = (128, 1, 1) is plenty: at TP=4 (H=32) each CTA writes
    32*576 = 18432 bf16 elements = 144 elements/thread.
    """
    assert q_nope_abs.num_dims == 3
    assert q_pe.num_dims == 3
    # q_nope_pe may be 3D (N, H, D_TOTAL) or 2D (N, H*D_TOTAL) — same
    # byte layout, the register code handles both. 2D is convenient so
    # the existing q_nope_pe buffer doesn't need to be reshaped.
    assert q_nope_pe.num_dims in (2, 3)
    assert q_nope_abs.dim(0) == q_pe.dim(0) == q_nope_pe.dim(0)
    H = q_nope_abs.dim(1)
    assert q_pe.dim(1) == H
    D_TOTAL = q_nope_abs.dim(2) + q_pe.dim(2)
    if q_nope_pe.num_dims == 3:
        assert q_nope_pe.dim(1) == H
        assert q_nope_pe.dim(2) == D_TOTAL
    else:
        assert q_nope_pe.dim(1) == H * D_TOTAL
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(q_nope_abs, (0, -1, -1), -1, True)
    tb_graph.new_input(q_pe,       (0, -1, -1), -1, True)
    tb_graph.new_input(q_nope_pe,  (0, -1, -1), -1, True)
    pk.kn_graph.customized([q_nope_abs, q_pe, q_nope_pe], tb_graph)
    params = [1] if pe_only else []
    pk.kn_graph.register_task(tb_graph, "assemble_q_decode_sm100", params)


def _linear_fp8_bmm_sm100_layer(
    pk,
    input_fp8: DTensor,
    input_scale: DTensor,
    weight_fp8: DTensor,
    weight_scale: DTensor,
    output: DTensor,
    grid_dim: tuple,    # (m_shards_per_head, h_shards, 1)
    block_dim: tuple,   # (256, 1, 1) on SM100
):
    # Per-head FP8 batched matmul on SM100. Computes
    #     output[n, h, :] = input[n, h, :] @ weight[h, :, :]^T  (per head)
    # decode-only, batch_size <= 16. The H dimension is exposed as an
    # explicit workload split (grid.y) on top of the existing swapAB
    # M-tile split (grid.x). First cut requires grid.y == H — one head
    # per CTA — so the kernel stays a thin forward to the swapAB GEMM.
    #
    # Tensor layouts (all 3D; dim 1 is the head axis):
    #   input_fp8     [N, H, D_in]
    #   input_scale   [N, H, packed_K]   uint32 UE8M0 (4 logical scales / uint32)
    #   weight_fp8    [H, D_out, D_in]
    #   weight_scale  [H, D_out, packed_K]
    #   output        [N, H, D_out]
    #
    # Constraints (asserted at registration time):
    #   - D_out / grid.x must be a multiple of MMA_M=128
    #   - D_in must be a multiple of BLOCK_K=128
    #   - batch_size N <= MMA_N=16 (decode-only)
    #   - H % grid.y == 0; first cut requires H_PER_TASK == 1
    # Weight stays 3D (H, D_out, D_in) — the per-head TMA stride depends
    # on the explicit H dim. Input/output may be 2D (N, H*D_*) or 3D
    # (N, H, D_*); same byte layout, partition map adjusts the dim index.
    assert weight_fp8.num_dims == 3
    assert weight_scale.num_dims == 3
    assert input_fp8.num_dims in (2, 3)
    assert input_scale.num_dims in (2, 3)
    assert output.num_dims in (2, 3)
    params = []
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    in_h_axis = 1 if input_fp8.num_dims == 3 else 1
    in_sc_h_axis = 1 if input_scale.num_dims == 3 else 1
    out_h_axis = 1
    out_m_axis = 2 if output.num_dims == 3 else 1
    # input_fp8 / input_scale: grid.y splits the head axis. For 3D
    # (N, H, D_in), head axis is dim 1; for 2D (N, H*D_in), head
    # axis is also dim 1 because the partition map's axis index
    # refers to the DTensor's dim sequence; dim 1 still partitions
    # into H equal slices of D_in each (per-CTA STensor.dim[1] = D_in).
    tb_graph.new_input(input_fp8,    (-1, in_h_axis, -1), -1, True)
    tb_graph.new_input(input_scale,  (-1, in_sc_h_axis, -1), -1, True)
    # weight_fp8 / weight_scale [H, D_out, D_in or packed_K]:
    # grid.x splits dim 1 (D_out), grid.y splits dim 0 (H).
    tb_graph.new_input(weight_fp8,   (1, 0, -1), -1, True)
    tb_graph.new_input(weight_scale, (1, 0, -1), -1, True)
    # output: dim 1 (H) split by grid.y. For 3D, dim 2 (D_out) split
    # by grid.x; for 2D, dim 1 (H*D_out) is also split — but the
    # partition needs the SAME dim for both H and D_out splits, which
    # only works in 3D form. For 2D output, grid.x must be 1.
    if output.num_dims == 3:
        tb_graph.new_input(output, (out_m_axis, out_h_axis, -1), -1, True)
    else:
        assert grid_dim[0] == 1, (
            "linear_fp8_bmm with 2D output requires grid.x=1 "
            "(D_out cannot be sharded across CTAs when packed flat)")
        tb_graph.new_input(output, (-1, 1, -1), -1, True)
    pk.kn_graph.customized(
        [input_fp8, input_scale, weight_fp8, weight_scale, output], tb_graph)
    pk.kn_graph.register_task(tb_graph, "linear_fp8_bmm_sm100", params)


def _linear_fp8_bmm_dense_sm100_layer(
    pk,
    input_fp8: DTensor,
    input_scale: DTensor,
    weight_fp8: DTensor,
    weight_scale: DTensor,
    output: DTensor,
    grid_dim: tuple,    # (1, h_shards, 1)  (grid.x must be 1: D_out=128=BN)
    block_dim: tuple,   # (256, 1, 1) on SM100
):
    # Per-head FP8 batched matmul wrapping the DENSE block-scaled GEMM body
    # (float32 scales) instead of swapAB (UE8M0). Computes
    #     output[n, h, :] = input[n, h, :] @ weight[h, :, :]^T  (per head)
    # decode-only, batch_size <= 16, one head per CTA (grid.y == H).
    #
    # Alternative to the swapAB BMM body for the DSv3 decode BMM2 (o-down
    # un-absorption): the float32 128-K-aligned block scales are
    # split-K-friendly, whereas swapAB's UE8M0 packs at 512-K and cannot
    # split a per-head K=512. Same math, different scale encoding.
    #
    # Tensor layouts (all 3D; dim 1 is the head axis for activation):
    #   input_fp8     [N, H, D_in]
    #   input_scale   [N, H, nk]          float32 (nk = D_in / 128)
    #   weight_fp8    [H, D_out, D_in]
    #   weight_scale  [H, D_out/128, nk]  float32 (D_out=128 -> dim1 = 1)
    #   output        [N, H, D_out]       (2D [N, H*D_out] also accepted)
    #
    # Constraints (asserted at registration time):
    #   - D_out (per head, = N) must be a multiple of BN=128 -> grid.x == 1
    #   - D_in must be a multiple of BK=128
    #   - batch_size N <= 16 (decode-only)
    #   - H % grid.y == 0; first cut requires H_PER_TASK == 1 (grid.y == H)
    assert weight_fp8.num_dims == 3
    assert weight_scale.num_dims == 3
    assert input_fp8.num_dims == 3, (
        "linear_fp8_bmm_dense requires 3D input [N, H, D_in]")
    assert input_scale.num_dims == 3, (
        "linear_fp8_bmm_dense requires 3D float32 input_scale [N, H, nk]")
    assert output.num_dims in (2, 3)
    assert grid_dim[0] == 1, (
        "linear_fp8_bmm_dense requires grid.x == 1 (per-head D_out=128=BN)")
    params = []
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    # input_fp8 / input_scale: grid.y splits the head axis (dim 1).
    tb_graph.new_input(input_fp8,   (-1, 1, -1), -1, True)
    tb_graph.new_input(input_scale, (-1, 1, -1), -1, True)
    # weight_fp8 / weight_scale [H, ...]: grid.y splits dim 0 (H).
    # grid.x == 1, so dim 1 (D_out) is not sharded.
    tb_graph.new_input(weight_fp8,   (-1, 0, -1), -1, True)
    tb_graph.new_input(weight_scale, (-1, 0, -1), -1, True)
    # output: dim 1 (H) split by grid.y. grid.x == 1 so D_out unsharded.
    if output.num_dims == 3:
        tb_graph.new_input(output, (-1, 1, -1), -1, True)
    else:
        tb_graph.new_input(output, (-1, 1, -1), -1, True)
    pk.kn_graph.customized(
        [input_fp8, input_scale, weight_fp8, weight_scale, output], tb_graph)
    pk.kn_graph.register_task(
        tb_graph, "linear_fp8_bmm_dense_sm100", params)


def linear_fp8_bmm_layer(
    pk,
    input_fp8: DTensor,
    input_scale: DTensor,
    weight_fp8: DTensor,
    weight_scale: DTensor,
    output: DTensor,
    grid_dim: tuple,
    block_dim: tuple,
    dense: bool,
):
    """Unified per-head FP8 BMM entry — dispatches on the scale encoding
    (registered task names unchanged).

    dense=False: swapAB body, UE8M0-packed uint32 scales (DSv3 decode
        BMM1, q_nope un-absorption; supports grid.x > 1 D_out shards).
    dense=True:  DENSE block-scaled body, float32 128-K-group scales
        (DSv3 decode BMM2, o-down un-absorption; requires grid.x == 1).
    """
    impl = (_linear_fp8_bmm_dense_sm100_layer if dense
            else _linear_fp8_bmm_sm100_layer)
    impl(
        pk,
        input_fp8=input_fp8,
        input_scale=input_scale,
        weight_fp8=weight_fp8,
        weight_scale=weight_scale,
        output=output,
        grid_dim=grid_dim,
        block_dim=block_dim,
    )


def _fp8_gemm_dense_layer_impl(
    pk,
    task_name: str,
    input_fp8: DTensor,
    weight_fp8: DTensor,
    input_scale: DTensor,
    weight_scale: DTensor,
    output: DTensor,
    num_workers: int,
    runtime_m_mode: int = 0,
):
    # A: [M,K], B: [N,K], C: [M,N]. The kernel distributes output tiles
    # across `num_workers` persistent tasks. Inputs/output may also be
    # 3D (M, H_split, K/H_split or D_out/H_split) when the caller wants
    # to keep the head dimension explicit downstream (e.g. for BMM); the
    # GEMM kernel itself sees the buffer as flat M*K / M*N bytes via TMA.
    assert input_fp8.num_dims in (2, 3)
    assert weight_fp8.num_dims == 2
    assert input_scale.num_dims == 2
    assert weight_scale.num_dims == 2
    assert output.num_dims in (2, 3)
    M = input_fp8.dim(0)
    K = (input_fp8.dim(1) if input_fp8.num_dims == 2
         else input_fp8.dim(1) * input_fp8.dim(2))
    N = weight_fp8.dim(0)
    assert weight_fp8.dim(1) == K
    assert output.dim(0) == M
    out_flat_n = (output.dim(1) if output.num_dims == 2
                  else output.dim(1) * output.dim(2))
    assert out_flat_n == N
    params = [M, N, K, num_workers]
    if runtime_m_mode:
        params.append(runtime_m_mode)
    tb_graph = TBGraph(CyTBGraph((num_workers, 1, 1), (256, 1, 1), 1, 64))
    tb_graph.new_input(input_fp8, (-1, -1, -1), -1, True)
    tb_graph.new_input(weight_fp8, (-1, -1, -1), -1, True)
    tb_graph.new_input(input_scale, (-1, -1, -1), -1, True)
    tb_graph.new_input(weight_scale, (-1, -1, -1), -1, True)
    tb_graph.new_input(output, (-1, -1, -1), -1, True)
    pk.kn_graph.customized(
        [input_fp8, weight_fp8, input_scale, weight_scale, output],
        tb_graph,
    )
    pk.kn_graph.register_task(tb_graph, task_name, params)


def _fp8_gemm_dense_smallm_layer(pk, input_fp8, weight_fp8, input_scale,
                                 weight_scale, output, num_workers,
                                 runtime_m_mode: int = 0):
    _fp8_gemm_dense_layer_impl(
        pk, "fp8_gemm_dense_smallm_sm100",
        input_fp8, weight_fp8, input_scale, weight_scale, output,
        num_workers, runtime_m_mode=runtime_m_mode)


def _fp8_gemm_dense_mediumm_layer(pk, input_fp8, weight_fp8, input_scale,
                                  weight_scale, output, num_workers,
                                  runtime_m_mode: int = 0):
    _fp8_gemm_dense_layer_impl(
        pk, "fp8_gemm_dense_mediumm_sm100",
        input_fp8, weight_fp8, input_scale, weight_scale, output,
        num_workers, runtime_m_mode=runtime_m_mode)


# Variants that fuse per-128-col-group UE8M0 quantize
# into the GEMM epilogue — output is FP8 + packed scale uint32 instead
# of bf16. Eliminates the downstream per_token_group_quantize_fp8 task
# in the BMM Q-up chain (q_b_nope_decode → quantize → BMM): we drop
# the quantize task and the BMM reads our FP8 + scale directly.
def _fp8_gemm_dense_fp8out_layer_impl(
    pk,
    task_name: str,
    input_fp8: DTensor,
    weight_fp8: DTensor,
    input_scale: DTensor,
    weight_scale: DTensor,
    output_fp8: DTensor,
    output_scale: DTensor,
    num_workers: int,
    runtime_m_mode: int = 0,
):
    # Same A/B/sa/sb input plumbing as the bf16 variant. Outputs are two
    # tensors (FP8 buf + packed uint32 scale); the bgraph attaches both
    # so the task tuple is (4 inputs, 2 outputs). Scale layout: flat
    # uint32 stride = N/128 entries per row (one per K-group), matching
    # what per_token_group_quantize_fp8 produces today for the BMM
    # input on the q_b_nope path.
    assert input_fp8.num_dims in (2, 3)
    assert weight_fp8.num_dims == 2
    assert input_scale.num_dims == 2
    assert weight_scale.num_dims == 2
    assert output_fp8.num_dims in (2, 3)
    assert output_scale.num_dims in (2, 3)
    M = input_fp8.dim(0)
    K = (input_fp8.dim(1) if input_fp8.num_dims == 2
         else input_fp8.dim(1) * input_fp8.dim(2))
    N = weight_fp8.dim(0)
    assert weight_fp8.dim(1) == K
    assert output_fp8.dim(0) == M
    out_flat_n = (output_fp8.dim(1) if output_fp8.num_dims == 2
                  else output_fp8.dim(1) * output_fp8.dim(2))
    assert out_flat_n == N, (out_flat_n, N)
    assert N % 128 == 0, (
        "fp8_gemm_dense_fp8out requires N divisible by 128: " + str(N))
    params = [M, N, K, num_workers]
    if runtime_m_mode:
        params.append(runtime_m_mode)
    tb_graph = TBGraph(CyTBGraph((num_workers, 1, 1), (256, 1, 1), 1, 64))
    tb_graph.new_input(input_fp8, (-1, -1, -1), -1, True)
    tb_graph.new_input(weight_fp8, (-1, -1, -1), -1, True)
    tb_graph.new_input(input_scale, (-1, -1, -1), -1, True)
    tb_graph.new_input(weight_scale, (-1, -1, -1), -1, True)
    tb_graph.new_input(output_fp8, (-1, -1, -1), -1, True)
    tb_graph.new_input(output_scale, (-1, -1, -1), -1, True)
    pk.kn_graph.customized(
        [input_fp8, weight_fp8, input_scale, weight_scale,
         output_fp8, output_scale],
        tb_graph,
    )
    pk.kn_graph.register_task(tb_graph, task_name, params)


def _fp8_gemm_dense_smallm_fp8out_layer(
    pk, input_fp8, weight_fp8, input_scale, weight_scale,
    output_fp8, output_scale, num_workers, runtime_m_mode: int = 0):
    _fp8_gemm_dense_fp8out_layer_impl(
        pk, "fp8_gemm_dense_smallm_fp8out_sm100",
        input_fp8, weight_fp8, input_scale, weight_scale,
        output_fp8, output_scale, num_workers,
        runtime_m_mode=runtime_m_mode)


def _fp8_gemm_dense_mediumm_fp8out_layer(
    pk, input_fp8, weight_fp8, input_scale, weight_scale,
    output_fp8, output_scale, num_workers, runtime_m_mode: int = 0):
    _fp8_gemm_dense_fp8out_layer_impl(
        pk, "fp8_gemm_dense_mediumm_fp8out_sm100",
        input_fp8, weight_fp8, input_scale, weight_scale,
        output_fp8, output_scale, num_workers,
        runtime_m_mode=runtime_m_mode)


def fp8_gemm_dense_layer(
    pk,
    input_fp8,
    weight_fp8,
    input_scale,
    weight_scale,
    num_workers,
    output=None,
    runtime_m_mode: int = 0,
    variant: str = None,
    fp8out: bool = False,
    output_fp8=None,
    output_scale=None,
):
    """Unified dense FP8 GEMM entry — dispatches to the smallm/mediumm
    tile variants (+ their *_fp8out epilogue-quantize flavors). The
    registered task names are unchanged.

    variant: "smallm" | "mediumm" | None. None auto-picks by the rule
        every DSv3 builder call site used: smallm when
        pk.max_seq_length <= 512, else mediumm.
    fp8out: select the epilogue-UE8M0-quantize flavor — the GEMM emits
        FP8 + packed uint32 scale directly (pass output_fp8 +
        output_scale instead of output).
    """
    if variant is None:
        variant = "smallm" if pk.max_seq_length <= 512 else "mediumm"
    assert variant in ("smallm", "mediumm"), variant
    if fp8out:
        assert output is None and output_fp8 is not None \
            and output_scale is not None, (
            "fp8_gemm_dense_layer(fp8out=True) takes output_fp8 + "
            "output_scale, not output")
        impl = (_fp8_gemm_dense_smallm_fp8out_layer
                if variant == "smallm"
                else _fp8_gemm_dense_mediumm_fp8out_layer)
        impl(pk, input_fp8, weight_fp8, input_scale, weight_scale,
             output_fp8, output_scale, num_workers,
             runtime_m_mode=runtime_m_mode)
    else:
        assert output is not None and output_fp8 is None \
            and output_scale is None, (
            "fp8_gemm_dense_layer takes output (bf16), not "
            "output_fp8/output_scale, unless fp8out=True")
        impl = (_fp8_gemm_dense_smallm_layer if variant == "smallm"
                else _fp8_gemm_dense_mediumm_layer)
        impl(pk, input_fp8, weight_fp8, input_scale, weight_scale,
             output, num_workers, runtime_m_mode=runtime_m_mode)


def linear_splitk_swapAB_fp8_layer(
    pk,
    input_fp8: DTensor,
    input_scale: DTensor,
    weight_fp8: DTensor,
    weight_scale: DTensor,
    output: DTensor,
    grid_dim: tuple,    # (num_M_shards, split_k_factor, 1)
    block_dim: tuple,   # (256, 1, 1) on SM100
    *,
    accumulate: bool,
):
    # Split-K variant of linear_fp8_swapAB_layer. grid.y CTAs each compute
    # a K-slice partial and TMA reduce-add into the shared output tile.
    #
    # The kernel uses tma_reduce_add_async and unconditionally adds onto
    # whatever `output` already contains. The `accumulate` flag selects:
    #   accumulate=True  -> caller owns `output` (e.g. residual). The
    #                       matmul is added on top; no tensor_init.
    #   accumulate=False -> layer prepends a tensor_init that zeroes
    #                       `output` first, so the result is a pure sum.
    # tensor_init shares the linear's grid_dim and per-tensor input_maps,
    # so grid.y CTAs zero the same tile redundantly (kept for dep-edge
    # alignment with the linear).
    #
    # Constraints (asserted at registration time):
    #   - output.dim[1] / grid.x must be a multiple of 128 (per-task N)
    #   - input.dim[1]  / grid.y must be a multiple of 128 (per-task K)
    #   - batch_size <= 16 (decode-only)
    if not accumulate:
        pk.tensor_init_layer(
            target=output,
            dummy=input_fp8,
            grid_dim=grid_dim,
            block_dim=block_dim,
            dummy_input_map=(-1, 1, -1),
            target_input_map=(1, -1, -1),
        )
    params = []
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    # input_fp8 [batch, K]: grid.y splits K (dim 1).
    tb_graph.new_input(input_fp8, (-1, 1, -1), 1, True)
    # input_scale [batch, packed_K]: same K-split.
    tb_graph.new_input(input_scale, (-1, 1, -1), 1, True)
    # weight_fp8 [output, K]: grid.x splits output (dim 0), grid.y splits K (dim 1).
    tb_graph.new_input(weight_fp8, (0, 1, -1), 1, True)
    # weight_scale [output, packed_K]: same partition as weight.
    tb_graph.new_input(weight_scale, (0, 1, -1), 1, True)
    # output [batch, output]: grid.x splits dim 1; grid.y does NOT
    # partition (all grid.y CTAs reduce-add into the same M-shard).
    tb_graph.new_input(output, (1, -1, -1), -1, True)
    pk.kn_graph.customized(
        [input_fp8, input_scale, weight_fp8, weight_scale, output], tb_graph)
    pk.kn_graph.register_task(
        tb_graph, "splitk_linear_fp8_swapAB_sm100", params)


def nvshmem_global_argmax_layer(
    pk,
    partial_value: DTensor,
    partial_index: DTensor,
    scratch_value: DTensor,
    scratch_index: DTensor,
    output: DTensor,
    grid_dim: tuple,
    block_dim: tuple,
    vocab_offset: int,
    valid_vocab_size: int,
    partial_chunk_size: int,
):
    assert pk.world_size > 1
    assert pk.use_nvshmem
    assert partial_value.num_dims == 2  # (batch_size, num_partial_tasks)
    assert partial_index.num_dims == 2  # (batch_size, num_partial_tasks)
    assert scratch_value.num_dims == 2  # (world_size, batch_size)
    assert scratch_index.num_dims == 2  # (world_size, batch_size)
    assert output.num_dims == 2  # (batch_size, 1)
    assert partial_value.dim(0) == partial_index.dim(0)
    assert partial_value.dim(1) == partial_index.dim(1)
    assert scratch_value.dim(0) == pk.world_size
    assert scratch_index.dim(0) == pk.world_size
    assert scratch_value.dim(1) == partial_value.dim(0)
    assert scratch_index.dim(1) == partial_value.dim(0)
    assert partial_chunk_size > 0
    assert 0 <= valid_vocab_size <= partial_value.dim(1) * partial_chunk_size

    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(partial_value, (1, 0, -1), -1, True)
    tb_graph.new_input(partial_index, (1, 0, -1), -1, True)
    tb_graph.new_input(scratch_value, (-1, -1, -1), -1, True)
    tb_graph.new_input(scratch_index, (-1, -1, -1), -1, True)
    tb_graph.new_input(output, (0, 1, -1), -1, True)
    pk.kn_graph.customized(
        [partial_value, partial_index, scratch_value, scratch_index, output],
        tb_graph,
    )
    pk.kn_graph.register_task(
        tb_graph,
        "nvshmem_global_argmax",
        [
            pk.world_size,
            pk.mpi_rank,
            vocab_offset,
            valid_vocab_size,
            partial_chunk_size,
        ],
    )
    allocate_nvshmem_teams(pk, grid_dim[0] * grid_dim[1] * grid_dim[2])
