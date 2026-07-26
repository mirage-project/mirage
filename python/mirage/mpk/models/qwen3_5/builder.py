"""Qwen3.5-35B-A3B-FP8 registry builder.

Assembles the decode/chunked-prefill task graph specified by
`docs/qwen35/v1-architecture.md` §2 out of the tasks M2-I4..I7 and M2-I12/I13
added:

    embed
    for i in 0..39:
        pre-norm (Gemma)
        GDN  : qkv/z fp8 GEMMs + bf16 ba -> gdn_conv1d(234) -> gdn_recurrent(237)
               -> out_proj (fp8, residual)
        ATTN : fused QKVG fp8 GEMM -> paged_attention_sm100 (gate epilogue +
               Q-loop) -> o_proj (fp8, residual)
        post-norm (Gemma)
        MoE  : router -> topk_softmax -> [routed: quantize -> w13(241) ->
               SwiGLU -> quantize -> w2(242)] and [shared: quantize ->
               gate_up -> SwiGLU -> quantize -> down -> sigmoid_gate(238)]
               -> moe_mul_sum_add
    final norm (Gemma) -> lm_head (bf16, vocab 248320) -> argmax

Registered under the checkpoint's own name; `MPK.build()` reaches it through
`get_builder()` (`model_registry.py:25`, `mpk.py:458-467`).

Five things worth knowing before editing this file:

1. **Dense GEMMs are fp8 with the checkpoint's PRESERVED fp32 block scales**
   (task 279 / `linear_fp8_blockscale_layer`), never the UE8M0 requant path —
   `v1-architecture.md` §6.2, amended. Routed experts use the fp32-scale
   grouped fallback ids 241/242 with UNEXPANDED `[E, N/128, K/128]` scales
   (M2-I13's P2 verdict). The shared expert rides the SAME fp8 path as the
   routed ones (M2-I7 measured bf16-dequant 5-50x worse against the oracle).

2. **The routed and shared branches get SEPARATE `quantize_fp8` tasks.** One
   quantize feeding both would be a fork-producer whose consumer `w13` is a
   join-consumer (it also waits on the router) — `build_annotated_graph`
   rejects that as case 3 (`annotated_graph.cc:642-661`). Same bytes, one extra
   task.

3. **`topk_softmax` ZEROES its input logits buffer** as it reads it
   (`persistent_kernel.py:1718`), so every layer gets its own router-logits
   buffer and nothing reads it after the router.

4. **`residual=` edges are legal because of residual stripping.** `h` feeds both
   the pre-norm and the projection's residual slot; the direct edge is stripped
   because the computed path already reaches it (`annotated_graph.cc:467-551`),
   which is what keeps `h`'s producer from being a fork+join producer.

5. **Two silent-corruption footguns are asserted here** (`mpk-gaps.md` §8
   risk 5): `mbt >= mbr` (otherwise surplus requests get 0 tokens and stall
   forever) and `max_num_pages >= mbr * ceil(max_seq_length / page_size)`
   (otherwise the page FIFO wraps and hands out pages another request owns).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch

from ....core import bfloat16, float32, float8_e4m3, int32, int64
from ...model_registry import register_model_builder
from ...persistent_kernel import PersistentKernel
from ..graph_builder import GraphBuilder, MirageModelConfig
from ..utils import grid_for_rmsnorm_linear_layer
from . import rope
from .weight_loader import Qwen35Config, Qwen35WeightLoader, resolve_snapshot

BLOCK = 128

# The MoE router task (`topk_softmax_sm100.cuh`) has NO loop over rows: with a
# strictly-256-thread block it covers exactly
# `WARPS_PER_CTA * ROWS_PER_WARP = 8 * (WARP_SIZE * VPT / NUM_EXPERTS)` query
# rows (= 16 at Qwen3.5's 256 experts), and `thread_row` is derived from
# threadIdx alone, so a second task instance would recompute the same rows
# rather than the next slice. Any chunk wider than this silently leaves
# `topk_w`/`routing` at ZERO for the surplus rows -- the routed experts then
# contribute nothing for those tokens at EVERY one of the 40 layers, while the
# shared expert and the residual keep flowing, so the corruption is a quiet
# quality loss rather than a crash.
#
# M2-I9 hit exactly this: at max_num_batched_tokens=128 the first 16 prefill
# rows were correct and rows 16+ degraded, which made every AC-3 prompt diverge
# at generated position 0 (the token read from the last prefill row). Found by
# per-boundary bisection -- `layer_0_topk_w` already had 14 zero rows.
# M2-I4..I8's single-layer gates all used 8-token chunks and could not see it.
#
# The real fix is a row loop (or a row-offset task param) in the kernel, which
# also touches DeepSeek-V3's router and needs its own validation. Until then
# this is a hard capacity bound, asserted rather than commented.
MOE_ROUTER_MAX_ROWS_PER_TASK = 16

# `quantize_fp8_layer`'s default `(-1,-1,-1)` hands EVERY task the whole tensor,
# and the kernel then loops over all of its rows (it cannot use blockIdx.x as a
# row index -- that is the physical worker id under the persistent runtime). A
# `grid_dim=(mbt,1,1)` launch therefore quantized the whole activation mbt=16
# times: M3-I1 measured 84.1 ms of worker time per decode step at bs1 across the
# 240 call sites (3840 tasks x 21.9 us) and a 4540 us wall span -- 29.7% of the
# step and the single largest task type at bs<=4, for 5.3 ms of useful work.
#
# Splitting grid.x over tensor dim 0 (the token axis) gives each task exactly
# its own row. Bit-exact by construction: a 128-element group's fp8 bytes and
# its fp32 block scale are computed from that group alone, and the kernel's row
# loop carries no state across rows, so moving rows between CTAs cannot change a
# byte. Every qwen3.5 quantize site is `scale_ue8m0=False` (preserved fp32 block
# scales, M2-I12/I13), whose scale is row-major with the input's row axes -- the
# precondition `quantize_fp8_layer` asserts. The same tuple serves the 2-D
# [mbt, hidden] sites and the 3-D [mbt, topk, inter] MoE-activation site: dim 0
# is the token axis in both, so a 3-D task keeps its whole [1, topk, inter]
# slice (topk rows) and a 2-D task keeps one row.
QUANTIZE_ROW_SPLIT = (0, -1, -1)


def fp8_grid(output_size: int) -> int:
    """Task count for a preserved-block-scale dense GEMM.

    The kernel's per-task N slice must be a whole number of 128-row scale
    blocks (`linear_fp8_blockscale_sm100.cuh:120`), and grid.x splits both the
    weight's rows and the scale's dim 0 (`persistent_kernel.py:2059-2060`), so
    one scale row per task is the finest legal split.
    """
    assert output_size % BLOCK == 0, (
        f"preserved-scale FP8 output size {output_size} must be a multiple of {BLOCK}")
    return output_size // BLOCK


@register_model_builder(
    "Qwen3.5-35B-A3B-FP8",
    "Qwen/Qwen3.5-35B-A3B-FP8",
    "qwen3_5",
    "Qwen3.5",
)
class Qwen35Builder(GraphBuilder):
    def __init__(self, mpk: PersistentKernel, weights: Optional[dict] = None):
        super().__init__(mpk, weights)
        self.mbt = mpk.max_num_batched_tokens
        self.mbr = mpk.max_num_batched_requests
        self.max_num_pages = mpk.max_num_pages
        self.page_size = mpk.page_size
        self.max_seq_length = mpk.max_seq_length
        self.world_size = mpk.world_size
        self.rank = mpk.mpi_rank
        self.tokenizer = None
        self.model_name: Optional[str] = None
        self.model_path: Optional[str] = None
        self.config: Optional[Qwen35Config] = None
        self.loader: Optional[Qwen35WeightLoader] = None
        self.eos_token_id = 248044  # config.json text_config.eos_token_id
        # tuning knobs (task-count / parallelism), overridable before build
        self.gdn_conv_channel_blocks = 8
        self.moe_n_splits = 2
        self.qk_dense_path = "fp8"
        # When set, every intermediate is a host-visible torch tensor
        # (`attach_input`) instead of an opaque `new_tensor`, and is recorded in
        # `self.buffers`. This is the `online_notoken` "fixed tensor" pattern the
        # Qwen3 builder already uses (`models/qwen3/builder.py:120-163`); the
        # single-layer test-mode gates read every op boundary through it, and
        # M2-I9 gets the same handle for divergence bisection.
        self.expose_intermediates = False
        # Narrow variant of the above for M2-I9's per-position logit evidence:
        # makes ONLY the lm_head output host-visible, so a single forced-prefix
        # step can be read back as a real logit vector (the tie-flip
        # adjudication AC-3 requires) without paying for 40 layers of exposed
        # intermediates.
        self.expose_logits = False
        self.buffers: Dict[str, torch.Tensor] = {}
        self._keep: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # entry points
    # ------------------------------------------------------------------
    def build_from_model(self, model_name: Optional[str] = None,
                         model_path: Optional[str] = None):
        self.model_name, self.model_path = model_name, model_path
        snapshot = resolve_snapshot(model_name, model_path)
        self.loader = Qwen35WeightLoader(snapshot, device="cuda",
                                         qk_dense_path=self.qk_dense_path)
        self.config = self.loader.config
        self.eos_token_id = self.config.eos_token_id
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(snapshot)
        except Exception as exc:                                  # noqa: BLE001
            print(f"[qwen3_5] tokenizer unavailable ({exc}); encode/decode disabled")
        self.build_from_weights(self.loader.load(), self.config)

    def build_from_config(self, model_config: MirageModelConfig):
        raise NotImplementedError(
            "Qwen3.5 builds from the checkpoint (build_from_model) or from a "
            "prepared weight dict (build_from_weights); MirageModelConfig has "
            "no slot for the GDN state pools, the block scales or layer_types.")

    # ------------------------------------------------------------------
    def build_from_weights(self, weights: Dict[str, torch.Tensor],
                           config: Qwen35Config, *, with_lm_head: bool = True):
        """Wire the whole graph from an already-materialised weight dict."""
        self.config = config
        self.weights = weights
        self._check_footguns()
        self._check_vocab()
        self._alloc_caches()
        self._alloc_state_pools()
        self._attach_common()
        self._attach_model_level()
        self._build_graph(with_lm_head=with_lm_head)

    def build_layer_probe(self, weights: Dict[str, torch.Tensor],
                          config: Qwen35Config, layer_idx: int,
                          hidden: torch.Tensor):
        """Wire exactly ONE decoder layer, with `hidden` as its input residual
        stream, and expose every intermediate as a readable torch tensor.

        This is the single-layer test-mode gate's entry point (and M2-I9's
        divergence-bisection handle): it skips the embedding and the lm_head so
        the layer under test is fed the oracle's own hidden state rather than
        an accumulated one. Returns the layer's output DTensor; the per-op
        buffers are in `self.buffers`.
        """
        self.config = config
        self.weights = weights
        self.expose_intermediates = True
        self._check_footguns()
        self._alloc_caches()
        self._alloc_state_pools()
        self._attach_common()
        h = self.mpk.attach_input(hidden, name="probe_hidden")
        self.buffers["probe_hidden"] = hidden
        return self._build_layer(layer_idx, h)

    # ------------------------------------------------------------------
    # the two silent-corruption asserts (mpk-gaps.md §8 risk 5)
    # ------------------------------------------------------------------
    def _check_footguns(self):
        assert self.mbt >= self.mbr, (
            f"max_num_batched_tokens ({self.mbt}) < max_num_batched_requests "
            f"({self.mbr}): surplus requests are admitted with 0 tokens and "
            f"stall forever (persistent_kernel.cuh:326, mpk-gaps.md §8 risk 5)")
        assert self.mbt <= MOE_ROUTER_MAX_ROWS_PER_TASK, (
            f"max_num_batched_tokens ({self.mbt}) > "
            f"{MOE_ROUTER_MAX_ROWS_PER_TASK}: the MoE router task routes only "
            f"the first {MOE_ROUTER_MAX_ROWS_PER_TASK} rows of a chunk and "
            f"leaves topk_w/routing at ZERO for the rest, so every surplus "
            f"token loses its routed experts at all {self.config.num_layers} "
            f"layers (M2-I9 root cause; see MOE_ROUTER_MAX_ROWS_PER_TASK)")
        needed = self.mbr * math.ceil(self.max_seq_length / self.page_size)
        assert self.max_num_pages >= needed, (
            f"max_num_pages ({self.max_num_pages}) < max_num_batched_requests "
            f"* ceil(max_seq_length / page_size) ({self.mbr} * "
            f"{math.ceil(self.max_seq_length / self.page_size)} = {needed}): "
            f"the page FIFO wraps (page_queue[head % MPK_MAX_NUM_PAGES]) and "
            f"hands out pages another request still owns -> silent KV "
            f"corruption (mpk-gaps.md §8 risk 5)")

    def _check_vocab(self):
        c = self.config
        self.padded_vocab_size = c.padded_vocab_size
        assert self.padded_vocab_size == c.vocab_size, (
            f"vocab {c.vocab_size} needs padding to {self.padded_vocab_size}; "
            f"v1-architecture.md §7 asserts 248320 = 970*256 needs none")
        assert grid_for_rmsnorm_linear_layer(self.padded_vocab_size) is not None

    # ------------------------------------------------------------------
    # buffers
    # ------------------------------------------------------------------
    def _alloc_caches(self):
        c = self.config
        n = max(1, len(c.attn_layers))
        shape = (n, self.max_num_pages, self.page_size,
                 c.num_key_value_heads, c.head_dim)
        self.k_cache = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
        self.v_cache = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
        cos, sin = rope.build_cos_sin_table(
            torch.arange(self.max_seq_length), head_dim=c.head_dim,
            rotary_dim=c.rotary_dim, theta=c.rope_theta,
            dtype=torch.bfloat16, device="cuda")
        self.cos_table, self.sin_table = cos.contiguous(), sin.contiguous()
        self._keep += [self.k_cache, self.v_cache, self.cos_table, self.sin_table]

    def _alloc_state_pools(self):
        """Per-request GDN state, indexed by request SLOT (`v1-arch` §3.1).

        conv `[mbr, kernel-1, conv_dim]` bf16 and recurrent
        `[mbr, v_heads, v_dim, k_dim]` **fp32** — fp32 is mandatory, the
        checkpoint sets `mamba_ssm_dtype: float32` and the AC-3 reference's
        recurrence is fp32-state math. Lifetime is kernel-side: a slot at
        `step == 0` treats the stored state as zero, so slot reuse re-zeros
        implicitly (`v1-arch` §3.3) — nothing to reset here.
        """
        c = self.config
        n = max(1, len(c.gdn_layers))
        self.conv_state = torch.zeros(
            (n, self.mbr, c.linear_conv_kernel_dim - 1, c.conv_dim),
            dtype=torch.bfloat16, device="cuda")
        self.recurrent_state = torch.zeros(
            (n, self.mbr, c.linear_num_value_heads,
             c.linear_value_head_dim, c.linear_key_head_dim),
            dtype=torch.float32, device="cuda")
        self._keep += [self.conv_state, self.recurrent_state]

    _TORCH_DTYPE = {
        "bf16": torch.bfloat16, "fp32": torch.float32,
        "fp8_e4m3": torch.float8_e4m3fn, "int32": torch.int32,
        "int64": torch.int64,
    }

    def _t(self, dims, dtype, name):
        if not self.expose_intermediates:
            return self.mpk.new_tensor(dims=dims, dtype=dtype, name=name,
                                       io_category="cuda_tensor")
        buf = torch.zeros(dims, dtype=self._TORCH_DTYPE[str(dtype)], device="cuda")
        self.buffers[name] = buf
        self._keep.append(buf)
        return self.mpk.attach_input(buf, name=name)

    def _attach_common(self):
        pk, c = self.mpk, self.config
        self.cos_dt = pk.attach_input(self.cos_table, name="cos_position_embedding")
        self.sin_dt = pk.attach_input(self.sin_table, name="sin_position_embedding")
        self._attn_slot = {li: j for j, li in enumerate(c.attn_layers)}
        self._gdn_slot = {li: j for j, li in enumerate(c.gdn_layers)}

    def _attach_model_level(self):
        pk, w = self.mpk, self.weights
        for key in ("input_tokens", "output_tokens"):
            assert pk.meta_tensors[key].dim() == 2, (
                f"{key} must be [max_num_batched_tokens, 1] as production wires "
                f"it (demo/qwen3/demo.py:261-262) — argmax_reduce_layer asserts "
                f"a 2-D output (persistent_kernel.py:2451). Test mode's 1-D "
                f"auto-default does not fit; pass it explicitly.")
        self.input_tokens = pk.attach_input(pk.meta_tensors["input_tokens"],
                                            name="input_token")
        self.output_tokens = pk.attach_input(pk.meta_tensors["output_tokens"],
                                             name="output_token")
        self.embed_w = pk.attach_input(w["embed_tokens"], name="embed_tokens")
        self.norm_w = pk.attach_input(w["model_norm"], name="model_norm_weight")
        self.lm_head_w = (pk.attach_input(w["lm_head"], name="lm_head")
                          if "lm_head" in w else None)

    # ------------------------------------------------------------------
    # fp8 dense helper
    # ------------------------------------------------------------------
    def _fp8_linear(self, name: str, x, w_name: str, out, *, residual=None,
                    k: int):
        """quantize(x) -> preserved-block-scale fp8 GEMM (optionally +residual).

        A dedicated quantize task per call site: sharing one would make the
        quantize a fork-producer, and any consumer that also waits on another
        producer (the router, for the MoE branches) then trips case 3.
        """
        pk, w = self.mpk, self.weights
        xq = self._t((self.mbt, k), float8_e4m3, f"{name}_xq")
        xs = self._t((self.mbt, k // BLOCK), float32, f"{name}_xs")
        pk.quantize_fp8_layer(input=x, output_fp8=xq, output_scale=xs,
                              grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1),
                              scale_ue8m0=False, row_partition=QUANTIZE_ROW_SPLIT)
        weight = pk.attach_input(w[w_name], name=w_name)
        scale = pk.attach_input(w[w_name + "_scale"], name=w_name + "_scale")
        n = w[w_name].shape[0]
        pk.linear_fp8_blockscale_layer(
            input_fp8=xq, input_scale=xs, weight_fp8=weight, weight_scale=scale,
            output=out, grid_dim=(fp8_grid(n), 1, 1), block_dim=(256, 1, 1),
            residual=residual)

    # ------------------------------------------------------------------
    # graph
    # ------------------------------------------------------------------
    def _build_graph(self, *, with_lm_head: bool):
        pk, c = self.mpk, self.config
        h = self._t((self.mbt, c.hidden_size), bfloat16, "embed_out")
        pk.embed_layer(input=self.input_tokens, weight=self.embed_w, output=h,
                       grid_dim=(1, 1, 1), block_dim=(128, 1, 1), input_source=1)

        for i in range(c.num_layers):
            h = self._build_layer(i, h)

        nrm = self._t((self.mbt, c.hidden_size), bfloat16, "final_norm_out")
        pk.rmsnorm_layer(input=h, weight=self.norm_w, output=nrm,
                         grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1))
        if not with_lm_head:
            self.last_hidden = nrm
            return

        if self.expose_logits and not self.expose_intermediates:
            buf = torch.zeros((self.mbt, self.padded_vocab_size),
                              dtype=torch.bfloat16, device="cuda")
            self.buffers["argmax_in"] = buf
            self._keep.append(buf)
            logits = pk.attach_input(buf, name="argmax_in")
        else:
            logits = self._t((self.mbt, self.padded_vocab_size), bfloat16,
                             "argmax_in")
        pk.linear_layer(input=nrm, weight=self.lm_head_w, output=logits,
                        grid_dim=(grid_for_rmsnorm_linear_layer(
                            self.padded_vocab_size), 1, 1),
                        block_dim=(128, 1, 1))
        part_v = self._t((self.mbt, pk.num_workers), bfloat16, "argmax_part_value")
        part_i = self._t((self.mbt, pk.num_workers), int64, "argmax_part_index")
        pk.argmax_partial_layer(input=logits, output=(part_v, part_i),
                                grid_dim=(pk.num_workers, 1, 1),
                                block_dim=(128, 1, 1))
        pk.argmax_reduce_layer(input=(part_v, part_i), output=self.output_tokens,
                               grid_dim=(1, 1, 1), block_dim=(128, 1, 1))

    def _build_layer(self, i: int, h):
        pk, w, c = self.mpk, self.weights, self.config
        pre_w = pk.attach_input(w[f"layer_{i}_input_layernorm"],
                                name=f"layer_{i}_input_layernorm")
        nrm = self._t((self.mbt, c.hidden_size), bfloat16, f"layer_{i}_pre_norm")
        pk.rmsnorm_layer(input=h, weight=pre_w, output=nrm,
                         grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1))

        if c.layer_types[i] == "linear_attention":
            h = self._build_gdn(i, nrm, h)
        else:
            h = self._build_attention(i, nrm, h)
        return self._build_moe(i, h)

    # ---- GDN ---------------------------------------------------------
    def _build_gdn(self, i: int, nrm, h):
        pk, w, c = self.mpk, self.weights, self.config
        slot = self._gdn_slot[i]

        # qkv and z share ONE quantize: both consumers have a single producer,
        # so this fork never meets a join (annotated_graph.cc case 3).
        xq = self._t((self.mbt, c.hidden_size), float8_e4m3, f"layer_{i}_gdn_xq")
        xs = self._t((self.mbt, c.hidden_size // BLOCK), float32, f"layer_{i}_gdn_xs")
        pk.quantize_fp8_layer(input=nrm, output_fp8=xq, output_scale=xs,
                              grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1),
                              scale_ue8m0=False, row_partition=QUANTIZE_ROW_SPLIT)
        qkv = self._t((self.mbt, c.conv_dim), bfloat16, f"layer_{i}_gdn_qkv")
        z = self._t((self.mbt, c.gdn_z_dim), bfloat16, f"layer_{i}_gdn_z")
        for tag, out in (("in_proj_qkv", qkv), ("in_proj_z", z)):
            name = f"layer_{i}_gdn_{tag}"
            pk.linear_fp8_blockscale_layer(
                input_fp8=xq, input_scale=xs,
                weight_fp8=pk.attach_input(w[name], name=name),
                weight_scale=pk.attach_input(w[name + "_scale"], name=name + "_scale"),
                output=out, grid_dim=(fp8_grid(w[name].shape[0]), 1, 1),
                block_dim=(256, 1, 1))

        # b/a stay bf16 unconditionally: both shards are in
        # modules_to_not_convert and N=32 < block_n (vllm-graph.md §6 g.7).
        ba_n = w[f"layer_{i}_gdn_in_proj_ba"].shape[0]
        ba = self._t((self.mbt, ba_n), bfloat16, f"layer_{i}_gdn_ba")
        pk.linear_layer(input=nrm,
                        weight=pk.attach_input(w[f"layer_{i}_gdn_in_proj_ba"],
                                               name=f"layer_{i}_gdn_in_proj_ba"),
                        output=ba, grid_dim=(max(1, ba_n // 64), 1, 1),
                        block_dim=(128, 1, 1))

        qkv_c = self._t((self.mbt, c.conv_dim), bfloat16, f"layer_{i}_gdn_qkv_c")
        conv_state = pk.attach_input(self.conv_state[slot],
                                     name=f"layer_{i}_gdn_conv_state")
        pk.gdn_conv1d_layer(
            input=qkv,
            weight=pk.attach_input(w[f"layer_{i}_gdn_conv1d"],
                                   name=f"layer_{i}_gdn_conv1d"),
            conv_state=conv_state, output=qkv_c,
            grid_dim=(self.mbr, self.gdn_conv_channel_blocks, 1),
            block_dim=(256, 1, 1))

        g_out = self._t((self.mbt, c.gdn_z_dim), bfloat16, f"layer_{i}_gdn_out")
        pk.gdn_recurrent_layer(
            qkv=qkv_c, ba=ba,
            alog_dtbias=pk.attach_input(w[f"layer_{i}_gdn_alog_dtbias"],
                                        name=f"layer_{i}_gdn_alog_dtbias"),
            state=pk.attach_input(self.recurrent_state[slot],
                                  name=f"layer_{i}_gdn_state"),
            z=z,
            norm_w=pk.attach_input(w[f"layer_{i}_gdn_norm"],
                                   name=f"layer_{i}_gdn_norm"),
            output=g_out, num_k_heads=c.linear_num_key_heads,
            grid_dim=(c.linear_num_value_heads, self.mbr, 1),
            block_dim=(256, 1, 1))

        out = self._t((self.mbt, c.hidden_size), bfloat16, f"layer_{i}_attn_resid")
        self._fp8_linear(f"layer_{i}_gdn_out_proj", g_out,
                         f"layer_{i}_gdn_out_proj", out, residual=h,
                         k=c.gdn_z_dim)
        return out

    # ---- full attention ----------------------------------------------
    def _build_attention(self, i: int, nrm, h):
        pk, w, c = self.mpk, self.weights, self.config
        slot = self._attn_slot[i]
        qkvg = self._t((self.mbt, c.qkvg_dim), bfloat16, f"layer_{i}_qkvg")
        if self.qk_dense_path == "fp8":
            self._fp8_linear(f"layer_{i}_qkvg_proj", nrm, f"layer_{i}_qkvg_proj",
                             qkvg, k=c.hidden_size)
        else:
            pk.linear_layer(
                input=nrm,
                weight=pk.attach_input(w[f"layer_{i}_qkvg_proj"],
                                       name=f"layer_{i}_qkvg_proj"),
                output=qkvg,
                grid_dim=(grid_for_rmsnorm_linear_layer(c.qkvg_dim), 1, 1),
                block_dim=(128, 1, 1))

        attn = self._t((self.mbt, c.num_attention_heads * c.head_dim), bfloat16,
                       f"layer_{i}_attn_out")
        pk.paged_attention_layer(
            input=qkvg,
            k_cache=pk.attach_input(self.k_cache[slot], name=f"layer_{i}_k_cache"),
            v_cache=pk.attach_input(self.v_cache[slot], name=f"layer_{i}_v_cache"),
            q_norm=pk.attach_input(w[f"layer_{i}_q_norm"], name=f"layer_{i}_q_norm"),
            k_norm=pk.attach_input(w[f"layer_{i}_k_norm"], name=f"layer_{i}_k_norm"),
            cos_pos_embed=self.cos_dt, sin_pos_embed=self.sin_dt,
            output=attn, grid_dim=(self.mbr, c.num_key_value_heads, 1),
            block_dim=(256, 1, 1), enable_qk_norm=True,
            attn_output_gate=True,
            # P3 measured the post-5715c6f smem arena at head_dim 256 / GQA 8:1
            # as admissible only up to 4 queries per pass; the in-task Q-loop
            # covers longer chunks (v1-architecture.md §4.3).
            max_tokens_per_pass=min(4, self.mbt))

        out = self._t((self.mbt, c.hidden_size), bfloat16, f"layer_{i}_attn_resid")
        self._fp8_linear(f"layer_{i}_o_proj", attn, f"layer_{i}_o_proj", out,
                         residual=h, k=c.num_attention_heads * c.head_dim)
        return out

    # ---- MoE ----------------------------------------------------------
    def _build_moe(self, i: int, h):
        pk, w, c = self.mpk, self.weights, self.config
        mbt, topk, inter = self.mbt, c.num_experts_per_tok, c.moe_intermediate_size

        post_w = pk.attach_input(w[f"layer_{i}_post_attention_layernorm"],
                                 name=f"layer_{i}_post_attention_layernorm")
        x = self._t((mbt, c.hidden_size), bfloat16, f"layer_{i}_post_norm")
        pk.rmsnorm_layer(input=h, weight=post_w, output=x,
                         grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1))

        # ---- router (bf16, never quantized: vllm-graph.md §2.3.1) --------
        logits = self._t((mbt, c.num_experts), bfloat16, f"layer_{i}_router_logits")
        pk.linear_layer(input=x,
                        weight=pk.attach_input(w[f"layer_{i}_router"],
                                               name=f"layer_{i}_router"),
                        output=logits,
                        grid_dim=(min(grid_for_rmsnorm_linear_layer(c.num_experts),
                                      c.num_experts // 8), 1, 1),
                        block_dim=(256, 1, 1))
        topk_w = self._t((mbt, topk), float32, f"layer_{i}_topk_w")
        routing = self._t((c.num_experts, mbt), int32, f"layer_{i}_routing")
        mask = self._t((c.num_experts + 1,), int32, f"layer_{i}_moe_mask")
        pk.moe_topk_softmax_routing_layer(
            input=logits, output=(topk_w, routing, mask), grid_dim=(1, 1, 1),
            block_dim=(256, 1, 1),
            # HF hands the combine BF16 weights (router_top_value.to(dtype));
            # probe P5 pinned this clause empirically.
            round_weights_to_input_dtype=True)

        # ---- routed experts ---------------------------------------------
        rq = self._t((mbt, c.hidden_size), float8_e4m3, f"layer_{i}_moe_xq")
        rs = self._t((mbt, c.hidden_size // BLOCK), float32, f"layer_{i}_moe_xs")
        pk.quantize_fp8_layer(input=x, output_fp8=rq, output_scale=rs,
                              grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
                              scale_ue8m0=False, row_partition=QUANTIZE_ROW_SPLIT)
        grid_x = min(c.num_experts, mbt * topk)
        mid = self._t((mbt, topk, 2 * inter), bfloat16, f"layer_{i}_moe_mid")
        pk.moe_fp8_blockscale_layer(
            input_fp8=rq, input_scale=rs,
            weight_fp8=pk.attach_input(w[f"layer_{i}_w13"], name=f"layer_{i}_w13"),
            weight_scale=pk.attach_input(w[f"layer_{i}_w13_scale"],
                                         name=f"layer_{i}_w13_scale"),
            moe_routing_indices=routing, moe_mask=mask, output=mid,
            grid_dim=(grid_x, self.moe_n_splits, 1), block_dim=(256, 1, 1),
            w13_linear=True)
        act = self._t((mbt, topk, inter), bfloat16, f"layer_{i}_moe_act")
        pk.moe_silu_mul_layer(input=mid, output=act, grid_dim=(mbt, topk, 1),
                              block_dim=(128, 1, 1))
        aq = self._t((mbt, topk, inter), float8_e4m3, f"layer_{i}_moe_actq")
        as_ = self._t((mbt, topk, inter // BLOCK), float32, f"layer_{i}_moe_acts")
        pk.quantize_fp8_layer(input=act, output_fp8=aq, output_scale=as_,
                              grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
                              scale_ue8m0=False, row_partition=QUANTIZE_ROW_SPLIT)
        down = self._t((mbt, topk, c.hidden_size), bfloat16, f"layer_{i}_moe_down")
        pk.moe_fp8_blockscale_layer(
            input_fp8=aq, input_scale=as_,
            weight_fp8=pk.attach_input(w[f"layer_{i}_w2"], name=f"layer_{i}_w2"),
            weight_scale=pk.attach_input(w[f"layer_{i}_w2_scale"],
                                         name=f"layer_{i}_w2_scale"),
            moe_routing_indices=routing, moe_mask=mask, output=down,
            grid_dim=(grid_x, self.moe_n_splits, 1), block_dim=(256, 1, 1),
            w13_linear=False)

        # ---- shared expert (own quantize; see module docstring note 2) ----
        shared_inter = c.shared_expert_intermediate_size
        smid = self._t((mbt, 2 * shared_inter), bfloat16, f"layer_{i}_shared_mid")
        self._fp8_linear(f"layer_{i}_shared_gate_up", x,
                         f"layer_{i}_shared_gate_up", smid, k=c.hidden_size)
        sact = self._t((mbt, shared_inter), bfloat16, f"layer_{i}_shared_act")
        pk.silu_mul_layer(input=smid, output=sact,
                          grid_dim=(shared_inter // BLOCK, 1, 1),
                          block_dim=(128, 1, 1))
        sout = self._t((mbt, c.hidden_size), bfloat16, f"layer_{i}_shared_out")
        self._fp8_linear(f"layer_{i}_shared_down", sact, f"layer_{i}_shared_down",
                         sout, k=shared_inter)

        r_prime = self._t((mbt, c.hidden_size), bfloat16, f"layer_{i}_r_prime")
        pk.sigmoid_gate_mul_add_layer(
            input=x,
            gate_weight=pk.attach_input(w[f"layer_{i}_shared_expert_gate"],
                                        name=f"layer_{i}_shared_expert_gate"),
            shared=sout, residual=h, output=r_prime,
            grid_dim=(mbt, 1, 1), block_dim=(256, 1, 1))

        out = self._t((mbt, c.hidden_size), bfloat16, f"layer_{i}_moe_out")
        pk.moe_mul_sum_add_layer(input=down, weight=topk_w, residual=r_prime,
                                 output=out, grid_dim=(mbt, 1, 1),
                                 block_dim=(128, 1, 1))
        return out

    # ------------------------------------------------------------------
    def encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, ids: torch.Tensor):
        return self.tokenizer.decode(ids, skip_special_tokens=True)
