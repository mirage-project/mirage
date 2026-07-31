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
import os
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

# Rows the MoE router task (`topk_softmax_sm100.cuh`) covers in ONE pass: with a
# strictly-256-thread block that is
# `WARPS_PER_CTA * ROWS_PER_WARP = 8 * (WARP_SIZE * VPT / NUM_EXPERTS)` query
# rows (= 16 at Qwen3.5's 256 experts and the VPT the registration picks).
#
# This used to be a HARD CAP, asserted here. `thread_row` was derived from
# threadIdx alone with no loop around it, so any chunk wider than one pass
# silently left `topk_w`/`routing` at ZERO for the surplus rows -- the routed
# experts then contributed nothing for those tokens at EVERY one of the 40
# layers, while the shared expert and the residual kept flowing, so the
# corruption was a quiet quality loss rather than a crash.
#
# M2-I9 hit exactly this: at max_num_batched_tokens=128 the first 16 prefill
# rows were correct and rows 16+ degraded, which made every AC-3 prompt diverge
# at generated position 0 (the token read from the last prefill row). Found by
# per-boundary bisection -- `layer_0_topk_w` already had 14 zero rows.
# M2-I4..I8's single-layer gates all used 8-token chunks and could not see it.
#
# M3-I5b put the row loop in the kernel (both the softmax router and DeepSeek's
# sigmoid sibling), so mbt is no longer bounded by it: a router task makes
# ceil(mbt / MOE_ROUTER_ROWS_PER_TILE) passes and every row is routed. The
# constant survives only as the cost unit -- raising mbt above it buys admission
# headroom at one extra router pass per 16 rows, which is what the M3-I5b mbt
# A/B measures (demo/qwen3_5/accept/opt/m3i5b/prep.md).
MOE_ROUTER_ROWS_PER_TILE = 16

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

# The router runs on all `mbt` rows, but a decode step fills only
# `qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]` of them. Rows above that hold
# the previous iteration's residue -- attention and GDN write per REQUEST slot,
# so nothing refreshes a row no request owns -- and that residue is different in
# every padding row, so each one contributes its own top-8 marks. M3-I1 measured
# the result in the profile: 56.4 / 59.4 / 60.2 / 70.1 / 86.7 ACTIVATED expert
# groups per layer at bs 1/2/4/8/16, where the live tokens need at most
# min(256, 8*bs) = 8 / 16 / 32 / 64 / 128. At bs1 the grouped GEMM therefore
# streams ~7x the expert weight the routing asks for.
#
# `gate_padding_rows` marks experts for live rows only. It is bit-exact for
# every live row: a row's top-k is a per-row reduction over its own 256 logits,
# the grouped GEMM gathers rows in ascending token order (so live rows keep
# their slots), and each (token, slot) pair is owned by exactly one expert, so
# no output element changes owner. Only padding rows' intermediates move -- and
# M2's AC-3 already measured that live rows are independent of them: every
# prompt's 64 output token ids are byte-identical at bs 1/2/4/8/16, i.e.
# identical whether the other 15 rows hold residue or 15 different live prompts.
#
# Set False to rebuild the pre-M3-I8 graph (the A/B `base` arm).
#
# MECHANISM CLAIM (corrected + closed 2026-07-27, opt/m3i8/results/f1_closure/
# rootcause.md): activated groups per layer = |union over live rows of top8(r)|
# -- exactly 8 at live=1 (measured 8.0000, min=max, 32/32 steady iterations),
# 8k minus router collisions at live=k (bs2 measured 15.731 vs the collision
# model's 15.75). The 2026-07-27 default-OFF interlude was an INSTRUMENT
# artifact: the oracle's 1us long-task threshold sat inside the empty-task
# latency tail (1.6% of empties cross it; the histogram is empty from 2us to
# 48us, real tiles start at 48us). Oracle threshold for task 241/242
# classification is 4us. Falsifier for any future run: with the 4us separator,
# activated > 8*n_live in any steady-decode iteration (at bs1: != 8.000).
MOE_GATE_PADDING_ROWS = True


# Per-task N slice for the preserved-block-scale dense GEMM, keyed by the
# projection's (N, K). These are the slices the ferret `dense-fp8-blockscale`
# winner (workspace4 tag v011) was BENCHMARKED at -- one entry per real Qwen3.5
# dense call site -- and the kernel's ring depths are tuned per (K, slice), so a
# slice not in this table is not a slice we have evidence for. The measured
# effect of slicing is large: the ferret score went 0.727 (slice 128, one scale
# block per task) -> 0.862 the moment per-shape slicing landed, and the deep
# cp.async rings that carried it to 1.011 only FIT the worker's shared memory at
# a narrow slice. The mechanism is MPK-specific: one persistent worker CTA per SM
# runs one task at a time, so a projection dispatched as N/128 tasks occupies
# only N/128 of the 148 SMs -- 16 of them for the N=2048 out_proj/o_proj pair.
FP8_DENSE_N_SLICE = {
    (8192, 2048): 64,   # GDN in_proj_qkv                 -> 128 tasks
    (4096, 2048): 32,   # GDN in_proj_z                   -> 128 tasks
    (9216, 2048): 64,   # attention qkv(g)_proj           -> 144 tasks
    (2048, 4096): 16,   # GDN out_proj / attn o_proj      -> 128 tasks (residual)
    (1024, 2048): 32,   # shared-expert gate_up           ->  32 tasks
    (2048,  512): 64,   # shared-expert down              ->  32 tasks
}


def fuse_silu_quant() -> bool:
    """True unless MPK_FUSE_SILU_QUANT=0 disables fusing the MoE activation SwiGLU
    into its quantize (M4-I9). DEFAULT ON since 2026-07-30 -- see SHIP THE STACK below.

    SHIP THE STACK, NOT ONE FLAG. This and the two flags below were flipped on
    together on the strength of the three-flag arm (`S`), not their solo numbers:
    e2e 1.026 / 1.034 / 1.038 / 1.021 / 1.004x at bs 1/2/4/8/16 with 15/15 paired
    reps favouring S at every batch size, AC-3 STABLE and bit-exact 10/10 at all
    five, and registers/stack/spill unchanged at 255 / 96 B / 0 (the shared
    register budget is AT its ceiling, so this mattered). DO NOT enable
    MPK_FUSE_NORM_QUANT alone: solo it MEASURES 0.986x at bs16, a real
    regression, and only composes to a win on top of this flag -- the mechanism
    is in fuse_norm_quant's docstring.

    The routed-expert chain is `w13 -> moe_silu_mul -> quantize -> w2` and
    `layer_i_moe_act` has exactly one consumer, so the two middle tasks can be
    one. Why that is the lever: M4-I8's exact step decomposition showed AC-4's
    binding floors are `max(longest weighted dependency chain, total task work /
    128)` = 1.179x / 1.116x / 1.127x of vLLM's whole step at bs 1/8/16, and
    those survive every scheduler, queue-policy and dispatch-latency fix by
    construction. The only lever that lowers them is a SHORTER chain, and
    removing a record removes its duration AND its ~1.15 us of event-visibility
    latency AND its ~1.55 us of queue-pop latency AND its barrier pair.

    Measured on the M4-I8 buffers (opt/m4i9/): this site is the largest
    ADMISSIBLE fusion at every batch size -- 40 chain records (one per layer),
    cp_exact -177.3 / -189.0 / -184.4 us and 45-50 us of chain gap at bs 1/8/16.
    It also drops the bf16 `moe_act` round trip, 256 KiB per layer at mbt=16.

    AC-3 + the e2e A/B have now run at this HEAD (opt/m4i9/), which is what
    promoted it. It is a compile-time GRAPH change, so a kernel dir must not be
    reused across values -- the same trap M3-I7 hit with the admission cap.
    """
    return os.environ.get("MPK_FUSE_SILU_QUANT", "1") == "1"


def fuse_norm_quant() -> bool:
    """True when MPK_FUSE_NORM_QUANT=1 fuses each layer's PRE-NORM into the
    quantize that consumes it (M4-I9 flag A). DEFAULT ON since 2026-07-30, but
    ONLY AS PART OF THE STACK -- see the measured caveat below.

    Every layer's chain opens `rms_norm -> quantize -> dense_fp8`, and those two
    front tasks have identical geometry (both one task per token row on the same
    [1, hidden] tile), so they can be one. Modelled cp_exact effect on HEAD:
    -62.0 / -63.3 / -67.8 us at bs 1/8/16 over 40 layers.

    THE MODEL WAS WRONG AND THE MEASUREMENT RULES: solo e2e is 1.016 / 0.997 /
    0.986x at bs 1/8/16 -- a REGRESSION at bs8 and bs16 -- and measured cp_exact
    moves +42 / +64 us where the model predicted -63 / -68. Mechanism (M4-I9's
    durable finding): the projection assumed the host absorbs the victim's work
    for free (`inc = 0`), but `inc` is a function of OCCUPANCY. At bs1 the work
    bound is 0.58x of vLLM's step, the machine is starved, and the model holds to
    within 5%. At bs8/bs16 the work bound is at or above cp, so the host's extra
    work lands on a BUSY worker and -- because the fused task sits on the chain --
    cp GROWS. A fusion's sign inverts with load. What makes flag F the exception
    is that F also WIDENS its victim (quantize moves from 16 tasks/layer onto the
    silu grid's 128); fusing two EQUAL-WIDTH stages buys chain length by spending
    width, which is only free on a starved machine. This flag is on because the
    three-flag stack composes to a win at every batch size; do not enable it alone.

    Applies at the two PRE-NORM sites only:
      * GDN layers    pre_norm -> gdn_xq        (3 outputs: the bf16 norm is
                                                 still read by the `ba` linear)
      * ATTN layers   pre_norm -> qkvg_proj_xq  (2 outputs: the quantize was the
                                                 norm's only consumer)

    NOT applied at the POST-NORM site, and that is a hard graph constraint rather
    than a scope choice: `post_norm -> moe_xq` would make the fused task a
    fork-producer (post_norm still feeds the router, the shared-expert quantize
    and the sigmoid gate) whose consumer `w13` is a JOIN-consumer (it also waits
    on the router), which `build_annotated_graph` rejects as case 3
    (annotated_graph.cc:642-661) -- the same rule that already forces the routed
    and shared branches to carry separate quantizes (see the module docstring,
    note 2). `post_norm -> shared_gate_up_xq` would be legal but is off the
    critical path and tiny.

    Compile-time GRAPH change, so a kernel dir must not be reused across values.
    """
    return os.environ.get("MPK_FUSE_NORM_QUANT", "1") == "1"


def fuse_recur_quant() -> bool:
    """True when MPK_FUSE_RECUR_QUANT=1 fuses the GDN out-projection's quantize
    into the recurrence that produces its input (M4-I9 flag C). DEFAULT ON since
    2026-07-30, as part of the stack (solo e2e 1.018 / 1.002 / 0.999x at bs 1/8/16
    -- neutral-to-flat on its own; the win is the composition, see
    fuse_silu_quant). Its measured cp_exact also inverted vs the model (-43 / +25
    us where the model said -130 / -107) for the occupancy reason documented in
    fuse_norm_quant.

    The largest of M4-I9's remaining admissible fusions: modelled cp_exact effect
    -128.7 / -129.5 / -107.2 us at bs 1/8/16 on top of flag F, over the 30 GDN
    layers.

    Why it is legal, and why it needed the "with care" label. The recurrence's
    gated-RMSNorm epilogue already holds one whole v-head of bf16 results, and
    `linear_value_head_dim == 128` is EXACTLY one fp8 scale group -- so the
    quantize its consumer used to do needs no regrouping and no cross-task
    reduction. In the split configuration (grid.z > 1 at small batch) only the
    LAST-ARRIVING task reaches the epilogue, elected by M3-I3's `__threadfence()`
    + `atomicAdd` arrival counter, so exactly one task writes the group. The
    counter is NOT a barrier -- no task waits on a peer -- which is the property
    that makes it sound under a persistent work-queue scheduler, and the fusion
    inherits it unchanged.

    It also needs no NEW input, which is what makes it possible at all: the
    recurrence already has 7 inputs and `MAX_INPUTS_PER_TASK` is 7.

    The ATTENTION sibling (`o_proj_xq` into `paged_attention`) is deliberately
    NOT implemented. It is worth only 32.4 / 32.9 / 26.4 us of cp_exact, and the
    attention body is being rewritten by a concurrent parity loop -- a fusion
    that lives inside a kernel someone else is replacing is a merge conflict by
    construction, and the composable alternative (a separate quantize task) is
    what already ships.

    Compile-time GRAPH change, so a kernel dir must not be reused across values.
    """
    return os.environ.get("MPK_FUSE_RECUR_QUANT", "1") == "1"


def fp8_dense_baseline() -> bool:
    """True when MPK_FP8_DENSE_BASELINE=1 pins the pre-M4-I2 dispatch.

    The A/B measurement arm: slice 128 here plus the same -D in the JIT flags
    (persistent_kernel.py) makes the generated task-279 code the golden path at
    the old grid, so both arms come from ONE tree and can interleave inside one
    GPU claim. Not a production knob -- leave it unset.
    """
    return os.environ.get("MPK_FP8_DENSE_BASELINE") == "1"


def fp8_slice(output_size: int, reduction_size: int, batched_tokens: int) -> int:
    """Per-task N slice for a preserved-block-scale dense GEMM.

    Falls back to the 128-row scale block -- the pre-M4-I2 dispatch -- for any
    (N, K) the ferret run did not benchmark, and for max_num_batched_tokens > 16,
    where the kernel's fast path is inadmissible (its per-warp B ring assumes
    TILE_M == 16, i.e. one 16-row MMA row-block per warp) and only the golden
    path, which needs whole scale blocks, can run.
    """
    if batched_tokens > 16 or fp8_dense_baseline():
        return BLOCK
    return FP8_DENSE_N_SLICE.get((output_size, reduction_size), BLOCK)


def fp8_grid(output_size: int, reduction_size: int, batched_tokens: int) -> int:
    """Task count for a preserved-block-scale dense GEMM.

    grid.x splits the weight's rows, the output's columns, and the scale's dim 0
    (`persistent_kernel.py`: `linear_fp8_blockscale_layer`). Since M4-I2 the
    per-task slice may be FINER than the checkpoint's 128-row scale block, which
    is why the scale has to be row-replicated first -- see
    `Qwen35Builder._fp8_block_scale`.
    """
    n_slice = fp8_slice(output_size, reduction_size, batched_tokens)
    assert output_size % n_slice == 0, (
        f"preserved-scale FP8 output size {output_size} must be a multiple of "
        f"its per-task slice {n_slice}")
    return output_size // n_slice


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
        # grid.y of BOTH grouped MoE GEMMs (tasks 241/242): how many tasks share
        # one expert's output columns.  This is a GRAPH-WIDTH knob, and M4-I5
        # measured what it is worth: at bs1 the routed GEMMs offer only 16 live
        # tasks per dependency level (8 activated expert groups x 2 splits), so
        # they run 2346 + 1317 us of a 9823 us decode step on 16 of 128 workers,
        # and for 98% / 91% of that span they are the ONLY stage running.  That
        # is the largest single width residual in the graph at the batch size
        # AC-4 binds hardest (bs1, 2.79x vLLM).
        #
        # DEFAULT IS UNCHANGED AT 2.  MPK_MOE_N_SPLITS overrides it before build;
        # it changes the emitted template arguments (per-task OUTPUT_SIZE), so a
        # kernel dir must not be reused across values -- the same trap M3-I7 hit
        # with the admission cap.
        #
        # Legality (asserted in `_build_moe`): the per-task N slice must be a
        # whole number of 128-row checkpoint scale blocks
        # (`moe_fp8_blockscale_sm100.cuh:134`, OUTPUT_SIZE % BLOCK_N == 0), so
        # w13 (N = 2*moe_intermediate = 1024) caps at 8 and w2 (N = hidden =
        # 2048) at 16; the shared knob therefore caps at 8.
        #
        # Soundness under the persistent scheduler -- the M3-I3 test, applied:
        # does this need a barrier?  NO.  grid.y partitions the OUTPUT COLUMNS,
        # so each task owns a disjoint output range, the whole K reduction stays
        # inside one task, and each n-block's accumulator was already
        # independent.  There is no cross-task reduction, hence no arrival
        # counter, no epilogue election and no co-residency requirement -- the
        # property a persistent work-queue scheduler cannot guarantee is not
        # needed here.  Bit-exact by construction for the same reason.
        #
        # Cost, priced: the dispatched task count per call site goes 256 -> 128*k,
        # and a task with no routed rows still costs a queue pop plus a
        # dependency check (measured 0.4 us).  The wave-depth model in
        # opt/m4i5/scripts/ceiling.py charges that and is why k=4 was NEUTRAL at
        # bs8 in M3-I8 (it doubles the worker depth and halves the task) while
        # k=8 need not be.
        self.moe_n_splits = int(os.environ.get("MPK_MOE_N_SPLITS", "2"))
        self.qk_dense_path = "fp8"
        # GDN recurrent decode v-row split (grid.z of task 237) and its
        # cp.async ring depth.  The op is only `num_v_heads * mbr` tasks wide,
        # so at small batch it leaves most of the 128 workers idle; splitting
        # the v-rows multiplies the task count.  Defaults are set per mbr in
        # `_gdn_split_default`, and can be overridden here or with
        # MPK_GDN_SPLIT / MPK_GDN_DEPTH before build (both change the generated
        # code, so a kernel dir must not be reused across values).
        self.gdn_split: Optional[int] = None
        self.gdn_depth = int(os.environ.get("MPK_GDN_DEPTH", "2"))
        # Full-attention `max_tokens_per_pass` (= Q_PASS_SIZE = MAX_TOKENS).
        # See `_attn_q_pass_default` for why this is a perf knob and not just a
        # smem sizing constant; MPK_ATTN_Q_PASS overrides it before build (it
        # changes the generated code, so a kernel dir must not be reused across
        # values).
        self.attn_q_pass = 2
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
        # Per-layer handoff for M4-I9 flag A: the (xq, xs) pair the fused
        # pre-norm produced, or None when the flag is off.
        self._pre_xq = None

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
        # Decode v-row-split scratch, shared by ALL GDN layers (layer L+1 is
        # transitively downstream of layer L, so two uses are never in flight).
        # Row = [head_v_dim fp32 `o` partials | arrival counter | padding]; the
        # counter is read as `unsigned int`, starts at 0 here and self-resets,
        # and the pad keeps every row 16 B aligned. Untouched when gdn_split==1.
        self.gdn_split_scratch = torch.zeros(
            (self.mbr, c.linear_num_value_heads, c.linear_value_head_dim + 8),
            dtype=torch.float32, device="cuda")
        self._keep += [self.conv_state, self.recurrent_state,
                       self.gdn_split_scratch]

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

    def _gdn_split_default(self) -> int:
        """grid.z for task 237: how many tasks share one (head, slot) decode.

        The op emits `num_v_heads * mbr` tasks per layer against 128 workers
        (`utils.get_configurations_from_gpu`), so at small batch it runs a
        fraction of the machine: 32 tasks at mbr=1. Splitting the v-rows over
        `split` cooperating tasks multiplies that, at the cost of one extra
        global round trip for the `o` partials and a slightly shorter per-task
        row loop. The defaults aim for roughly one full 128-worker wave; they
        were swept in-MPK rather than copied from the standalone kernel, whose
        8-blocks-per-SM occupancy regime does not exist here (MPK runs ONE
        worker CTA per SM).

        0 means "decode fast path OFF": the task falls back to the original
        unsplit implementation with grid.z == 1. That is the A/B base arm, and it
        lives in the SAME build so both arms can be measured in one window
        without swapping git trees (see `gdn_recurrent_layer`).
        """
        if self.gdn_split is not None:
            return int(self.gdn_split)
        env = os.environ.get("MPK_GDN_SPLIT")
        if env is not None and env != "":
            return int(env)
        heads = self.config.linear_num_value_heads
        target_workers = 128
        split = max(1, target_workers // max(1, heads * self.mbr))
        # must divide head_v_dim and stay a power of two for the row slicing
        while split > 1 and self.config.linear_value_head_dim % split != 0:
            split //= 2
        return split

    def _attn_q_pass_default(self) -> int:
        """`max_tokens_per_pass` (= Q_PASS_SIZE) for the full-attention task.

        P3 measured the post-5715c6f smem arena at head_dim 256 / GQA 8:1 as
        admissible only up to 4 queries per pass; the in-task Q-loop covers
        longer chunks (v1-architecture.md §4.3).  Within that ceiling the value
        is a real perf knob, because `task_register.cc` sets MAX_TOKENS from it
        and the kernel derives

            MMA_ITERS_M = ceil(MAX_TOKENS * NUM_QO_PER_KV / 16) = ceil(Q_PASS/2)

        at 16 q / 2 kv heads.  The per-thread accumulator is
        `float o[MMA_ITERS_M][HEAD_DIM/16][8]` = MMA_ITERS_M * 128 floats, so
        Q_PASS 3-4 asks for 256 floats and does not fit the 255-register file.
        Because MPK inlines every task body into ONE `persistent_kernel`, that
        overflow is not local to attention -- ptxas allocates a single register
        budget and stack frame for the whole megakernel (M3-I6a, `ptxas -v` on
        the generated TU):

            Q_PASS   registers   stack frame   spill st / ld
              4         255         576 B       780 / 976 B
              2         240         144 B          0 /   0 B
              1         238         144 B          0 /   0 B

        i.e. the attention accumulator was the ONLY source of spilling in the
        megakernel.  The KV loop's rescale step touches every `o` element once
        per KV tile, so at Q_PASS=4 that spill was paid per KV tile and its cost
        therefore grew with decode context.  Measured at the pinned 256/1024
        geometry (256-token prompt, msl=897, decode context ~801-896), attention
        wallspan per step fell 1447 -> 752 us at bs1 (-48.0%) and 1491 -> 724 us
        at bs8 (-51.4%); the marginal cost per KV token fell 0.1249 -> 0.0536 us
        (R^2 >= 0.999).  Relieving the shared budget also sped up families this
        knob does not touch -- dense fp8 2939 -> 2811 us/step, GDN recurrent
        234 -> 219 -- so the whole decode step fell ~9-10%.

        2 rather than 1: both are 0-spill and their decode cost is within 1.7%,
        but Q_PASS=1 halves the pass length and so doubles the pass COUNT of
        every prefill chunk (16 passes for an mbt=16 chunk instead of 8), which
        measured 20-24% worse attention wallspan in every prefill-containing
        window.  2 also fills the m16n16k16 tile exactly (2 tokens * 8 q-heads
        = 16 rows), so no MMA rows are wasted during prefill.

        `MPK_ATTN_Q_PASS` overrides it so both arms live in one tree and can be
        swept without swapping git trees (same idiom as `MPK_GDN_SPLIT`).
        """
        env = os.environ.get("MPK_ATTN_Q_PASS")
        if env is not None and env != "":
            v = int(env)
            assert 1 <= v <= 4, (
                f"MPK_ATTN_Q_PASS={v} outside the smem-admissible range 1..4 "
                "(P3: head_dim 256 / GQA 8:1 blows MAX_DYNAMIC_SHARED_MEMORY at "
                "MAX_TOKENS >= 6; the kernel static_asserts it)")
            return min(v, self.mbt)
        return min(self.attn_q_pass, self.mbt)

    def _attach_common(self):
        pk, c = self.mpk, self.config
        self.cos_dt = pk.attach_input(self.cos_table, name="cos_position_embedding")
        self.sin_dt = pk.attach_input(self.sin_table, name="sin_position_embedding")
        # One shared decode-split scratch for every GDN layer (see
        # `_alloc_state_pools`); attach it once, like the RoPE tables.
        self.gdn_split_dt = pk.attach_input(self.gdn_split_scratch,
                                            name="gdn_split_scratch")
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
    def _fp8_block_scale(self, w_name: str, n_tasks: int):
        """Attach `weight_scale_inv` with exactly ONE row per task.

        The checkpoint ships [N/128, K/128]: one fp32 scale per 128x128 weight
        block. When the per-task N slice is finer than 128 rows, 128//slice tasks
        share a scale block -- and MPK's grid split cannot express that, because
        it divides dim0 by grid.x with INTEGER division (runtime.cc), which would
        silently hand every task row 0. So the rows are REPLICATED to one per
        task: `repeat_interleave` copies the same fp32 values, so the numbers the
        kernel promotes with are bit-identical to the checkpoint's, and the
        ordinary split then gives each task its containing block row.

        Cost is negligible -- the whole dense scale set is ~340 KiB before
        replication -- and the replicated tensor is kept alive by
        `attach_input`'s own reference list.
        """
        pk = self.mpk
        raw = self.weights[w_name + "_scale"]
        assert raw.dim() == 2, f"{w_name}_scale must be [N/128, K/128]"
        n_blocks = raw.shape[0]
        if n_blocks == n_tasks:
            return pk.attach_input(raw, name=w_name + "_scale")
        assert n_tasks % n_blocks == 0, (
            f"{w_name}: {n_tasks} tasks do not partition {n_blocks} scale "
            f"blocks evenly")
        rep = n_tasks // n_blocks
        rt = raw.repeat_interleave(rep, dim=0).contiguous()
        return pk.attach_input(rt, name=f"{w_name}_scale_x{rep}")

    def _fp8_linear(self, name: str, x, w_name: str, out, *, residual=None,
                    k: int):
        """quantize(x) -> preserved-block-scale fp8 GEMM (optionally +residual).

        A dedicated quantize task per call site: sharing one would make the
        quantize a fork-producer, and any consumer that also waits on another
        producer (the router, for the MoE branches) then trips case 3.
        """
        pk = self.mpk
        xq = self._t((self.mbt, k), float8_e4m3, f"{name}_xq")
        xs = self._t((self.mbt, k // BLOCK), float32, f"{name}_xs")
        pk.quantize_fp8_layer(input=x, output_fp8=xq, output_scale=xs,
                              grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1),
                              scale_ue8m0=False, row_partition=QUANTIZE_ROW_SPLIT)
        self._fp8_gemm(name, xq, xs, w_name, out, residual=residual, k=k)

    def _fp8_gemm(self, name: str, xq, xs, w_name: str, out, *, residual=None,
                  k: int):
        """The GEMM half of `_fp8_linear`, split out so a call site whose
        (xq, xs) came from somewhere else -- M4-I9 flag A's fused pre-norm, or
        flag C's fused producer -- can wire it without a second quantize."""
        pk, w = self.mpk, self.weights
        weight = pk.attach_input(w[w_name], name=w_name)
        n = w[w_name].shape[0]
        n_tasks = fp8_grid(n, k, self.mbt)
        scale = self._fp8_block_scale(w_name, n_tasks)
        pk.linear_fp8_blockscale_layer(
            input_fp8=xq, input_scale=xs, weight_fp8=weight, weight_scale=scale,
            output=out, grid_dim=(n_tasks, 1, 1), block_dim=(256, 1, 1),
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
        # The argmax split must DIVIDE padded_vocab_size, and that is a stricter
        # requirement than "use every worker". The codegen instantiates
        # argmax_{partial,reduce}_sm100_kernel<T, mbt, CHUNK_SIZE, N> with
        # CHUNK_SIZE = padded_vocab_size / N computed by FLOOR division, and the
        # scan loop runs to CHUNK_SIZE with no bound against the true vocab. So
        # when N does not divide the vocab, N*CHUNK_SIZE < padded_vocab_size and
        # the tail token ids are silently never scanned, while every task's slice
        # offset shifts against the real row stride. Measured: at N=136 the
        # codegen emits CHUNK_SIZE=1825, tiling only 1825*136 = 248200 of 248320
        # ids -- decode then produced a deterministic WRONG token from step 0 on
        # 9 of 10 AC-3 prompts. At N=128 it emits 1940 and 1940*128 = 248320
        # exactly, which is the only reason the shipped configuration is correct.
        #
        # So pick the largest divisor of padded_vocab_size that is <= num_workers
        # rather than num_workers itself. 248320 = 2^9 * 5 * 97, whose largest
        # divisor <= 136 is 128 -- i.e. this keeps today's behaviour at 128
        # workers bit-for-bit while letting the megakernel widen past it. The
        # argmax is a tiny fraction of the step, so under-using a few workers
        # here costs far less than the width buys everywhere else.
        n_argmax = self._argmax_split(pk.num_workers)
        part_v = self._t((self.mbt, n_argmax), bfloat16, "argmax_part_value")
        part_i = self._t((self.mbt, n_argmax), int64, "argmax_part_index")
        pk.argmax_partial_layer(input=logits, output=(part_v, part_i),
                                grid_dim=(n_argmax, 1, 1),
                                block_dim=(128, 1, 1))
        pk.argmax_reduce_layer(input=(part_v, part_i), output=self.output_tokens,
                               grid_dim=(1, 1, 1), block_dim=(128, 1, 1))

    def _argmax_split(self, num_workers: int) -> int:
        """Largest divisor of ``padded_vocab_size`` that is <= ``num_workers``.

        See the call site for why an exact divisor is required rather than
        merely convenient: the argmax codegen floor-divides and its scan loop
        has no bound against the true vocab, so a non-divisor silently drops the
        tail ids and misaligns every slice.

        Returns ``num_workers`` unchanged whenever it already divides, so the
        128-worker configuration is untouched.
        """
        v = self.padded_vocab_size
        for n in range(min(num_workers, v), 0, -1):
            if v % n == 0:
                assert v // n >= 1
                return n
        raise AssertionError(f"no argmax split found for vocab {v}")

    def _build_layer(self, i: int, h):
        pk, w, c = self.mpk, self.weights, self.config
        pre_w = pk.attach_input(w[f"layer_{i}_input_layernorm"],
                                name=f"layer_{i}_input_layernorm")
        is_gdn = c.layer_types[i] == "linear_attention"

        # M4-I9 flag A: fuse the pre-norm into the quantize that consumes it.
        # `_pre_xq` is the (xq, xs) pair the fused task produced, or None; the
        # branch builders skip their own quantize when it is not None.
        self._pre_xq = None
        if fuse_norm_quant() and not self.expose_intermediates:
            k = c.hidden_size
            tag = f"layer_{i}_gdn" if is_gdn else f"layer_{i}_qkvg_proj"
            xq = self._t((self.mbt, k), float8_e4m3, f"{tag}_xq")
            xs = self._t((self.mbt, k // BLOCK), float32, f"{tag}_xs")
            # The bf16 norm is still materialised ONLY where something else
            # reads it: at GDN layers the bf16 `ba` projection does.
            nrm = (self._t((self.mbt, c.hidden_size), bfloat16,
                           f"layer_{i}_pre_norm") if is_gdn else None)
            pk.rmsnorm_quantize_fp8_layer(
                input=h, weight=pre_w, output_fp8=xq, output_scale=xs,
                norm_output=nrm, grid_dim=(self.mbt, 1, 1),
                block_dim=(128, 1, 1))
            self._pre_xq = (xq, xs)
        else:
            nrm = self._t((self.mbt, c.hidden_size), bfloat16,
                          f"layer_{i}_pre_norm")
            pk.rmsnorm_layer(input=h, weight=pre_w, output=nrm,
                             grid_dim=(self.mbt, 1, 1), block_dim=(128, 1, 1))

        if is_gdn:
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
        # M4-I9 flag A may already have produced this pair inside the pre-norm.
        if self._pre_xq is not None:
            xq, xs = self._pre_xq
        else:
            xq = self._t((self.mbt, c.hidden_size), float8_e4m3,
                         f"layer_{i}_gdn_xq")
            xs = self._t((self.mbt, c.hidden_size // BLOCK), float32,
                         f"layer_{i}_gdn_xs")
            pk.quantize_fp8_layer(input=nrm, output_fp8=xq, output_scale=xs,
                                  grid_dim=(self.mbt, 1, 1),
                                  block_dim=(128, 1, 1), scale_ue8m0=False,
                                  row_partition=QUANTIZE_ROW_SPLIT)
        qkv = self._t((self.mbt, c.conv_dim), bfloat16, f"layer_{i}_gdn_qkv")
        z = self._t((self.mbt, c.gdn_z_dim), bfloat16, f"layer_{i}_gdn_z")
        for tag, out in (("in_proj_qkv", qkv), ("in_proj_z", z)):
            name = f"layer_{i}_gdn_{tag}"
            n_tasks = fp8_grid(w[name].shape[0], w[name].shape[1], self.mbt)
            pk.linear_fp8_blockscale_layer(
                input_fp8=xq, input_scale=xs,
                weight_fp8=pk.attach_input(w[name], name=name),
                weight_scale=self._fp8_block_scale(name, n_tasks),
                output=out, grid_dim=(n_tasks, 1, 1),
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

        # M4-I9 flag C: the recurrence emits the fp8 pair its consumer needed,
        # and the bf16 `gdn_out` (whose only consumer was that quantize) is not
        # materialised at all.
        fuse_rq = fuse_recur_quant() and not self.expose_intermediates
        if fuse_rq:
            g_out = None
            oq = self._t((self.mbt, c.gdn_z_dim), float8_e4m3,
                         f"layer_{i}_gdn_out_proj_xq")
            os_ = self._t((self.mbt, c.gdn_z_dim // BLOCK), float32,
                          f"layer_{i}_gdn_out_proj_xs")
        else:
            g_out = self._t((self.mbt, c.gdn_z_dim), bfloat16,
                            f"layer_{i}_gdn_out")
            oq = os_ = None
        gdn_split = self._gdn_split_default()
        pk.gdn_recurrent_layer(
            qkv=qkv_c, ba=ba,
            alog_dtbias=pk.attach_input(w[f"layer_{i}_gdn_alog_dtbias"],
                                        name=f"layer_{i}_gdn_alog_dtbias"),
            state=pk.attach_input(self.recurrent_state[slot],
                                  name=f"layer_{i}_gdn_state"),
            z=z,
            norm_w=pk.attach_input(w[f"layer_{i}_gdn_norm"],
                                   name=f"layer_{i}_gdn_norm"),
            output=g_out,
            output_fp8=oq, output_scale=os_,
            split_scratch=self.gdn_split_dt,
            num_k_heads=c.linear_num_key_heads,
            grid_dim=(c.linear_num_value_heads, self.mbr,
                      max(1, gdn_split)),
            block_dim=(256, 1, 1),
            depth=self.gdn_depth,
            decode_fastpath=gdn_split > 0)

        out = self._t((self.mbt, c.hidden_size), bfloat16, f"layer_{i}_attn_resid")
        if fuse_rq:
            self._fp8_gemm(f"layer_{i}_gdn_out_proj", oq, os_,
                           f"layer_{i}_gdn_out_proj", out, residual=h,
                           k=c.gdn_z_dim)
        else:
            self._fp8_linear(f"layer_{i}_gdn_out_proj", g_out,
                             f"layer_{i}_gdn_out_proj", out, residual=h,
                             k=c.gdn_z_dim)
        return out

    # ---- full attention ----------------------------------------------
    def _build_attention(self, i: int, nrm, h):
        pk, w, c = self.mpk, self.weights, self.config
        slot = self._attn_slot[i]
        qkvg = self._t((self.mbt, c.qkvg_dim), bfloat16, f"layer_{i}_qkvg")
        if self.qk_dense_path == "fp8" and self._pre_xq is not None:
            # M4-I9 flag A: the pre-norm already emitted this site's (xq, xs),
            # so only the GEMM half of `_fp8_linear` is left to wire.
            self._fp8_gemm(f"layer_{i}_qkvg_proj", self._pre_xq[0],
                           self._pre_xq[1], f"layer_{i}_qkvg_proj", qkvg,
                           k=c.hidden_size)
        elif self.qk_dense_path == "fp8":
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
            max_tokens_per_pass=self._attn_q_pass_default())

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
            round_weights_to_input_dtype=True,
            gate_padding_rows=MOE_GATE_PADDING_ROWS)

        # ---- routed experts ---------------------------------------------
        rq = self._t((mbt, c.hidden_size), float8_e4m3, f"layer_{i}_moe_xq")
        rs = self._t((mbt, c.hidden_size // BLOCK), float32, f"layer_{i}_moe_xs")
        pk.quantize_fp8_layer(input=x, output_fp8=rq, output_scale=rs,
                              grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
                              scale_ue8m0=False, row_partition=QUANTIZE_ROW_SPLIT)
        grid_x = min(c.num_experts, mbt * topk)
        # `moe_n_splits` is grid.y on both grouped GEMMs; the per-task N slice
        # must stay a whole 128-row checkpoint scale block
        # (moe_fp8_blockscale_sm100.cuh:134).
        for _nm, _n in (("w13", 2 * inter), ("w2", c.hidden_size)):
            assert _n % (self.moe_n_splits * BLOCK) == 0, (
                f"moe_n_splits={self.moe_n_splits} is illegal for {_nm}: "
                f"N={_n} must split into whole {BLOCK}-row scale blocks "
                f"(max split for {_nm} is {_n // BLOCK})")
        mid = self._t((mbt, topk, 2 * inter), bfloat16, f"layer_{i}_moe_mid")
        pk.moe_fp8_blockscale_layer(
            input_fp8=rq, input_scale=rs,
            weight_fp8=pk.attach_input(w[f"layer_{i}_w13"], name=f"layer_{i}_w13"),
            weight_scale=pk.attach_input(w[f"layer_{i}_w13_scale"],
                                         name=f"layer_{i}_w13_scale"),
            moe_routing_indices=routing, moe_mask=mask, output=mid,
            grid_dim=(grid_x, self.moe_n_splits, 1), block_dim=(256, 1, 1),
            w13_linear=True)
        # M4-I9: SwiGLU + activation quantize as ONE task (default off). The
        # unfused pair is kept for `expose_intermediates`, because the
        # single-layer test-mode gates and M2-I9's divergence bisection read
        # `layer_i_moe_act` as a probe point and the fused task never
        # materialises it.
        #
        # The `else` branch declares its tensors in the ORIGINAL order (act,
        # then actq/acts) rather than hoisting the shared pair above the `if`.
        # Tensor creation order is device-allocation order in the generated TU,
        # so hoisting made the shipped-default TU differ from pre-M4-I9 by 240
        # lines of moved `cudaMalloc`s -- a difference with no reason to exist.
        # Kept this way so gate 1c can assert the strong form: byte-identical.
        if fuse_silu_quant() and not self.expose_intermediates:
            aq = self._t((mbt, topk, inter), float8_e4m3, f"layer_{i}_moe_actq")
            as_ = self._t((mbt, topk, inter // BLOCK), float32,
                          f"layer_{i}_moe_acts")
            pk.moe_silu_mul_quantize_fp8_layer(
                input=mid, output_fp8=aq, output_scale=as_,
                grid_dim=(mbt, topk, 1), block_dim=(128, 1, 1))
        else:
            act = self._t((mbt, topk, inter), bfloat16, f"layer_{i}_moe_act")
            pk.moe_silu_mul_layer(input=mid, output=act, grid_dim=(mbt, topk, 1),
                                  block_dim=(128, 1, 1))
            aq = self._t((mbt, topk, inter), float8_e4m3, f"layer_{i}_moe_actq")
            as_ = self._t((mbt, topk, inter // BLOCK), float32,
                          f"layer_{i}_moe_acts")
            pk.quantize_fp8_layer(input=act, output_fp8=aq, output_scale=as_,
                                  grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
                                  scale_ue8m0=False,
                                  row_partition=QUANTIZE_ROW_SPLIT)
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
