"""Qwen3 built from the low-level IR instead of from a call order.

builder.py decides where every task begins and ends by the order it calls
mpk.*_layer: each call IS a task, bound to a hand-written .cuh. This file holds
the same computation as `mirage.mpk.lowering` nodes and lowers it, so the task
boundaries are derived from dataflow and every task body is compiler-generated.

Three stages, in the order they run:

    build_qwen3      the model as SSA nodes -- shapes only, no tiling, no
                     boundaries
    partition_as_today   where the boundaries go: the same cuts builder.py's
                     call order implies, so the two can be compared directly
    build            bind HF weights and hand the whole thing to lower()

Four nodes stay OPAQUE. Three have no muGraph op at all -- embedding (a
gather), the KV-cache append inside attention prep (a stateful scatter), and
argmax (a reduction to indices). The fourth, rmsnorm, has one but is opaque by
default (see rmsnorm below). They are still nodes, so the graph stays connected
and a partition can see the boundary; lower() hands each to a registered
handler instead of to search.

Entry point:

    graph, groups = plan(shapes)
    bindings, raw = bind_weights(pk, model, shapes, ...)
    graph, groups, env = build(pk, shapes, bindings, meta, planned=(graph, groups))
"""
from __future__ import annotations

import dataclasses
import os

from ...lowering import (MATMUL_K_TILES, MATMUL_N_TILES, default_forloop,
                         forloop_candidates, grid_candidates, has_rms_norm,
                         is_opaque, lower, make_group)
from ...lowering.node import ModelGraph, Value


@dataclasses.dataclass
class Qwen3Shapes:
    """Everything the graph needs; all of it is in the HF config."""
    tokens: int
    hidden: int
    intermediate: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    vocab: int
    vocab_padded: int = 0
    max_seq: int = 4096

    @property
    def out_vocab(self) -> int:
        return self.vocab_padded or self.vocab

    @property
    def qkv_dim(self) -> int:
        return (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim

    @property
    def attn_dim(self) -> int:
        return self.num_q_heads * self.head_dim

    @classmethod
    def from_hf(cls, cfg, tokens: int) -> "Qwen3Shapes":
        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads)
        return cls(tokens=tokens, hidden=cfg.hidden_size,
                   intermediate=cfg.intermediate_size,
                   num_layers=cfg.num_hidden_layers,
                   num_q_heads=cfg.num_attention_heads,
                   num_kv_heads=cfg.num_key_value_heads,
                   head_dim=head_dim, vocab=cfg.vocab_size)


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------

def rmsnorm(g: ModelGraph, x: Value, w: Value) -> Value:
    """Qwen3's norm is rms_norm(x) * weight, opaque by default.

    MPK_GRAPH_RMSNORM=1 emits it as two real ops instead, which
    partition_as_today already pairs back into one "rmsnorm" group -- the
    pairing rule was written for exactly this. Off by default until it is
    measured, so the default graph path is unchanged.

    Search CAN schedule rms_norm * w: re-measured at (8, 1024), it returns one
    usable task schedule. A BARE rms_norm is the one that does not reliably
    work (1 run in 3), and Qwen3 never needs one.

    When opaque it stays ONE node rather than two, because the two halves are
    only ever a task together.
    """
    if os.environ.get("MPK_GRAPH_RMSNORM") == "1":
        return g.mul(g.rms_norm(x), w)
    return g.opaque("rmsnorm", [x, w], x.dims)


def mlp(g: ModelGraph, x: Value, wg: Value, wu: Value, wd: Value) -> Value:
    """silu(x @ Wg) * (x @ Wu) @ Wd.

    Both projections are emitted before the activation so silu and mul land
    adjacent, which is the shape MPK already fuses as silu_mul_layer.
    """
    gate = g.matmul(x, wg)
    up = g.matmul(x, wu)
    return g.matmul(g.mul(g.silu(gate), up), wd)


def build_layer(g: ModelGraph, x: Value, w: dict, s: Qwen3Shapes,
                layer: int) -> Value:
    """One decoder layer. `w` holds this layer's weights by short name."""
    with g.scope(layer=layer, tag=f"l{layer}.attn"):
        h = rmsnorm(g, x, w["in_norm"])
        qkv = g.matmul(h, w["qkv"])
        # qk-norm, RoPE, the KV-cache append and the attention core, as one
        # hand-written task. Everything it reads is named here so a lowering
        # handler can find it in group.external_inputs rather than having to
        # know the model.
        attn = g.opaque("attention",
                        [qkv, w["q_norm"], w["k_norm"], w["cos"], w["sin"],
                         w["k_cache"], w["v_cache"]],
                        (s.tokens, s.attn_dim), layer=layer)
        x = g.add(x, g.matmul(attn, w["o"]))

    with g.scope(layer=layer, tag=f"l{layer}.mlp"):
        h = rmsnorm(g, x, w["post_norm"])
        x = g.add(x, mlp(g, h, w["gate"], w["up"], w["down"]))
    return x


def build_qwen3(s: Qwen3Shapes, *, num_layers: int | None = None) -> ModelGraph:
    """The whole model. num_layers trims it for tests."""
    n = s.num_layers if num_layers is None else num_layers
    g = ModelGraph("qwen3")

    tokens = g.new_input((s.tokens, 1), "input_tokens", role="feed")
    embed_w = g.new_input((s.vocab, s.hidden), "embed")
    with g.scope(tag="embed"):
        x = g.opaque("embedding", [tokens, embed_w], (s.tokens, s.hidden))

    for i in range(n):
        w = {
            "in_norm":   g.new_input((1, s.hidden), f"l{i}.in_norm"),
            "qkv":       g.new_input((s.hidden, s.qkv_dim), f"l{i}.qkv"),
            # 1-D: paged_attention_layer asserts num_dims == 1 for these.
            "q_norm":    g.new_input((s.head_dim,), f"l{i}.q_norm"),
            "k_norm":    g.new_input((s.head_dim,), f"l{i}.k_norm"),
            "cos":       g.new_input((s.max_seq, s.head_dim), "cos"),
            "sin":       g.new_input((s.max_seq, s.head_dim), "sin"),
            "k_cache":   g.new_input((1, 1), f"l{i}.k_cache", role="feed"),
            "v_cache":   g.new_input((1, 1), f"l{i}.v_cache", role="feed"),
            "o":         g.new_input((s.attn_dim, s.hidden), f"l{i}.o"),
            "post_norm": g.new_input((1, s.hidden), f"l{i}.post_norm"),
            "gate":      g.new_input((s.hidden, s.intermediate), f"l{i}.gate"),
            "up":        g.new_input((s.hidden, s.intermediate), f"l{i}.up"),
            "down":      g.new_input((s.intermediate, s.hidden), f"l{i}.down"),
        }
        x = build_layer(g, x, w, s, i)

    with g.scope(tag="head"):
        x = rmsnorm(g, x, g.new_input((1, s.hidden), "final_norm"))
        logits = g.matmul(x, g.new_input((s.hidden, s.out_vocab), "lm_head"))
        out = g.opaque("argmax", [logits], (s.tokens, 1))
    g.mark_output(out)
    return g


# ---------------------------------------------------------------------------
# Where the boundaries go
# ---------------------------------------------------------------------------

def partition_as_today(g: ModelGraph) -> list:
    """One group per node, except silu+mul which MPK already fuses into
    silu_mul_layer.

    This is builder.py's call order written as a partition, which is what makes
    the graph path comparable to it: same cuts, generated bodies.
    """
    # Pair by dataflow, not by position: the only consumer of a silu is its
    # mul, and of an rms_norm its weight multiply.
    pair_with, tag_of = {}, {}
    for i, n in enumerate(g.nodes):
        if n.op not in ("silu", "rms_norm"):
            continue
        cons = g.consumers(n.output)
        if len(cons) == 1 and g.nodes[cons[0]].op == "mul":
            pair_with[i] = cons[0]
            tag_of[i] = "silu_mul" if n.op == "silu" else "rmsnorm"

    absorbed = set(pair_with.values())
    groups = []
    for i, n in enumerate(g.nodes):
        if i in absorbed:
            continue
        if i in pair_with:
            groups.append(make_group(g, [i, pair_with[i]], tag_of[i]))
        elif is_opaque(n.op):
            groups.append(make_group(g, [i], n.op.split(":", 1)[1]))
        else:
            groups.append(make_group(g, [i], n.op))
    return groups


# ---------------------------------------------------------------------------
# The codegen knobs, as env-selected candidates
# ---------------------------------------------------------------------------

def grid_from_env(graph):
    """The grid each group is given, from MPK_MATMUL_N_TILE / MPK_NORM_ROWS.

    MPK_MATMUL_N_TILE is the N tile a matmul group gets, which becomes the
    MMA's M under swapAB -- so 64 (what every hand-written generated_* layer
    uses, and the default) or 128 (what linear_sm100 hardcodes). Wider means a
    bigger MMA and fewer blocks: N=3072 is 48 blocks at 64 and 24 at 128.
    Measured, 64 wins by 5.3%, so halving the parallelism costs more than the
    wider instruction buys.

    MPK_NORM_ROWS is the other kind of group. A group that REDUCES over its
    last dim cannot split that dim -- a 64-wide slice would ask each block to
    normalise a sixteenth of a row, a different computation -- so it splits
    ROWS instead, and this says how many rows one block takes. It pulls two
    ways: tb::ReductionKernel gives each output a warp, so a block holding one
    row keeps one warp of eight busy while a block holding eight keeps all
    eight -- but rows per block is also blocks per task, and 8 blocks of the
    148 workers beats 1. The default is 1 row per block; task_search's
    rows_grid_candidates enumerates the rest.

    Neither is a rule. Which way each lands is whole-model throughput, so both
    are env knobs to be swept.
    """
    tile = int(os.environ.get("MPK_MATMUL_N_TILE", "64"))
    if tile not in MATMUL_N_TILES:
        raise ValueError(f"MPK_MATMUL_N_TILE must be one of "
                         f"{MATMUL_N_TILES}, got {tile}")
    norm_rows = int(os.environ.get("MPK_NORM_ROWS", "1"))

    def pick(group):
        # grid_candidates already says which grids are legal for this kind of
        # group. A group the requested value does not divide keeps cands[0] --
        # the default for its kind -- rather than failing the whole lowering.
        cands = grid_candidates(graph, group)
        if has_rms_norm(graph, group):
            want = (max(1, group.output.dims[0] // norm_rows), 1, 1)
        else:
            want = (group.output.dims[-1] // tile, 1, 1)
        return want if want in cands else cands[0]

    return pick


def knobs_from_env(graph):
    """The per-group K tile and pipeline depth, from the env.

    Both are enumerated in lowering/task_search.py; the defaults here are what
    the sweep measured, whole-model, on one card:

      MPK_MATMUL_K_TILE   64 | 128 (default) | 256
      MPK_MATMUL_STAGES   2..8, 4 (default)

                 stages=2   stages=3   stages=4   stages=5   stages=6  stages=8
      K=64        3445.8         --     3624.0         --         --    3615.4
      K=128       3773.8         --     3901.0     3894.6     3899.4  over smem
      K=256       3853.6     3896.3   over smem   over smem  over smem over smem

    Two things that were NOT obvious before measuring. A K tile of at least 128
    is the whole lever -- 64 tops out near 3620 however deep the pipeline goes,
    so the two knobs are not interchangeable even though they spend the same
    shared memory. And past 4 stages nothing is bought: 4, 5 and 6 differ by
    0.2%, less than the 0.06% run-to-run spread times three. The default is
    therefore the cheapest point on the flat part, not the deepest that fits.

    The over-smem cells fail at registration in seconds (register_generated_
    task's budget check), not as an illegal address after a full megakernel
    compile.
    """
    k_tile = int(os.environ.get("MPK_MATMUL_K_TILE", "128"))
    if k_tile not in MATMUL_K_TILES:
        raise ValueError(f"MPK_MATMUL_K_TILE must be one of {MATMUL_K_TILES}, "
                         f"got {k_tile}")
    stages = int(os.environ.get("MPK_MATMUL_STAGES", "4"))
    if not 2 <= stages <= 8:
        raise ValueError(f"MPK_MATMUL_STAGES must be in [2, 8], got {stages}")

    def forloop_for(group):
        want = default_forloop(graph, group, k_tile)
        # A group whose K does not divide the requested tile keeps the default
        # rather than failing the whole lowering.
        cands = forloop_candidates(graph, group)
        return want if want in cands else default_forloop(graph, group)

    def stages_for(group):
        return stages

    return forloop_for, stages_for


# ---------------------------------------------------------------------------
# The tasks the graph does not generate
# ---------------------------------------------------------------------------

def opaque_handlers(pk, shapes: Qwen3Shapes, meta: dict):
    """A handler per opaque op. Each is handed the group and its already-bound
    input DTensors, in the order the graph names them."""

    def embedding(pk, group, ins, out):
        tokens, weight = ins
        pk.embed_layer(input=tokens, weight=weight, output=out,
                       grid_dim=(shapes.tokens, 1, 1), block_dim=(128, 1, 1))

    def rmsnorm(pk, group, ins, out):
        x, w = ins
        pk.rmsnorm_layer(input=x, weight=w, output=out,
                         grid_dim=(shapes.tokens, 1, 1), block_dim=(128, 1, 1))

    def attention(pk, group, ins, out):
        qkv, q_norm, k_norm, cos, sin, k_cache, v_cache = ins
        pk.paged_attention_layer(
            input=qkv, k_cache=k_cache, v_cache=v_cache,
            q_norm=q_norm, k_norm=k_norm,
            cos_pos_embed=cos, sin_pos_embed=sin, output=out,
            grid_dim=(pk.max_num_batched_requests, shapes.num_kv_heads, 1),
            block_dim=(128, 1, 1))

    def argmax(pk, group, ins, out):
        (logits,) = ins
        pk.argmax_partial_layer(
            input=logits, output=(meta["argmax_value"], meta["argmax_index"]),
            grid_dim=(pk.num_workers, 1, 1), block_dim=(128, 1, 1))
        pk.argmax_reduce_layer(
            input=(meta["argmax_value"], meta["argmax_index"]), output=out,
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1))

    return {"embedding": embedding, "rmsnorm": rmsnorm,
            "attention": attention, "argmax": argmax}


def down_projection_fallback(pk, ins, out, grid):
    """Qwen3's down projection is (T,3072)@(3072,1024). Search never fuses a
    matmul whose K is not a power of two, so this one keeps a hand-written
    SCHEDULE -- the body is still generated."""
    x, w = ins
    pk.generated_linear_layer(input=x, weight_t=w, output=out, grid_dim=grid,
                              block_dim=(256, 1, 1),
                              forloop_range=x.dim(1) // 64)


def pick_fallback(group):
    """What a matmul group falls back to when search finds NOTHING.

    Only the down projection needs one, and every matmul group carries the same
    tag, so decide on shape: K not a power of two is what search refuses."""
    if len(group.external_inputs) != 2:
        return None
    k = group.external_inputs[0].dims[-1]
    return down_projection_fallback if k & (k - 1) else None


def make_override(raw_weights=None, hand_written=()):
    """Which matmul groups skip search and take the hand-written linear task.

    Search succeeding is not search winning. The lm_head searches fine, but
    default_grid gives it one 64-wide output column per block -- 2374 blocks
    for a padded vocab -- where linear_sm100 splits N across num_workers (148)
    itself. That is the single largest cost difference between this path and
    the imperative one.

    The hand-written task reads its weight as (N, K), the raw HF layout, while
    a searched matmul reads (K, N), so an overridden weight is bound raw.
    """
    raw_weights = raw_weights or {}

    def pick(group):
        if len(group.external_inputs) not in (2, 3):
            return None
        name = group.external_inputs[1].name
        if name not in raw_weights or name not in hand_written:
            return None
        if len(group.nodes) != 1:
            # linear_layer computes out = x @ W and nothing else, so it cannot
            # stand in for a group that also does something to the result.
            raise ValueError(
                f"MPK_HANDWRITTEN names {name!r}, but its group {group.tag!r} "
                f"fuses {len(group.nodes)} ops; the hand-written linear task "
                f"can only replace a lone matmul. Use a partition that leaves "
                f"it unfused, or drop it from MPK_HANDWRITTEN.")
        w = raw_weights[name]

        # One block per worker, NOT the imperative builder's
        # grid_for_rmsnorm_linear_layer. That rule looks like the right one --
        # it is the same task, and linear_sm100 hardcodes MMA_M = 128, so its
        # 64 blocks x 64 columns for qkv waste less of each MMA than 128 x 32.
        # Measured, it LOSES: 5051.8 / 5108.2 against 5278.1 / 5275.7 for
        # num_workers. Under the megakernel's worker model the parallelism is
        # worth more than the MMA efficiency, so do not "fix" this back.
        def hand_written_linear(pk, ins, out, grid):
            pk.linear_layer(input=ins[0], weight=w, output=out,
                            grid_dim=(pk.num_workers, 1, 1),
                            block_dim=(128, 1, 1))

        return hand_written_linear

    return pick


# ---------------------------------------------------------------------------
# Plan, bind, build
# ---------------------------------------------------------------------------

def plan(shapes: Qwen3Shapes, *, num_layers=None):
    """The graph and its groups, before any weight is bound.

    build_qwen3 needs only shapes, so the boundaries are known BEFORE
    bind_weights runs. Callers that need both hand this back to build() so the
    partition is not run twice.
    """
    graph = build_qwen3(shapes, num_layers=num_layers)
    return graph, partition_as_today(graph)


def bind_weights(pk, model, shapes: Qwen3Shapes, *, num_layers=None,
                 cos=None, sin=None, tokens=None, lm_head_weight=None,
                 hand_written=()) -> tuple[dict, dict]:
    """HF weights -> the graph's named inputs. Returns (bindings, raw_weights).

    Every projection is transposed to (K, N): the graph does `x @ W`, and a
    searched matmul reads its weight that way, which is also what
    generated_linear_layer means by weight_t. Transposing here rather than in
    the graph keeps the graph about shapes, not storage.

    q/k/v are interleaved BY KV GROUP into one weight, not concatenated:
    paged_attention_layer reads its input as [Q_g0, K_g0, V_g0, Q_g1, ...],
    which is what mpk.shuffle_tensors builds in the imperative path. A plain
    concat compiles and runs and produces fluent-looking nonsense.
    """
    from ..utils import shuffle_tensors

    n = shapes.num_layers if num_layers is None else num_layers
    at = lambda t, nm: pk.attach_input(torch_tensor=t, name=nm)

    # HF stores a norm weight 1-D, (hidden,), and rmsnorm_layer reads it that
    # way. The GRAPH declares it (1, hidden) -- a searched task multiplies a
    # (rows, hidden) tile by it -- and a 1-D DTensor makes the replayed mul
    # fail with "threadblock elementbinary: unsupported operands". Reshape only
    # when the norm is actually in the graph; the opaque path is untouched, and
    # q_norm/k_norm stay 1-D either way because they belong to the opaque
    # attention task.
    at_norm = ((lambda t, nm: at(t.reshape(1, -1).contiguous(), nm))
               if os.environ.get("MPK_GRAPH_RMSNORM") == "1" else at)

    b = {"embed": at(model.model.embed_tokens.weight, "embed")}
    if tokens is not None:
        b["input_tokens"] = at(tokens, "input_token")
    if cos is not None:
        b["cos"], b["sin"] = at(cos, "cos"), at(sin, "sin")

    shuffled_qkv = {}
    for i in range(n):
        L = model.model.layers[i]
        qkv = shuffle_tensors([L.self_attn.q_proj.weight,
                               L.self_attn.k_proj.weight,
                               L.self_attn.v_proj.weight],
                              split=shapes.num_kv_heads, dim=0)
        shuffled_qkv[i] = qkv
        b[f"l{i}.in_norm"] = at_norm(L.input_layernorm.weight, f"l{i}_in_norm")
        b[f"l{i}.qkv"] = at(qkv.t().contiguous(), f"l{i}_qkv_t")
        b[f"l{i}.q_norm"] = at(L.self_attn.q_norm.weight, f"l{i}_q_norm")
        b[f"l{i}.k_norm"] = at(L.self_attn.k_norm.weight, f"l{i}_k_norm")
        b[f"l{i}.k_cache"] = at(model.model.kv_cache[0][i], f"l{i}_k_cache")
        b[f"l{i}.v_cache"] = at(model.model.kv_cache[1][i], f"l{i}_v_cache")
        b[f"l{i}.o"] = at(L.self_attn.o_proj.weight.t().contiguous(),
                          f"l{i}_o_t")
        b[f"l{i}.post_norm"] = at_norm(L.post_attention_layernorm.weight,
                                       f"l{i}_post_norm")
        b[f"l{i}.gate"] = at(L.mlp.gate_proj.weight.t().contiguous(),
                             f"l{i}_gate_t")
        b[f"l{i}.up"] = at(L.mlp.up_proj.weight.t().contiguous(), f"l{i}_up_t")
        b[f"l{i}.down"] = at(L.mlp.down_proj.weight.t().contiguous(),
                             f"l{i}_down_t")

    b["final_norm"] = at_norm(model.model.norm.weight, "final_norm")

    # The hand-written linear tasks read a weight as (N, K) -- the raw HF
    # layout -- while a searched matmul reads (K, N). A weight an override may
    # claim therefore needs the other orientation. Build only the one actually
    # used: at (153600, 1024) each lm_head copy is 314 MB.
    lm = model.lm_head.weight if lm_head_weight is None else lm_head_weight
    raw = {}
    if "lm_head" in hand_written:
        raw["lm_head"] = b["lm_head"] = at(lm, "lm_head_raw")
    else:
        b["lm_head"] = at(lm.t().contiguous(), "lm_head_t")

    # Any projection may be asked for raw. Which ones are worth it is
    # measurement: a generated matmul task costs ~2x the hand-written one for
    # the same arithmetic, so naming a weight in MPK_HANDWRITTEN substitutes
    # the hand-written task for that group. Each raw copy is a full weight, so
    # only the requested ones are built.
    for i in range(n):
        L = model.model.layers[i]
        for short, t in ((f"l{i}.o", L.self_attn.o_proj.weight),
                         (f"l{i}.down", L.mlp.down_proj.weight),
                         (f"l{i}.qkv", shuffled_qkv[i]),
                         (f"l{i}.gate", L.mlp.gate_proj.weight),
                         (f"l{i}.up", L.mlp.up_proj.weight)):
            if short in hand_written:
                raw[short] = at(t, short.replace(".", "_") + "_raw")
    return b, raw


def build(pk, shapes: Qwen3Shapes, bindings: dict, meta: dict, *,
          num_layers=None, raw_weights=None, hand_written=(), planned=None,
          verbose: bool = False):
    """Lower Qwen3 onto `pk`.

    bindings   graph input name -> attached DTensor (weights, caches, tokens)
    meta       the extra tensors the opaque tasks need but the graph does not
               name: argmax_value / argmax_index / output_token
    planned    a (graph, groups) plan() already produced, so the partition is
               not run a second time
    """
    graph, groups = planned if planned is not None else plan(
        shapes, num_layers=num_layers)
    forloop_for, stages_for = knobs_from_env(graph)
    return graph, groups, lower(
        pk, graph, groups, bindings,
        grid_for=grid_from_env(graph),
        forloop_for=forloop_for, stages_for=stages_for,
        outputs={graph.outputs[0].name: meta["output_token"]},
        opaque=opaque_handlers(pk, shapes, meta),
        fallbacks=pick_fallback,
        overrides=make_override(raw_weights, hand_written),
        reuse_buffers=True, verbose=verbose)
