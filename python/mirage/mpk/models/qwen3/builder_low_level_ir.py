"""Qwen3 built from the low-level IR instead of from a call order.

builder.py decides where every task begins and ends by the order it calls
mpk.*_layer. This assembles the same model from the IR's node vocabulary and
hands it to lower(), so the boundaries come from dataflow. What is left here
is Qwen3. See docs/superoptimizer_ir.md.
"""
from __future__ import annotations

import dataclasses
import os

from ...lowering import default_partition, lower
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
    max_seq: int = 4096          # RoPE table length
    seq_len: int = 512           # decode window; the staging buffers' S
    max_reqs: int = 8
    num_workers: int = 0     # MUST come from pk.num_workers, not the SM count
    q_pad: int = 8           # tcgen05 wants the MMA's N a multiple of 8
    split_attention: bool = False   # the searched core cannot lower today

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


def rmsnorm(g: ModelGraph, x: Value, w: Value) -> Value:
    """rms_norm(x) * weight -- two ops the partition pairs into one task."""
    return g.mul(g.rms_norm(x), w)


def mlp(g: ModelGraph, x: Value, wg: Value, wu: Value, wd: Value) -> Value:
    """silu(x @ Wg) * (x @ Wu) @ Wd."""
    gate = g.matmul(x, wg)
    up = g.matmul(x, wu)
    return g.matmul(g.mul(g.silu(gate), up), wd)


def _attn_inputs(qkv: Value, w: dict) -> list:
    """What either attention path reads, in task order."""
    return [qkv, w["q_norm"], w["k_norm"], w["cos"], w["sin"], w["k_cache"], w["v_cache"]]


def attention(g: ModelGraph, qkv: Value, w: dict, s: Qwen3Shapes,
              layer: int) -> Value:
    """One opaque task, or prep + a searched core + finalize.

    Prep is opaque because it appends to the KV cache. The split form does not
    lower today -- see docs/superoptimizer_ir.md.
    """
    if not s.split_attention:
        return g.opaque("attention", _attn_inputs(qkv, w),
                        (s.tokens, s.attn_dim),
                        layer=layer, num_kv_heads=s.num_kv_heads)

    fold = s.num_kv_heads * s.max_reqs
    q, mask, kt, v = g.opaque_multi(
        "attn_prep", _attn_inputs(qkv, w),
        [(fold, s.q_pad, s.head_dim),
         (fold, 1, s.seq_len),
         (fold, s.head_dim, s.seq_len),
         (fold, s.seq_len, s.head_dim)],
        inits=[None, -30000.0, None, None],
        layer=layer, num_kv_heads=s.num_kv_heads)

    e = g.exp(g.add(g.matmul(q, kt), mask))
    core = g.matmul(g.div(e, g.reduction(e, 2)), v)
    return g.opaque("attn_finalize", [core], (s.tokens, s.attn_dim),
                    layer=layer)


def build_layer(g: ModelGraph, x: Value, w: dict, s: Qwen3Shapes,
                layer: int) -> Value:
    """One decoder layer. `w` holds this layer's weights by short name."""
    h = rmsnorm(g, x, w["in_norm"])
    qkv = g.matmul(h, w["qkv"])
    x = g.add(x, g.matmul(attention(g, qkv, w, s, layer), w["o"]))

    h = rmsnorm(g, x, w["post_norm"])
    x = g.add(x, mlp(g, h, w["gate"], w["up"], w["down"]))

    return x


def build_qwen3(s: Qwen3Shapes, *, num_layers: int | None = None) -> ModelGraph:
    """The whole model. num_layers trims it for tests."""
    n = s.num_layers if num_layers is None else num_layers
    g = ModelGraph("qwen3")

    tokens = g.new_input((s.tokens, 1), "input_tokens", role="feed")
    embed_w = g.new_input((s.vocab, s.hidden), "embed")
    x = g.opaque("embedding", [tokens, embed_w], (s.tokens, s.hidden),
                 tokens=s.tokens)

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

    x = rmsnorm(g, x, g.new_input((1, s.hidden), "final_norm"))
    logits = g.matmul(x, g.new_input((s.hidden, s.out_vocab), "lm_head"))
    out, _, _ = g.opaque_multi(
        "argmax", [logits],
        [(s.tokens, 1), (s.tokens, s.num_workers), (s.tokens, s.num_workers)],
        dtypes=[None, None, "int64"])
    g.mark_output(out)
    return g



# Everything between the two opaque halves of attention is one task.
OPAQUE_RUNS = (("attn_prep", "attn_finalize", "attn_core"),)


def _split_core(graph, groups, at: int):
    """Cut each attn_core group after `at` nodes.

    default_partition makes the whole core one task, and that one cannot be
    registered: 6 ops over 4 inputs leave ops outside the fused threadblock
    graph. Smaller pieces do lower -- every split works except one that leaves
    a group starting with `reduction`.
    """
    from ...lowering.group import make_group

    out = []
    for grp in groups:
        ids = list(grp.nodes)
        if grp.tag != "attn_core" or not 0 < at < len(ids):
            out.append(grp)
            continue
        head, tail = ids[:at], ids[at:]
        if graph.nodes[tail[0]].op == "reduction":
            raise ValueError("a group starting with `reduction` cannot lower")
        out.append(make_group(graph, head, "attn_core_a"))
        out.append(make_group(graph, tail, "attn_core_b"))
    return out


def _apply_pattern(graph, pattern):
    """Re-cut every run of non-opaque nodes using `pattern` cyclically.

    A candidate is ranked on ONE layer; the layers are identical, so the same
    run lengths replayed across the graph give the whole model that partition.
    A run length that would build an illegal group (too many inputs, a bad
    shape) falls back to single nodes for that piece rather than failing.
    """
    from ...lowering.group import make_group
    from ...lowering.node import is_opaque as _op
    from ...lowering.partition import _tag

    groups, run = [], []

    def cut(ids):
        try:
            groups.append(make_group(graph, ids, _tag(graph, ids)))
        except ValueError:
            for i in ids:
                groups.append(make_group(graph, [i], graph.nodes[i].op))

    def flush():
        # Restart the pattern at every run. Cycling it continuously across the
        # graph lets the phase drift after the first run, so later layers get
        # cuts the ranking never validated -- and they fail to lower.
        k = 0
        while run:
            n = min(pattern[k % len(pattern)], len(run))
            cut(run[:n])
            del run[:n]
            k += 1

    for i, nd in enumerate(graph.nodes):
        if _op(nd.op):
            flush()
            groups.append(make_group(graph, [i], nd.op.split(":", 1)[1]))
        else:
            run.append(i)
    flush()
    return groups


def plan(shapes: Qwen3Shapes, *, num_layers=None, fuse: bool = True,
         core_split: int = 5, pattern=None):
    """The graph and its groups, before any weight is bound."""
    graph = build_qwen3(shapes, num_layers=num_layers)
    runs = OPAQUE_RUNS if shapes.split_attention else ()
    groups = default_partition(graph, opaque_runs=runs, fuse=fuse)
    if shapes.split_attention:
        groups = _split_core(graph, groups, core_split)
    if pattern:
        groups = _apply_pattern(graph, tuple(pattern))
    return graph, groups


def bind_weights(pk, model, shapes: Qwen3Shapes, *, num_layers=None,
                 cos=None, sin=None, tokens=None, lm_head_weight=None) -> dict:
    """HF weights -> the graph's named inputs."""
    from ..utils import shuffle_tensors

    n = shapes.num_layers if num_layers is None else num_layers
    at = lambda t, nm: pk.attach_input(torch_tensor=t, name=nm)

    # HF stores norm weights 1-D; the graph declares them (1, hidden).
    at_norm = lambda t, nm: at(t.reshape(1, -1).contiguous(), nm)

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

    lm = model.lm_head.weight if lm_head_weight is None else lm_head_weight
    b["lm_head"] = at(lm.t().contiguous(), "lm_head_t")
    return b


def build(pk, shapes: Qwen3Shapes, bindings: dict, meta: dict, *,
          num_layers=None, planned=None, verbose: bool = False):
    """Lower Qwen3 onto `pk`."""
    if planned is not None:
        graph, groups = planned
    else:
        # MPK_FUSE=0: one task per node, for partition comparisons.
        graph, groups = plan(shapes, num_layers=num_layers,
                             fuse=os.environ.get("MPK_FUSE", "1") != "0")
    return graph, groups, lower(
        pk, graph, groups, bindings,
        outputs={graph.outputs[0].name: meta["output_token"]},
        verbose=verbose)
