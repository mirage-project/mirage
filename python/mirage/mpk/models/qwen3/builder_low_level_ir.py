"""Qwen3 built from the low-level IR instead of from a call order.

builder.py decides where every task begins and ends by the order it calls
mpk.*_layer. This assembles the same model out of the IR's node vocabulary --
lowering.OPS, which the superoptimizer schedules, and lowering.OPAQUE_OPS,
which hand-written MPK tasks compute -- and hands the result to lower(). The
task boundaries then come from dataflow rather than from this file's call
order, and everything reusable lives in lowering/: where the boundaries go
(default_partition), what computes an opaque node (standard_handlers), and the
schedule knobs (knobs_from_env). What is left here is Qwen3.
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
    # argmax reduces through one slot per WORKER -- not per SM. On a B200
    # those differ (128 vs 148), and a wrong extent here is not a crash:
    # the reduction reads the wrong slots and the model decodes garbage.
    # The caller must set it from pk.num_workers.
    num_workers: int = 0
    # tcgen05 needs the MMA's N a multiple of 8, and swapAB puts the decode
    # token count there -- prep zeroes the pad rows so they are benign.
    q_pad: int = 8

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
    """Qwen3's norm: rms_norm(x) * weight, two ops the partition pairs back
    into one task. It has a muGraph op, so it is never opaque."""
    return g.mul(g.rms_norm(x), w)


def mlp(g: ModelGraph, x: Value, wg: Value, wu: Value, wd: Value) -> Value:
    """silu(x @ Wg) * (x @ Wu) @ Wd."""
    gate = g.matmul(x, wg)
    up = g.matmul(x, wu)
    return g.matmul(g.mul(g.silu(gate), up), wd)


def build_layer(g: ModelGraph, x: Value, w: dict, s: Qwen3Shapes,
                layer: int) -> Value:
    """One decoder layer. `w` holds this layer's weights by short name."""
    with g.scope(layer=layer, tag=f"l{layer}.attn"):
        h = rmsnorm(g, x, w["in_norm"])
        qkv = g.matmul(h, w["qkv"])
        if os.environ.get("MPK_GRAPH_ATTENTION") == "1":
            # prep is opaque because it APPENDS to the KV cache -- a side
            # effect no dataflow graph can express -- and stages q/k^T/v/mask
            # in the fold-dim (kv_head-major) layout the core reads. The core
            # itself is a real subgraph, so search schedules it.
            fold = s.num_kv_heads * s.max_reqs
            q_st, mask_st, kt_st, v_st = g.opaque_multi(
                "attn_prep",
                [qkv, w["q_norm"], w["k_norm"], w["cos"], w["sin"],
                 w["k_cache"], w["v_cache"]],
                [(fold, s.q_pad, s.head_dim), (fold, 1, s.seq_len),
                 (fold, s.head_dim, s.seq_len), (fold, s.seq_len, s.head_dim)],
                layer=layer, num_kv_heads=s.num_kv_heads)
            e = g.exp(g.add(g.matmul(q_st, kt_st), mask_st))
            pad = g.matmul(g.div(e, g.reduction(e, 2)), v_st)
            attn = g.opaque("attn_finalize", [pad], (s.tokens, s.attn_dim),
                            layer=layer)
        else:
            attn = g.opaque("attention",
                            [qkv, w["q_norm"], w["k_norm"], w["cos"], w["sin"],
                             w["k_cache"], w["v_cache"]],
                            (s.tokens, s.attn_dim), layer=layer,
                            num_kv_heads=s.num_kv_heads)
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

    with g.scope(tag="head"):
        x = rmsnorm(g, x, g.new_input((1, s.hidden), "final_norm"))
        logits = g.matmul(x, g.new_input((s.hidden, s.out_vocab), "lm_head"))
        out, _, _ = g.opaque_multi(
            "argmax", [logits],
            [(s.tokens, 1), (s.tokens, s.num_workers),
             (s.tokens, s.num_workers)],
            dtypes=[None, None, "int64"])
    g.mark_output(out)
    return g



# The one run of nodes only Qwen3 can name: everything between the two opaque
# halves of attention is a single task. default_partition supplies the rest,
# and the cuts it derives are the ones builder.py's *_layer call order implies
# -- which is what makes the graph path comparable to it.
OPAQUE_RUNS = (("attn_prep", "attn_finalize", "attn_core"),)


def plan(shapes: Qwen3Shapes, *, num_layers=None):
    """The graph and its groups, before any weight is bound."""
    graph = build_qwen3(shapes, num_layers=num_layers)
    return graph, default_partition(graph, opaque_runs=OPAQUE_RUNS)


def bind_weights(pk, model, shapes: Qwen3Shapes, *, num_layers=None,
                 cos=None, sin=None, tokens=None, lm_head_weight=None) -> dict:
    """HF weights -> the graph's named inputs."""
    from ..utils import shuffle_tensors

    n = shapes.num_layers if num_layers is None else num_layers
    at = lambda t, nm: pk.attach_input(torch_tensor=t, name=nm)

    # HF stores a norm weight 1-D; the graph declares it (1, hidden) because a
    # searched task multiplies a (rows, hidden) tile by it.
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
    graph, groups = planned if planned is not None else plan(shapes, num_layers=num_layers)
    return graph, groups, lower(
        pk, graph, groups, bindings,
        outputs={graph.outputs[0].name: meta["output_token"]},
        verbose=verbose)
