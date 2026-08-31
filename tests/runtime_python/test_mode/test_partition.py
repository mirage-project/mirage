"""Choosing task boundaries instead of writing them.

The five MPK_COMPILED_MLP_IMPL variants are the partition space today, one
hand-written each. partition.py enumerates that space and filters it down to
what MPK can actually lower.

The filters here are all pure graph shape or memoised search probes; the
measurement that picks a winner is a separate step (rank_partitions.py).
"""
import pytest

from mirage.mpk.lowering import make_group
from mirage.mpk.models.qwen3.builder_low_level_ir import (
    Qwen3Shapes, build_qwen3, partition_as_today)
from mirage.mpk.lowering.partition import (check_fork_join, check_shapes,
                                   enumerate_partitions, feasible_partitions,
                                   group_signature, MAX_GROUP_INPUTS)

QWEN3 = Qwen3Shapes(tokens=8, hidden=1024, intermediate=3072, num_layers=28,
                    num_q_heads=16, num_kv_heads=8, head_dim=128, vocab=151936)


def _one_layer():
    g = build_qwen3(QWEN3, num_layers=1)
    return g, [i for i, n in enumerate(g.nodes) if n.layer == 0]


def test_todays_partition_is_legal():
    """The baseline has to survive its own filter, or the filter is wrong.

    It very nearly did not: MPK's check rejects a task that is both a
    join-consumer and a fork-consumer, and a transformer layer's residual add
    looks exactly like that until you account for residual stripping -- a
    direct edge u->v is dropped when a longer path u->...->v exists. Without
    that step every partition, including this one, is rejected.
    """
    g = build_qwen3(QWEN3, num_layers=2)
    assert check_fork_join(g, partition_as_today(g)) is None


def test_opaque_nodes_force_cuts():
    """An opaque task cannot be fused, so it is always alone in its group."""
    g, layer = _one_layer()
    for partition in enumerate_partitions(g, layer):
        for grp in partition:
            ops = [g.nodes[i].op for i in grp.nodes]
            if any(o.startswith("opaque:") for o in ops):
                assert len(ops) == 1, ops


def test_shape_filter_matches_what_search_accepts():
    g, _ = _one_layer()
    # gate+up+silu+mul reads x, Wg, Wu -> 3 inputs, at the cap
    grp = make_group(g, [7, 8, 9, 10], "up_silu")
    assert len(grp.external_inputs) == MAX_GROUP_INPUTS
    assert check_shapes(grp) is None
    # adding the down projection needs a fourth weight
    wide = make_group(g, [7, 8, 9, 10, 11], "up_silu_down")
    assert "inputs" in (check_shapes(wide) or "")


def test_signature_ignores_which_layer():
    """Qwen3 has 28 identical layers, so the same group in layer 0 and layer 1
    is one question for search. Without this the probe cost is 28x."""
    g = build_qwen3(QWEN3, num_layers=2)
    layers = [[i for i, n in enumerate(g.nodes) if n.layer == L] for L in (0, 1)]
    a = make_group(g, layers[0][7:11], "up_silu")
    b = make_group(g, layers[1][7:11], "up_silu")
    assert a.nodes != b.nodes
    assert group_signature(g, a) == group_signature(g, b)


def test_cheap_filters_keep_the_baseline_reachable():
    """Whatever else survives, the enumeration must contain today's MLP
    boundaries -- otherwise the search space excludes the known-good answer."""
    g, layer = _one_layer()
    kept, stats = feasible_partitions(g, layer, schedulable=None)
    assert stats["enumerated"] > 0
    assert kept, stats
    shapes = {tuple(x.tag for x in p) for p in kept}
    assert any("silu_mul" in tags for tags in shapes), sorted(shapes)[:3]


def test_the_shipped_partition_is_enumerated():
    """The enumeration must contain the partition MPK actually runs, EXACTLY
    -- not merely something with similar tags.

    It is the partition this project has whole-model numbers for, so if the
    enumeration cannot propose it there is nothing to rank against. Comparing
    node sets rather than tags matters: two partitions can share every tag and
    still cut the graph differently.
    """
    g, layer = _one_layer()
    in_layer = set(layer)
    kept, _ = feasible_partitions(g, layer, schedulable=None)
    found = {tuple(tuple(sorted(x.nodes)) for x in p) for p in kept}

    want = tuple(tuple(sorted(x.nodes)) for x in partition_as_today(g)
                 if set(x.nodes) <= in_layer)
    assert want in found, (
        f"as_today is not in the {len(found)} enumerated partitions: {want}")


@pytest.mark.slow
def test_search_filter_finds_the_known_good_fusion():
    """With the search probe on, the survivors should include the gate+up+SwiGLU
    fusion -- the variant measured 12.3% faster than three-task on Qwen3-0.6B
    decode. Slow: it runs search in subprocesses."""
    from mirage.mpk.lowering.partition import Schedulable
    g, layer = _one_layer()
    kept, stats = feasible_partitions(g, layer, schedulable=Schedulable(g))
    tags = {tuple(x.tag for x in p) for p in kept}
    assert any("matmul_matmul_silu_mul" in t for t in tags), sorted(tags)
    assert stats["kept"] >= 1, stats
