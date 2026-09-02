#!/usr/bin/env python3
"""Print the groups Qwen3 lowers to -- one group is one MPK task.

    python demo/qwen3_groups.py                 # the partition MPK runs
    python demo/qwen3_groups.py --layers 2      # two decoder layers
    python demo/qwen3_groups.py --enumerate     # every legal alternative

No GPU and no weights: this is graph construction and partitioning only, which
is the part of lowering that happens before anything is scheduled or compiled.
"""
import argparse

from mirage.mpk.lowering import default_partition, enumerate_partitions
from mirage.mpk.models.qwen3.builder_low_level_ir import (OPAQUE_RUNS,
                                                          Qwen3Shapes,
                                                          build_qwen3)

QWEN3_0_6B = dict(hidden=1024, intermediate=3072, num_layers=28, num_q_heads=16,
                  num_kv_heads=8, head_dim=128, vocab=151936)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--enumerate", action="store_true",
                    help="list every legal partition of one decoder layer")
    ap.add_argument("--nodes", action="store_true", help="print the graph too")
    args = ap.parse_args()

    shapes = Qwen3Shapes(tokens=args.tokens, seq_len=64, max_reqs=args.tokens,
                         num_workers=128, **QWEN3_0_6B)
    graph = build_qwen3(shapes, num_layers=args.layers)

    if args.nodes:
        print(graph.describe(), "\n")

    groups = default_partition(graph, opaque_runs=OPAQUE_RUNS)
    print(f"{len(graph)} nodes -> {len(groups)} groups "
          f"({args.layers} layer(s), {args.tokens} tokens)\n")
    for i, g in enumerate(groups):
        ins = " ".join("x".join(str(d) for d in v.dims)
                       for v in g.external_inputs)
        out = "x".join(str(d) for d in g.output.dims)
        extra = f" (+{len(g.extra_outputs)} more out)" if g.extra_outputs else ""
        print(f"{i:3d}  {g.tag:14} {str(list(g.nodes)):18} {ins:34} -> {out}{extra}")

    if not args.enumerate:
        return
    # Every legal alternative for ONE decoder layer. All 28 are identical, so
    # the layer is the only part worth enumerating; nothing here ranks them.
    layer = list(range(1, len(graph) - 3))
    cands = list(enumerate_partitions(graph, layer))
    print(f"\n{len(cands)} legal partitions of one decoder layer:")
    for i, p in enumerate(cands):
        print(f"{i:3d}  " + " | ".join(x.tag for x in p))


if __name__ == "__main__":
    main()
