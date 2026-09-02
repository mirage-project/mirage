#!/usr/bin/env python3
"""Rank Qwen3's candidate partitions from one profiled build, benchmark top-k.

    # 1. compile + profile once, to price every shape
    CUDA_VISIBLE_DEVICES=3 MPK_MODEL_SOURCE=mugraph MPK_DUMP_SHAPE_INDEX=1 \
      python3 tests/ci-tests/run_batch_perf.py --model Qwen/Qwen3-0.6B \
      --max-num-batched-tokens 8 --max-num-batched-requests 8 \
      --prompt-length 1 --max-seq-length 64 --profiling 2> shapes.log
    # 2. rank every candidate against those measurements
    python3 demo/qwen3_rank.py --profile mirage_0.csv --shapes shapes.log

Prediction is a RANKING device: ~5% low in absolute terms. The point is to
spend real builds only on the few candidates worth measuring.
"""
import argparse

from mirage.mpk.lowering import CostTable, rank, report
from mirage.mpk.lowering.node import is_opaque
from mirage.mpk.lowering.partition import (Schedulable, check_fork_join,
                                           check_shapes,
                                           enumerate_partitions)
from mirage.mpk.lowering.task_search import knobs_from_env
from mirage.mpk.models.qwen3.builder_low_level_ir import Qwen3Shapes, plan

QWEN3_0_6B = dict(hidden=1024, intermediate=3072, num_layers=28, num_q_heads=16,
                  num_kv_heads=8, head_dim=128, vocab=151936)

# What computes each opaque node, and how many tasks it runs. Both are the
# model's business: the handler picks the task and the grid.
OPAQUE_TASK = {"attention": "TASK_ATTN_SM100",
               "attn_prep": "TASK_ATTN_PREP_SM100",
               "attn_finalize": "TASK_ATTN_FINALIZE_SM100",
               "embedding": "TASK_EMBEDDING",
               "argmax": "TASK_ARGMAX_PARTIAL_SM100"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", required=True, help="mirage_0.csv")
    ap.add_argument("--shapes", required=True,
                    help="stderr of a MPK_DUMP_SHAPE_INDEX=1 run")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--filter", action="store_true",
                    help="probe every distinct group and drop candidates "
                         "containing one that cannot be scheduled")
    ap.add_argument("--probe", type=int, default=0, metavar="N",
                    help="ask search whether the top N actually lower; "
                         "ranking alone proposes groups that do not")
    ap.add_argument("--region", type=int, default=1,
                    help="how many decoder layers to enumerate over")
    args = ap.parse_args()

    cost = CostTable.from_profile(args.profile, args.shapes)
    print(f"priced {len(cost.by_shape)} generated shapes and "
          f"{len(cost.by_task)} hand-written tasks; "
          f"floor {cost.floor * 1e9:.0f} ns\n")

    shapes = Qwen3Shapes(tokens=args.tokens, seq_len=64, max_reqs=args.tokens,
                         num_workers=args.workers, **QWEN3_0_6B)
    graph, default = plan(shapes, num_layers=args.region)
    grid_for, forloop_for, _ = knobs_from_env(graph)
    opaque_tasks = {"attention": shapes.max_reqs * shapes.num_kv_heads,
                    "embedding": shapes.tokens, "argmax": args.workers}

    # Enumerate over the REPEATING unit -- the longest run of searchable
    # nodes between two opaque ones. That run is what every layer contains, so
    # a partition of it replays across the whole model unchanged.
    runs, cur = [], []
    for i in range(len(graph)):
        if is_opaque(graph.nodes[i].op):
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(i)
    if cur:
        runs.append(cur)
    ids = max(runs, key=len)
    print(f"repeating unit: {len(ids)} nodes "
          f"({' '.join(graph.nodes[i].op for i in ids)})")
    cands = []
    for part in enumerate_partitions(graph, ids):
        if check_fork_join(graph, part):
            continue
        if any(check_shapes(g) for g in part):
            continue
        cands.append(part)
    print(f"{len(graph)} nodes, {len(cands)} legal candidate partitions\n")

    if args.filter:
        # Ranking alone favours fusions search cannot schedule -- the whole
        # top of the list is unbuildable. Probe each DISTINCT group once
        # (there are far fewer groups than partitions) and keep only the
        # candidates whose every group lowers.
        from mirage.mpk.lowering.partition import group_signature
        sched = Schedulable(graph, grid_for=grid_for,
                            forloop_for=forloop_for)
        verdict = {}
        for part in cands:
            for g in part:
                if is_opaque(graph.nodes[g.nodes[0]].op):
                    continue
                sig = group_signature(graph, g)
                if sig not in verdict:
                    verdict[sig] = sched(g)
        ok = [p for p in cands
              if all(is_opaque(graph.nodes[g.nodes[0]].op)
                     or not verdict[group_signature(graph, g)] for g in p)]
        bad = sum(1 for v in verdict.values() if v)
        print(f"probed {len(verdict)} distinct groups: {bad} do not lower; "
              f"{len(ok)} of {len(cands)} candidates survive\n")
        cands = ok

    ranked = rank(graph, cands, cost, num_workers=args.workers,
                  grid_for=grid_for, opaque_task=OPAQUE_TASK,
                  opaque_tasks=opaque_tasks)
    print(report(ranked, args.top))
    def pattern_of(c):
        return ",".join(str(len(g.nodes)) for g in c.partition
                        if not is_opaque(graph.nodes[g.nodes[0]].op))

    if not args.probe:
        print("\npatterns to build (MPK_PARTITION=...):")
        for i, c in enumerate(ranked[:args.top]):
            print(f"  {i}: {c.makespan * 1e6:7.1f} us  "
                  f"MPK_PARTITION={pattern_of(c)}")
    else:
        # A predicted makespan says nothing about whether search can schedule
        # the groups. Probe in rank order and keep the ones that lower.
        print(f"\nprobing the top {args.probe} for schedulability:")
        sched = Schedulable(graph, grid_for=grid_for,
                            forloop_for=forloop_for)
        kept = 0
        for i, c in enumerate(ranked[:args.probe]):
            bad = next((sched(g) for g in c.partition
                        if not is_opaque(graph.nodes[g.nodes[0]].op)
                        and sched(g)), None)
            mark = "BUILDABLE" if not bad else f"no: {str(bad)[:60]}"
            print(f"  {i:3d} {c.makespan * 1e6:7.1f} us  "
                  f"MPK_PARTITION={pattern_of(c):28} {mark}")
            if not bad:
                kept += 1
                if kept >= args.top:
                    break
        print(f"\n{kept} of the top {args.probe} lower.")
    print(f"\nbest:    {ranked[0].tags}")
    print(f"default: {' | '.join(g.tag or str(g.nodes) for g in default)}")
    print(f"\nBuild and measure the top {args.top}; the prediction ranks, "
          f"the benchmark decides.")


if __name__ == "__main__":
    main()
