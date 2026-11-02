"""The task times a timing-build launch dumped (MPK_DUMP_TIMES=<file>.json: every task of every GPU, ns after its GPU's
start barrier) read per node over the GPUs: when the node's first task started and its last ended on each GPU, and how long
the CUs that ran it sat idle before it (the CU's previous task end -> this task's start).

    python amd/tools/task_times.py dump.json [--node N ...] [--cus]

Per node (in the order of the latest end over the GPUs): tasks per GPU, first start / last end on the fastest and the
slowest GPU, the median task, the median "landed" time (stage stamp 0, when it is inside the task: a GEMM's activation in
LDS, allreduce_send's producer done), and the GPU whose last task ended last. --cus: per GPU the CUs whose last task ended
last (their task list with gaps)."""
import argparse
import json
from collections import defaultdict
from statistics import median


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump")
    p.add_argument("--node", type=int, action="append", default=[], help="only these nodes")
    p.add_argument("--cus", action="store_true", help="per GPU the 4 CUs that ended last, their tasks")
    a = p.parse_args()
    d = json.load(open(a.dump))
    cols = d["cols"]
    ix = {c: i for i, c in enumerate(cols)}
    rows = [r for r in d["rows"] if r[ix["start"]] and r[ix["end"]]]
    us = lambda v: v / 1e3   # noqa: E731
    per = defaultdict(lambda: defaultdict(list))   # node -> gpu -> [(start, end, st0, cu, pos)]
    names = {}
    by_cu = defaultdict(list)
    for r in rows:
        g, cu, node, name, pos, s, e, st0 = (r[ix[k]] for k in ("gpu", "cu", "node", "name", "pos", "start", "end", "st0"))
        per[node][g].append((s, e, st0, cu, pos))
        names[node] = name
        by_cu[(g, cu)].append((s, e, node))   # (in start order: a CU runs its list in order)
    idle = defaultdict(list)   # node -> the gap before each of its tasks on its CU (us)
    for (g, cu), lst in by_cu.items():
        lst.sort()
        prev = 0
        for s, e, node in lst:
            idle[node].append(us(s - prev))
            prev = e
    gpus = sorted({r[ix["gpu"]] for r in rows})
    order = sorted(per, key=lambda n: max(max(e for _, e, *_ in per[n][g]) for g in per[n]))
    print(f"{len(rows)} tasks, {len(gpus)} GPUs; us after each GPU's start barrier")
    print(f"{'node':>14} {'tasks':>5} {'first start':>17} {'last end':>17} {'median':>6} {'landed':>13} {'idle before':>11}  late GPU")
    for n in order:
        if a.node and n not in a.node:
            continue
        fs = {g: us(min(s for s, *_ in per[n][g])) for g in per[n]}
        le = {g: us(max(e for _, e, *_ in per[n][g])) for g in per[n]}
        med = median(us(e - s) for g in per[n] for s, e, *_ in per[n][g])
        landed = [us(st0) for g in per[n] for s, e, st0, *_ in per[n][g] if st0 is not None and s <= st0 <= e]
        lt = f"{median(landed):5.1f}..{max(landed):5.1f}" if landed else "-"
        late = max(le, key=le.get)
        ntask = len(per[n][gpus[0]]) if gpus[0] in per[n] else 0
        print(f"{names[n][:10]:>10}#{n:<3} {ntask:5d} {min(fs.values()):7.1f}..{max(fs.values()):7.1f}"
              f" {min(le.values()):7.1f}..{max(le.values()):7.1f} {med:6.1f} {lt:>13} {median(idle[n]):11.1f}  GPU {late}")
    if a.cus:
        for g in gpus:
            ends = sorted(((max(e for _, e, _ in lst), cu) for (gg, cu), lst in by_cu.items() if gg == g), reverse=True)
            for end, cu in ends[:4]:
                lst = sorted(by_cu[(g, cu)])
                prev, items = 0, []
                for s, e, node in lst:
                    items.append(f"+{us(s - prev):.1f} {names[node][:8]}#{node} {us(s):.1f}-{us(e):.1f}")
                    prev = e
                print(f"GPU {g} CU {cu} ends {us(end):.1f}: " + " | ".join(items))


if __name__ == "__main__":
    main()
