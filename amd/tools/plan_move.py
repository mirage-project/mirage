"""Plan edits by hand for an A/B (the tasks and grids stay the same):

    python amd/tools/plan_move.py <plan.json> <out.json> A:B [A:B ...] [P:B:E1,E2,...]

A:B   on each CU (every GPU) where node A comes before node B, A's tasks go right after B's. e.g. 34:36 (shared down after
      moe_experts). The caller checks the new order is safe: nothing between A's old place and B may wait for A.
P:B:E (three fields) on each CU with node P: P's tasks go right after B's, and the CU's tasks of nodes E move to CUs without P
      (round robin, fewest tasks first), each right after that CU's last task of a node <= its own. e.g. 40:36:37,38,39:
      latent_up right after moe_experts, so its weights load while the CUs without it send R and normalise. Checked: no task
      comes before one it waits for on its CU (DEPS).
"""
import json
import sys


def move(plan: dict, a: int, b: int) -> int:
    moved = 0
    for gpu in plan["lists"]:
        for i, cu in enumerate(gpu):
            ids = [t[0] for t in cu]
            if a not in ids or b not in ids or ids.index(a) > ids.index(b):
                continue
            mine = [t for t in cu if t[0] == a]
            rest = [t for t in cu if t[0] != a]
            at = max(j for j, t in enumerate(rest) if t[0] == b) + 1
            gpu[i] = rest[:at] + mine + rest[at:]
            moved += 1
    return moved


def prefetch(plan: dict, p: int, b: int, evict: set) -> int:
    moved = 0
    for gpu in plan["lists"]:
        has_p = [any(t[0] == p for t in cu) for cu in gpu]
        out = []
        for i, cu in enumerate(gpu):
            if not has_p[i]:
                continue
            out += [t for t in cu if t[0] in evict]
            mine = [t for t in cu if t[0] == p]
            rest = [t for t in cu if t[0] != p and t[0] not in evict]
            at = max(j for j, t in enumerate(rest) if t[0] == b) + 1
            gpu[i] = rest[:at] + mine + rest[at:]
            moved += 1
        others = sorted((i for i in range(len(gpu)) if not has_p[i]), key=lambda i: (len(gpu[i]), i))
        for k, task in enumerate(out):
            cu = gpu[others[k % len(others)]]
            at = max((j + 1 for j, t in enumerate(cu) if t[0] <= task[0]), default=0)
            cu.insert(at, task)
        for cu in gpu:   # no task before one it waits for (the K3 layer's tail edges, transitively)
            ids = [t[0] for t in cu]
            for j, n in enumerate(ids):
                assert not (waits_for(n) & set(ids[j + 1:])), (n, ids)
    return moved


# the K3 MoE graph's edges among the nodes these edits move (demo/kimi_k3_moe/demo.py build_graph)
DEPS = {27: {26}, 29: {28}, 31: {30}, 34: {31}, 35: {34}, 36: {27, 29}, 37: {36}, 38: {37}, 39: {35}, 40: {38}, 41: {40},
        42: {39, 41}}


def waits_for(n: int) -> set:
    out, todo = set(), [n]
    while todo:
        for d in DEPS.get(todo.pop(), ()):
            if d not in out:
                out.add(d)
                todo.append(d)
    return out


def main():
    src, dst, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
    plan = json.load(open(src))
    for p in pairs:
        f = p.split(":")
        if len(f) == 3:
            n = prefetch(plan, int(f[0]), int(f[1]), {int(x) for x in f[2].split(",")})
            print(f"{f[0]} right after {f[1]}, nodes {f[2]} off its CUs: {n} CU lists")
            continue
        a, b = (int(x) for x in f)
        print(f"{a} after {b}: {move(plan, a, b)} CU lists changed")
    json.dump(plan, open(dst, "w"))


if __name__ == "__main__":
    main()
