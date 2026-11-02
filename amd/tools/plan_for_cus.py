"""A plan file for more task lists (CUs) than it was made for: a B300 plan (148 SMs) -> MI355X (256 CUs), a first plan to compile
and check on the AMD box before the search has AMD cost records.

The nodes whose grid is the SM count (their grid is not in the plan: the layer sets it; one task on every list; e.g. moe_experts
and residual_add, one task per SM, and allreduce_send after moe_experts) get one task per CU: on list i >= the old count, task
i of each such node, in the order those nodes appear on list 0.
Lists 0 .. old count - 1 stay as they are, and the grids stay (the one-task-per-SM nodes' grids come from the layer: the GPU's
CU count). Every task of the old plan stays on the same list, so dependencies keep their order.

python amd/tools/plan_for_cus.py IN_PLAN OUT_PLAN [--cus 256]
python amd/tools/plan_for_cus.py --self-test
"""
import argparse
import json
import os


def per_sm_nodes(lists: list, grids: dict) -> list:
    """The nodes sized by the SM count: grid not in the plan, one task on every list, tasks 0 .. lists - 1 (in the order they
    appear on list 0)."""
    order = []
    for node, pos in lists[0]:
        if node in order or str(node) in grids:
            continue
        xs = sorted(p[0] for lst in lists for nd, p in lst if nd == node)
        if xs == list(range(len(lists))) and all(sum(1 for nd, _ in lst if nd == node) == 1 for lst in lists):
            order.append(node)
    return order


def widen(plan: dict, cus: int) -> dict:
    out = {"grids": plan["grids"], "lists": []}
    for gpu_lists in plan["lists"]:
        n = len(gpu_lists)
        if cus < n:
            raise ValueError(f"{n} lists, {cus} CUs: fewer lists is not this tool's job")
        pool = per_sm_nodes(gpu_lists, plan["grids"])
        template = {node: list(pos) for node, pos in gpu_lists[0] if node in pool}
        new = [list(lst) for lst in gpu_lists]
        for i in range(n, cus):
            new.append([[node, [i] + template[node][1:]] for node in pool])
        out["lists"].append(new)
    for k in plan:
        if k not in out:
            out[k] = plan[k]
    return out


def self_test():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "..", "..", "demo", "kimi_k3_moe", "solver_plan.json")
    plan = json.load(open(src))
    wide = widen(plan, 256)
    for g, (old, new) in enumerate(zip(plan["lists"], wide["lists"])):
        assert len(new) == 256 and new[:len(old)] == old
        pool = per_sm_nodes(old, plan["grids"])
        assert pool, "a B300 plan has one-task-per-SM nodes"
        for node in pool:
            xs = sorted(p[0] for lst in new for nd, p in lst if nd == node)
            assert xs == list(range(256)), (g, node)
            for i, lst in enumerate(new[len(old):], len(old)):
                assert sum(1 for nd, p in lst if nd == node and p[0] == i) == 1
        assert sum(len(l) for l in new) == sum(len(l) for l in old) + (256 - len(old)) * len(pool)
    print(f"solver_plan.json: {len(plan['lists'][0])} -> 256 lists per GPU; nodes sized by the SM count "
          f"{per_sm_nodes(plan['lists'][0], plan['grids'])} "
          f"on every CU; the old lists unchanged")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inp", nargs="?")
    p.add_argument("out", nargs="?")
    p.add_argument("--cus", type=int, default=256)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
    else:
        json.dump(widen(json.load(open(a.inp)), a.cus), open(a.out, "w"))
        print(f"{a.inp} -> {a.out}: {a.cus} lists per GPU")
