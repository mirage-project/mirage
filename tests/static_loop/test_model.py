"""cost_records' new fields, model.py and explore.py on the small fake graph (python tests/static_loop/test_model.py; needs
OR-Tools for predict)."""
import json
import os
import tempfile

import fakes

fakes.import_mpk()
from mirage.mpk import compiler, cost_records, search  # noqa: E402
from mirage.mpk.loop import explore, model  # noqa: E402

COSTS = os.path.join(fakes.ROOT, "costs", "b300_v22.json")
BAR = 10_000
# per node: a task's duration (ns); the GEMM's divided by its K parts
DUR = {"gemm_tile": 2000, "reader": 1000, "pool": 3000, "tail": 500}


def close(a, b, tol=1e-6):
    assert abs(a - b) <= tol, (a, b)


def simulate(schedules):
    """Stamps of one launch: SM k starts 100 k ns after the barrier and runs its list; a task starts when the SM reaches it (its
    previous task's end) and
    begins its work when its inputs are done; end = begin + DUR. Returns (times[g][sm][i] = (start, end, 0, 0), timing[g])."""
    times, timing = [], []
    for s in schedules:
        nodes = s["nodes"]
        dur = {}
        for t, task in enumerate(s["all_tasks"]):
            n = nodes[str(task["node"])]
            dur[t] = DUR[n["name"]] // (n["grid"][1] if n["name"] == "gemm_tile" else 1)
        end, start = {}, {}
        free = [BAR + 100 * sm for sm in range(len(s["worker_task_queues"]))]
        idx = [0] * len(free)
        while any(idx[sm] < len(q) for sm, q in enumerate(s["worker_task_queues"])):
            moved = False
            for sm, q in enumerate(s["worker_task_queues"]):
                if idx[sm] < len(q):
                    t = q[idx[sm]]
                    if all(d in end for d in s["all_tasks"][t]["deps"]):
                        start[t] = free[sm]
                        end[t] = max([start[t]] + [end[d] for d in s["all_tasks"][t]["deps"]]) + dur[t]
                        free[sm] = end[t]
                        idx[sm] += 1
                        moved = True
            assert moved, "the lists deadlock"
        width = max(len(q) for q in s["worker_task_queues"]) + 1
        times.append([[(start[t], end[t], 0, 0) for t in q] + [(0, 0, 0, 0)] * (width - len(q)) for q in s["worker_task_queues"]])
        timing.append((BAR, max(end.values())))
    return times, timing


def compiled(seed=3):
    pk = fakes.small_graph()
    plan = explore.random_plan(pk, 2, 8, seed)
    d = tempfile.mkdtemp()
    paths, _ = compiler.compile_plan(pk, plan, d)
    schedules = [json.load(open(p)) for p in paths]
    return pk, plan, paths, schedules


def records_of(pk, schedules, launches):
    nodes = {n.graph_idx: n for n in compiler.graph_nodes(pk)}
    key_of = lambda n: search.cost_key(n, n.grid, search.producer_grids(n, list(nodes.values())))
    return cost_records.records_from_run(schedules, launches, nodes, key_of, 2, {1}, "t", {"gpu": "fake"})


def test_records():
    pk, plan, paths, schedules = compiled()
    launch = simulate(schedules)
    recs = records_of(pk, schedules, [launch])
    gemm_parts = plan["grids"][0][1]
    for r in recs:
        assert "p75_us" in r, r
        if r["task_type"] == "pool":
            continue
        # every value of a record is its node's DUR (rule D1 removes the waits), so median = p75 = p90 = DUR
        want = DUR[r["task_type"]] / (gemm_parts if r["task_type"] == "gemm_tile" else 1) / 1e3
        close(r["median_us"], want), close(r["p75_us"], want), close(r["p90_us"], want)
        assert set(r["ctx_stats"]) == {"cold_frac", "in_flight_median"}
        assert 0 <= r["ctx_stats"]["cold_frac"] <= 1
        if r["context"]["first_on_sm"]:
            assert r["ctx_stats"]["cold_frac"] == 1.0, r      # first on its SM: always the first of its node there
    # in flight: 3 tasks with [start, end) = [0, 10), [5, 20), [10, 15): at 0: 0 others; at 5: 1 (the first); at 10: 1 (the
    # second; the first ended at 10)
    st = {0: {"sm": 0, "index": 0, "start": 0, "end": 10}, 1: {"sm": 1, "index": 0, "start": 5, "end": 20},
          2: {"sm": 0, "index": 1, "start": 10, "end": 15}}
    cold, fl = cost_records.task_context(st, [{"node": 7}, {"node": 7}, {"node": 7}])
    assert cold == {0: True, 1: True, 2: False} and fl == {0: 0, 1: 1, 2: 1}, (cold, fl)
    cost_records.KEEP_VALUES = True
    try:
        r = records_of(pk, schedules, [launch])[0]
        assert len(r["values"]) == r["n"] and len(r["values"][0]) == 3
    finally:
        cost_records.KEEP_VALUES = False


def test_model_and_predict():
    pk, plan, paths, schedules = compiled()
    launches = [simulate(schedules)] * 3
    d = tempfile.mkdtemp()
    f = os.path.join(d, "c.json")
    cost_records.append_records(f, records_of(pk, schedules, launches))
    m = model.Model(f)
    nodes = compiler.graph_nodes(pk)
    gemm = nodes[0]
    k = plan["grids"][0][1]
    close(m.task_us(gemm, gemm.grid, search.producer_grids(gemm, nodes)), DUR["gemm_tile"] / k / 1e3)
    # an unmeasured grid of the GEMM: the nearest measured one (the only one), listed as a fallback
    other = next(g for g in compiler.candidate_grids(gemm) if g != gemm.grid)
    assert m.task_us(gemm, other, search.producer_grids(gemm, nodes)) is not None and m.fallbacks[-1][1] == 0
    assert model.Model(f, fallback=False).task_us(gemm, other, search.producer_grids(gemm, nodes)) is None
    summary = model.predict(pk, plan, m, time_limit_s=10)
    assert summary["status"] in ("OPTIMAL", "FEASIBLE") and summary["fallbacks"] == [], summary["fallbacks"]
    pred = model.node_spans_predicted(summary)
    meas = model.node_spans_measured(schedules, [l[0] for l in launches], [l[1] for l in launches])
    gpu_sets = {g: tl["gpus"] for g, tl in summary["timeline"].items()}
    names = model.node_output_names(pk)
    phase_of = {"gemm_out": "front", "reader_out": "front", "pool_out": "pool", "tail_out": "tail"}
    pe = model.phase_error(pred, meas, phase_of, names, gpu_sets)
    for ph, per in pe["phases"].items():
        for v in per.values():
            close(v["err"], v["meas"] - v["pred"], 1e-3)
    assert set(pe["phase_err"]) == {"front", "pool", "tail"}, pe
    # the measured span vs the prediction: the simulation starts SM k 100 k ns late (SM 7: 0.7 us), the model at 0
    span = max(t[1] - t[0] for t in launches[0][1]) / 1e3
    assert abs(span - summary["predicted_us"]) < 0.25 * span, (span, summary["predicted_us"])
    print(f"  predicted {summary['predicted_us']} us, simulated {span} us, phase errors {pe['phase_err']}")


def test_tracker_and_fixed():
    t = model.ErrorTracker(window=3)
    for i, e in enumerate([0.4, 0.6, 3.0, 0.2]):
        t.add(f"c{i}", {"front": e, "tail": 5.0})
    assert [x["candidate"] for x in t.entries] == ["c1", "c2", "c3"]
    close(t.mean_abs()["front"], (0.6 + 3.0 + 0.2) / 3, 1e-3)
    assert t.trusted_phases(limit_us=1.2, phases=["front", "tail", "experts"]) == []
    assert t.trusted_phases(limit_us=1.3, phases=["experts"]) == ["front"]     # experts: no measurement, not trusted
    assert model.ErrorTracker.from_dict(json.loads(json.dumps(t.to_dict()))).entries == t.entries
    plan = {"grids": {0: (4, 2, 1)}, "lists": [[[(0, (0, 0, 0)), (1, (0, 0, 0))], [(1, (1, 0, 0))]]]}
    assert model.fixed_for_untrusted(plan, [1]) == {"grids": {}, "lists": [[[(1, (0, 0, 0))], [(1, (1, 0, 0))]]]}


def test_explore():
    pk, plan, paths, schedules = compiled()
    grids_before = [n.grid for n in compiler.graph_nodes(pk)]
    for kind in ("move_node", "option"):
        for seed in range(4):
            new = explore.perturb_plan(pk, plan, seed, kind)
            assert [n.grid for n in compiler.graph_nodes(pk)] == grids_before   # pk's grids put back
            changed = {e[0] for g, gl in enumerate(new["lists"]) for sm, l in enumerate(gl)
                       for e in set(l) ^ set(plan["lists"][g][sm])}
            if kind == "move_node":
                assert len(changed) == 1 and new["grids"] == plan["grids"], (new["why"], changed)
            else:
                assert changed == {0} and new["grids"][0] != plan["grids"][0], new["why"]
    launch = simulate(schedules)
    recs = records_of(pk, schedules, [launch])
    targets = explore.exploration_targets(recs, pk, k=4)
    assert [t["values"] for t in targets] == sorted(t["values"] for t in targets) and targets[0]["values"] == 0, targets
    print("  exploration targets:", [(t["name"], t["grid"], t["first_on_sm"], t["values"]) for t in targets])


def test_costs_unchanged():
    """search.Costs on costs/b300_v22.json: the values before this change (Costs itself is unchanged; a guard)."""
    if not os.path.exists(COSTS):
        print("  skipped: no", COSTS)
        return
    c = search.Costs(COSTS)
    want = [("allreduce_send", {"grid": [148, 1, 1], "params": [3584, 16], "shapes": [[128, 3584], [8, 3584]]}, 1.84, None, 1.84),
            ("gemm_tile", {"grid": [7, 14, 1], "params": [0, 7168, 896, 0], "shapes": [[8, 7168], [896, 7168], [8, 896]]}, None, 4.528, 4.528),
            ("sum_send", {"grid": [56, 14, 1], "inputs": {"0": [56, 7, 1]}, "params": [], "shapes": [[8, 7168], [8, 7168]]}, 1.184, None, 1.184),
            ("gemm_tile", {"grid": [12, 14, 1], "params": [1, 7168, 1536, 0], "shapes": [[8, 7168], [1536, 7168], [8, 1536]]}, 3.424, 4.544, 3.488),
            ("gemm_tile", {"grid": [28, 2, 1], "params": [0, 7168, 3584, 0], "shapes": [[8, 7168], [3584, 7168], [8, 3584]]}, 8.744, 10.304, 9.36)]
    for tt, key, behind, first, anyc in want:
        assert (c.median(tt, key, first=False), c.median(tt, key, first=True), c.median(tt, key)) == (behind, first, anyc), (tt, key)
    m = model.Model(COSTS, stat="median")
    for tt, key, behind, first, anyc in want:
        assert m.stat_of(tt, key, first=False) == behind and m.stat_of(tt, key) == anyc


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
