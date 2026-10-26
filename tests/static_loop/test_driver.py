"""driver.py (init, round, ingest, status), passes.py and round_job's triage on the small fake graph with simulated stamps
(python tests/static_loop/test_driver.py; needs OR-Tools)."""
import json
import os
import shutil
import tempfile

import numpy as np

import fakes

fakes.import_mpk()
from mirage.mpk import compiler  # noqa: E402
from mirage.mpk.loop import candidate as cand_mod  # noqa: E402
from mirage.mpk.loop import driver, model, passes, triage  # noqa: E402
from test_model import compiled, records_of, simulate  # noqa: E402

PART_YAML = """name: fake_part
module: hooks.py
tp: 2
buckets: {4: {inputs: {a: /nowhere}}}
phases: {front: [gemm_out, reader_out], experts: [pool_out], tail: [tail_out]}
error_limits: {y: 1.0e-2, derivation: test}
"""


def make_run(root, name, pk, plan, cand_id, costs_path):
    """A run.py --record --predict directory from simulated stamps."""
    rd = os.path.join(root, "runs", name)
    paths, _ = compiler.compile_plan(pk, plan, os.path.join(rd, "schedules"))
    schedules = [json.load(open(p)) for p in paths]
    launches = [simulate(schedules) for _ in range(3)]
    with open(costs_path, "w") as f:
        json.dump({"records": records_of(pk, schedules, launches)}, f)
    np.save(os.path.join(rd, "task_times.npy"), np.array([t for t, _ in launches], dtype=np.int64))
    np.save(os.path.join(rd, "timing.npy"), np.array([t for _, t in launches], dtype=np.int64))
    m = model.Model(costs_path)
    summary = model.predict(pk, plan, m, time_limit_s=10)
    with open(os.path.join(rd, "prediction.json"), "w") as f:
        json.dump({"predicted_us": summary["predicted_us"], "timeline": summary["timeline"], "fallbacks": summary["fallbacks"]}, f)
    spans = [(max(e for _, e in tm) - min(b for b, _ in tm)) / 1e3 for _, tm in launches]
    with open(os.path.join(rd, "result.json"), "w") as f:
        json.dump({"candidate": cand_id, "spans_us": spans, "node_names": model.node_output_names(pk), "exchange_nodes": [],
                   "pool_nodes": [2], "check": True}, f)
    return rd


def test_round_flow():
    tmp = tempfile.mkdtemp()
    os.environ["MK_LOOP_ROOT"] = tmp
    driver.STATE_ROOT = tmp
    part_dir = os.path.join(tmp, "part")
    os.makedirs(part_dir)
    open(os.path.join(part_dir, "part.yaml"), "w").write(PART_YAML)
    pk, plan, _, _ = compiled(3)
    compiler.write_plan_file(os.path.join(part_dir, "plan.json"), plan)
    driver.init(part_dir, os.path.join(part_dir, "plan.json"), 4)
    rdir = driver.new_round(part_dir, ["search", "explore", "passes"], 60, 1, 1, "costs.json", None, 4)
    spec = json.load(open(os.path.join(rdir, "spec.json")))
    assert spec["fixed_nodes"] == [], spec["fixed_nodes"]          # no measured round yet: nothing known about phases
    inc = spec["incumbent"]
    # the round's results as round_job writes them: one WIN (a pass choice), one LOSS, one exploration
    win = cand_mod.derive(inc, "pass dry_pass=off:1", plan=dict(plan, dry={1: False}), passes={"dry_pass": "off:1"})
    loss = cand_mod.derive(inc, "search", plan=compiled(5)[1])
    for c in (win, loss):
        cand_mod.save(c, os.path.join(rdir, "cand", c["id"], "candidate.json"))
    run_dir = make_run(rdir, "record_incumbent", pk, plan, inc["id"], os.path.join(tmp, "costs.json"))
    schedules, task_times, timing = triage.load_run(run_dir)           # what round_job.triage does with the incumbent's run
    cp = triage.critical_path(schedules, task_times[0], timing[0], [], 2)
    tot = triage.attribute(cp)
    assert abs(tot["sum_us"] - cp["span_us"]) < 1e-6, (tot, cp["span_us"])
    res = {"round": 1, "status": "ok", "dropped": [],
           "runs": [{"name": "record_incumbent", "candidate": inc["id"], "exit": 0, "check": True, "median_us": 7.0, "record": True}],
           "candidates": [{"id": win["id"], "why": win["why"], "passes": win["passes"], "verdict": "WIN",
                           "steps": [{"step": "harness", "verdict": "UP", "deltas": [-0.5, -0.4]},
                                     {"step": "window", "verdict": "WIN", "deltas": [-0.5, -0.6, -0.4, -0.2]}]},
                          {"id": loss["id"], "why": "search", "passes": {}, "verdict": "LOSS",
                           "steps": [{"step": "harness", "verdict": "LOSS", "deltas": [1.0, 1.2]}]}],
           "ablations": [{"pass": "dry_pass", "choice": "off:1", "step": "window", "delta_us": -0.45, "candidate": win["id"]}]}
    json.dump(res, open(os.path.join(rdir, "results.json"), "w"))
    summary = driver.ingest(part_dir, rdir)
    assert summary["promoted"] == win["id"], summary
    state = driver.load_state("fake_part")
    assert state["incumbent"]["4"]["id"] == win["id"]
    assert set(state["phase_nodes"]) == {"front", "experts", "tail"}, state["phase_nodes"]
    kinds = [l["kind"] for l in driver.ledger_lines("fake_part")]
    for k in ("init", "record", "verdict", "ablation", "promote", "model_error"):
        assert k in kinds, (k, kinds)
    print("  model error:", [l["phase_err"] for l in driver.ledger_lines("fake_part") if l["kind"] == "model_error"],
          "trusted:", state["trusted_phases"])
    rdir2 = driver.new_round(part_dir, ["search"], 60, 0, 0, "costs.json", None, 4)
    spec2 = json.load(open(os.path.join(rdir2, "spec.json")))
    untrusted = [ph for ph in ("front", "experts", "tail") if ph not in state["trusted_phases"]]
    want = sorted(k for ph in untrusted for k in state["phase_nodes"][ph])
    assert spec2["fixed_nodes"] == want and spec2["incumbent"]["id"] == win["id"], (spec2["fixed_nodes"], want)
    assert passes.estimate(driver.ledger_lines("fake_part"), "dry_pass", "off:1") is None   # 1 ablation < 3
    driver.status(part_dir)
    shutil.rmtree(tmp)
    print("ok test_round_flow")


def test_passes():
    pk, plan, _, _ = compiled(3)
    new_plan, flags, used = passes.apply_all(pk, plan, {})
    assert used["dry_pass"] == "on" and used["pdl_trigger"] == "2" and flags == [] and "dry" not in new_plan
    abl = passes.ablation_candidates(pk, plan, used)
    names = sorted((a[0], a[1]) for a in abl)
    assert ("dry_pass", "off:1") in names and ("pdl_trigger", "1") in names, names
    p2, f2, _ = passes.apply_all(pk, plan, {"dry_pass": "off:1", "pdl_trigger": "0"})
    assert p2["dry"] == {1: False} and f2 == ["-DSTATIC_PDL_TRIGGER=0"]
    paths, _ = compiler.compile_plan(pk, p2, tempfile.mkdtemp())
    assert json.load(open(paths[0]))["nodes"]["1"]["dry"] is False
    try:
        passes.apply_all(pk, plan, {"prefetch": "l2"})
        raise AssertionError("prefetch=l2 should be infeasible")
    except passes.Infeasible as e:
        assert "prefetch" in str(e)
    lines = [{"kind": "ablation", "pass": "dry_pass", "choice": "off:1", "delta_us": d} for d in (0.4, 0.6, 0.5)]
    assert passes.estimate(lines, "dry_pass", "off:1") == 0.5
    print("ok test_passes")


def test_candidate_ids():
    pk, plan, _, _ = compiled(3)
    a = cand_mod.make("p", 4, plan)
    b = cand_mod.make("p", 4, cand_mod.plan_from_doc(a["plan"]))
    assert a["id"] == b["id"], "the same candidate twice: the same id"
    c = cand_mod.derive(a, "x", nvcc_flags=["-DX"])
    assert c["id"] != a["id"] and c["parent"] == a["id"]
    print("ok test_candidate_ids")


def test_round_job_start():
    """round_job.Round on a real spec from driver.new_round (the parts of a round that need no GPU): the spec, the ladder
    settings, the tree lints with the baseline."""
    from mirage.mpk.loop import lints, round_job
    tmp = tempfile.mkdtemp()
    driver.STATE_ROOT = tmp
    part_dir = os.path.join(tmp, "part")
    os.makedirs(part_dir)
    open(os.path.join(part_dir, "part.yaml"), "w").write(PART_YAML)
    pk, plan, _, _ = compiled(3)
    compiler.write_plan_file(os.path.join(part_dir, "plan.json"), plan)
    driver.init(part_dir, os.path.join(part_dir, "plan.json"), 4)
    rdir = driver.new_round(part_dir, ["explore"], 60, 1, 1, None, None, 4)
    r = round_job.Round(os.path.join(rdir, "spec.json"))
    assert r.ladder["noise_us"] == 0.3 and r.ladder["harness_rounds"] == 2, r.ladder
    tree = lints.apply_baseline(lints.lint_source(lints.tree_paths()))
    assert not lints.failed(tree), [str(f) for f in tree if f.level == "fail"]
    shutil.rmtree(tmp)
    print("ok test_round_job_start")


def test_option_plans_and_pinning():
    """explore.option_plans: one candidate per other grid of a node, every task placed once, spread over the least loaded SMs;
    random_plan leaves the graph's grids as they were (a later plan without explicit grids must not see its random grid)."""
    from mirage.mpk.loop import explore
    pk, plan, _, _ = compiled(3)
    before = [tuple(n.grid) for n in compiler.graph_nodes(pk)]
    explore.random_plan(pk, 2, 8, 7)
    assert [tuple(n.grid) for n in compiler.graph_nodes(pk)] == before, "random_plan re-gridded the graph"
    ps = explore.option_plans(pk, plan)
    assert ps and all(p["grids"][0] != plan["grids"].get(0) for p in ps), [p["why"] for p in ps]
    for p in ps:
        g = p["grids"][0]
        n_tasks = sum(1 for sm_list in p["lists"][0] for node, _ in sm_list if node == 0)
        assert n_tasks == g[0] * g[1] * g[2], (p["why"], n_tasks)
        compiler.compile_plan(pk, {k: v for k, v in p.items() if k != "why"}, tempfile.mkdtemp())
    print("ok test_option_plans_and_pinning")


def test_rebase():
    tmp = tempfile.mkdtemp()
    driver.STATE_ROOT = tmp
    part_dir = os.path.join(tmp, "part")
    os.makedirs(part_dir)
    open(os.path.join(part_dir, "part.yaml"), "w").write(PART_YAML + "graph_options: {deep: false}\n")
    pk, plan, _, _ = compiled(3)
    compiler.write_plan_file(os.path.join(part_dir, "plan.json"), plan)
    driver.init(part_dir, os.path.join(part_dir, "plan.json"), 4)
    driver.rebase(part_dir, os.path.join(part_dir, "plan.json"), 4, "graph changed", {"deep": True})
    st = driver.load_state("fake_part")
    assert st["incumbent"]["4"]["graph_options"] == {"deep": True} and st["errors"] is None and st["phase_nodes"] == {}
    assert driver.ledger_lines("fake_part")[-1]["kind"] == "rebase"
    shutil.rmtree(tmp)
    print("ok test_rebase")


if __name__ == "__main__":
    test_option_plans_and_pinning()
    test_rebase()
    test_round_job_start()
    test_candidate_ids()
    test_passes()
    test_round_flow()
