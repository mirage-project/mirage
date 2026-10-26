"""The round driver, on the local machine (docs/megakernel_dev_loop.md sections 2, 3, 12). The loop's state lives outside the
repo, in STATE_ROOT/<part>/ (default ~/mk_loop):

  state.json      round number, the incumbent candidate per bucket, the promotions since the last milestone, the model's
                  error tracker (model.ErrorTracker), the trusted phases, open questions for the user
  ledger.jsonl    one line per measurement: round, time, candidate, kind (harness / window / ablation / explore / record),
                  verdict (WIN / LOSS / NULL / FAIL-GATE / DIAG / EXPLORE / HARNESS-ONLY), numbers, why it was run
  costs.json      the loop's cost records: pushed to the box with each round's spec, copied back from the round's
                  costs_after.json at ingest
  round_<n>/      spec.json (what the box runs: round_job.py), sync.list, and after fetch the box's results

  python -m mirage.mpk.loop.driver init   --part demo/kimi_k3_moe --plan demo/kimi_k3_moe/solver_plan.json [--bucket 8]
                                          [--costs-seed costs/b300_v22.json]
                                          (again with another --bucket and its own plan: adds that token bucket)
  python -m mirage.mpk.loop.driver rebase --part demo/kimi_k3_kda --plan PLAN --why "port step 2: attn_residual"
                                          (the graph changed: a new incumbent; model error and phase map start over)
  python -m mirage.mpk.loop.driver round  --part demo/kimi_k3_moe [--generate search,explore,passes] [--search-time 300]
                                          [--explore-seeds 3] [--random-seeds 2] [--costs BOX_PATH] [--manifest BOX_PATH]
  python -m mirage.mpk.loop.driver ingest --part demo/kimi_k3_moe ROUND_DIR
  python -m mirage.mpk.loop.driver status --part demo/kimi_k3_moe

`round` prints the box steps (box.py sync / launch / wait / fetch), each its own call. `ingest` writes the ledger lines, updates
the model's error, and promotes the best WIN (one per round; a part without a server window is never promoted by the harness
alone). Milestone: 3 promotions since the last server run (section 12) -> the status says the server and e2e runs are due.
"""
import argparse
import json
import os
import time
from typing import List, Optional

from . import candidate as cand_mod
from . import part as part_mod

STATE_ROOT = os.path.expanduser(os.environ.get("MK_LOOP_ROOT", "~/mk_loop"))
BOX_STATE_ROOT = "~/mk/mk_loop"
MILESTONE_PROMOTIONS = 3


def state_dir(part_name: str) -> str:
    return os.path.join(STATE_ROOT, part_name)


def load_state(part_name: str) -> dict:
    with open(os.path.join(state_dir(part_name), "state.json")) as f:
        return json.load(f)


def save_state(part_name: str, state: dict) -> None:
    path = os.path.join(state_dir(part_name), "state.json")
    with open(path + ".tmp", "w") as f:
        json.dump(state, f, indent=1)
    os.replace(path + ".tmp", path)


def ledger_lines(part_name: str) -> List[dict]:
    path = os.path.join(state_dir(part_name), "ledger.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def append_ledger(part_name: str, lines: List[dict]) -> None:
    with open(os.path.join(state_dir(part_name), "ledger.jsonl"), "a") as f:
        for l in lines:
            f.write(json.dumps(l, sort_keys=True) + "\n")


def init(part_dir: str, plan_path: str, bucket: Optional[int], costs_seed: Optional[str] = None) -> None:
    from ..compiler import plan_from_file
    part = part_mod.load(part_dir, with_hooks=False)
    d = state_dir(part.name)
    tokens = bucket or part.buckets()[0]
    part.bucket(tokens)   # raises for a bucket part.yaml does not have
    if os.path.exists(os.path.join(d, "state.json")):   # a part already set up: add a bucket (each bucket needs its own plan:
        state = load_state(part.name)                   # its task set depends on the tokens)
        if str(tokens) in state["incumbent"]:
            raise SystemExit(f"{d}: bucket {tokens} already has an incumbent ({state['incumbent'][str(tokens)]['id']})")
    else:
        os.makedirs(d, exist_ok=True)
        state = {"part": part.name, "part_dir": os.path.relpath(part.dir, REPO), "round": 0, "incumbent": {},
                 "promotions_since_milestone": 0, "errors": None, "trusted_phases": [], "phase_nodes": {}, "questions": [],
                 "created": time.strftime("%Y-%m-%d %H:%M")}
    inc = cand_mod.make(part.name, tokens, plan_from_file(plan_path), part.graph_options(), why=f"initial plan {plan_path}")
    cand_mod.save(inc, os.path.join(d, "cand", inc["id"], "candidate.json"))
    state["incumbent"][str(tokens)] = inc
    if costs_seed:   # the loop's cost file starts as a copy of existing records (their code tag must be the current one)
        import shutil
        shutil.copy(costs_seed, os.path.join(d, "costs.json"))
    save_state(part.name, state)
    append_ledger(part.name, [{"round": 0, "time": time.strftime("%Y-%m-%d %H:%M"), "candidate": inc["id"], "kind": "init",
                               "verdict": "INCUMBENT", "why": inc["why"]}])
    print(f"loop of {part.name}: {d}; incumbent {inc['id']} for bucket {tokens}")


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def rebase(part_dir: str, plan_path: str, bucket: Optional[int], why: str, graph_options: Optional[dict] = None) -> None:
    """The part's graph changed (e.g. a port step added a node): the bucket's incumbent becomes this plan's candidate; the model
    error window and the phase -> node map start over (the nodes changed); the ledger and the cost records stay."""
    from ..compiler import plan_from_file
    part = part_mod.load(part_dir, with_hooks=False)
    state = load_state(part.name)
    tokens = str(bucket or part.buckets()[0])
    old = state["incumbent"].get(tokens)
    inc = cand_mod.make(part.name, int(tokens), plan_from_file(plan_path), part.graph_options(graph_options), why=f"rebase: {why}")
    cand_mod.save(inc, os.path.join(state_dir(part.name), "cand", inc["id"], "candidate.json"))
    state["incumbent"][tokens] = inc
    state["errors"], state["trusted_phases"], state["phase_nodes"] = None, [], {}
    save_state(part.name, state)
    append_ledger(part.name, [{"round": state["round"], "time": time.strftime("%Y-%m-%d %H:%M"), "candidate": inc["id"],
                               "kind": "rebase", "verdict": "INCUMBENT", "parent": old["id"] if old else None, "why": why}])
    print(f"{part.name} bucket {tokens}: incumbent {inc['id']} (rebase: {why}); was {old['id'] if old else None}")


def fixed_nodes_for(state: dict, part: part_mod.Part, inc: dict) -> List[int]:
    """The nodes of the phases the model is not trusted on (mean error above 1 us over the last 10 measured candidates, rule E3):
    the search keeps their placement. Before the first measured round the phases' nodes are not known yet: none fixed (the
    search's candidates then only count through the ladder, which measures them)."""
    untrusted = [ph for ph in part.cfg["phases"] if ph not in state.get("trusted_phases", [])]
    return sorted(int(k) for ph in untrusted for k in (state.get("phase_nodes") or {}).get(ph, []))


SYNC_DIRS = ("python/mirage/mpk", "include/mirage/static_megakernel")


def sync_files(part_dir: str, extra=()) -> List[str]:
    """What the box needs for a round: the compiler, the loop and the templates, the part's directory (tracked files and untracked
    ones that git does not ignore, e.g. the part's plug-in glue, without __pycache__), and part.yaml's sync_extra files."""
    import subprocess
    inside = not os.path.relpath(os.path.join(REPO, part_dir), REPO).startswith("..")
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", *SYNC_DIRS] + ([part_dir] if inside else []),
                         cwd=REPO, capture_output=True, text=True, check=True).stdout.split() + list(extra)
    if not inside:   # a part outside the repo (tests): every file of its directory
        out += [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(REPO, part_dir)) for f in fs]
    return sorted(f for f in set(out) if "__pycache__" not in f and os.path.isfile(os.path.join(REPO, f)))


def new_round(part_dir: str, generate: List[str], search_time: int, explore_seeds: int, random_seeds: int, costs: str,
              manifest: Optional[str], bucket: Optional[int], variants: Optional[List[dict]] = None,
              graph_options: Optional[List[dict]] = None) -> str:
    part = part_mod.load(part_dir, with_hooks=False)
    state = load_state(part.name)
    tokens = str(bucket or part.buckets()[0])
    inc = state["incumbent"][tokens]
    n = state["round"] + 1
    rdir = os.path.join(state_dir(part.name), f"round_{n}")
    os.makedirs(rdir, exist_ok=True)
    box_rdir = f"{BOX_STATE_ROOT}/{part.name}/round_{n}"
    gen = {}
    if "search" in generate:
        gen["search"] = {"time_s": search_time}
    if "explore" in generate:
        gen["explore"] = {"kinds": ["move_node", "option"], "seeds": list(range(100 * n, 100 * n + explore_seeds)),
                          "random": list(range(1000 * n, 1000 * n + random_seeds))}
    if "passes" in generate:
        gen["passes"] = True
    if "options" in generate:
        gen["options"] = {"per_node": 3}
    if variants:
        gen["variants"] = variants
    if graph_options:
        gen["graph_options"] = graph_options
    spec = {"round": n, "part_dir": state["part_dir"], "bucket": int(tokens), "incumbent": inc, "generate": gen,
            "costs": costs or f"{BOX_STATE_ROOT}/{part.name}/costs.json", "manifest": manifest, "fixed_nodes": fixed_nodes_for(state, part, inc),
            "ladder": {"noise_us": 0.3, "harness_rounds": 2, "window_rounds": 4, "reps": 20}, "model_stat": "p75"}
    with open(os.path.join(rdir, "spec.json"), "w") as f:
        json.dump(spec, f, indent=1)
    state["round"] = n
    save_state(part.name, state)
    print(f"round {n} of {part.name}: {rdir}/spec.json (generate {sorted(gen)}; fixed nodes {spec['fixed_nodes']})")
    files = sync_files(state["part_dir"], part.cfg.get("sync_extra", ()))
    with open(os.path.join(rdir, "sync.list"), "w") as f:
        f.write("\n".join(files) + "\n")
    print(f"box steps, each its own call (BOX=user@host, KEY=~/.ssh/...); {len(files)} files in {rdir}/sync.list:")
    print(f"  2. python -m mirage.mpk.loop.box sync --box $BOX --key $KEY {rdir}/sync.list")
    print(f"  3. python -m mirage.mpk.loop.box push --box $BOX --key $KEY {rdir}/spec.json {state_dir(part.name)}/costs.json "
          f"{BOX_STATE_ROOT}/{part.name}/   (spec.json goes into round_{n}/: box.py push puts spec.json there)")
    print(f"  4. python -m mirage.mpk.loop.box launch --box $BOX --key $KEY --log {box_rdir}/log.txt -- "
          f"cd ~/mk/mirage_upstream \\&\\& ~/mk/venv/bin/python -m mirage.mpk.loop.round_job {box_rdir}/spec.json")
    print(f"  5. python -m mirage.mpk.loop.box wait --box $BOX --key $KEY --log {box_rdir}/log.txt --marker ROUND-DONE")
    print(f"  6. python -m mirage.mpk.loop.box fetch --box $BOX --key $KEY {box_rdir} {rdir}")
    print(f"  7. python -m mirage.mpk.loop.driver ingest --part {part_dir} {rdir}")
    return rdir


def ingest(part_dir: str, rdir: str) -> dict:
    """The round's results -> ledger lines, the model error, the promotion. Returns the summary."""
    part = part_mod.load(part_dir, with_hooks=False)
    state = load_state(part.name)
    with open(os.path.join(rdir, "spec.json")) as f:
        spec = json.load(f)
    with open(os.path.join(rdir, "results.json")) as f:
        res = json.load(f)
    now, n, tokens = time.strftime("%Y-%m-%d %H:%M"), spec["round"], str(spec["bucket"])
    inc = spec["incumbent"]
    lines = []
    for run in res.get("runs", []):
        lines.append({"round": n, "time": now, "candidate": run["candidate"], "kind": run.get("kind", "record" if run.get("record") else "harness"),
                      "run": run["name"], "exit": run.get("exit"), "check": run.get("check"), "median_us": run.get("median_us"),
                      "layer_md5": run.get("layer_md5"), "code": run.get("code"), "verdict": None})
    wins = []
    for c in res.get("candidates", []):
        lines.append({"round": n, "time": now, "candidate": c["id"], "kind": "verdict", "verdict": c["verdict"], "why": c["why"],
                      "passes": c.get("passes"), "steps": c.get("steps"), "parent": inc["id"]})
        if c["verdict"] == "WIN":   # the window's deltas (or the harness's, for a part with promote_on: harness)
            w = c["steps"][-1]["deltas"]
            wins.append((sum(w) / len(w), c["id"]))
    for a in res.get("ablations", []):
        lines.append(dict(a, round=n, time=now, kind="ablation", candidate=a["candidate"], parent=inc["id"]))
    for d in res.get("dropped", []):
        lines.append({"round": n, "time": now, "kind": "dropped", "verdict": "FAIL-GATE", **d})
    promoted = None
    if wins and res.get("status") == "ok":
        best = min(wins)[1]
        cpath = os.path.join(rdir, "cand", best, "candidate.json")
        promoted = cand_mod.load(cpath)
        state["incumbent"][tokens] = promoted
        state["promotions_since_milestone"] = state.get("promotions_since_milestone", 0) + 1
        lines.append({"round": n, "time": now, "candidate": best, "kind": "promote", "verdict": "INCUMBENT", "parent": inc["id"],
                      "why": promoted["why"]})
    errors = model_errors(part, rdir, res)
    if errors:
        from .model import ErrorTracker
        tracker = ErrorTracker.from_dict(state.get("errors"))
        why_of = {c["id"]: c["why"] for c in res.get("candidates", [])}
        for e in errors:
            # the trust window: runs near the incumbent (the search starts from it); random plans fill records, their errors
            # stay in the ledger only
            # (a prediction that used nearest-key fallbacks is exploration-grade by design: it stays out too)
            if not why_of.get(e["candidate"], "").startswith("explore random") and not e["fallbacks"]:
                tracker.add(e["candidate"] or e["run"], e["phase_err"])
            state["phase_nodes"] = e["phase_nodes"]
            lines.append({"round": n, "time": now, "candidate": e["candidate"], "kind": "model_error", "run": e["run"],
                          "phase_err": e["phase_err"], "predicted_us": e["predicted_us"], "measured_span_us": e["measured_span_us"],
                          "fallbacks": len(e["fallbacks"])})
        state["errors"] = tracker.to_dict()
        state["trusted_phases"] = tracker.trusted_phases(phases=part.cfg["phases"])
    after = os.path.join(rdir, "costs_after.json")
    if os.path.exists(after):   # the round's records (round_job copies the cost file at its end) become the loop's cost file
        import shutil
        shutil.copy(after, os.path.join(state_dir(part.name), "costs.json"))
    append_ledger(part.name, lines)
    save_state(part.name, state)
    summary = {"round": n, "status": res.get("status"), "candidates": [(c["id"], c["why"], c["verdict"]) for c in res.get("candidates", [])],
               "promoted": promoted["id"] if promoted else None, "trusted_phases": state["trusted_phases"],
               "milestone_due": state["promotions_since_milestone"] >= MILESTONE_PROMOTIONS}
    print(json.dumps(summary, indent=1))
    return summary


def model_errors(part, rdir: str, res: dict) -> List[dict]:
    """Per timing run with a prediction: the model's per-phase error (model.run_error) from the run's prediction.json, schedules
    and stamps."""
    from . import model
    out = []
    for run in res.get("runs", []):
        rd = os.path.join(rdir, "runs", run["name"])
        if not run.get("record") or not os.path.exists(os.path.join(rd, "prediction.json")):
            continue
        try:
            out.append(dict(model.run_error(rd, part.phase_of()), run=run["name"]))
        except (OSError, ValueError, KeyError) as e:
            print(f"model error of {run['name']}: not computed ({e})")
    return out


def status(part_dir: str) -> None:
    part = part_mod.load(part_dir, with_hooks=False)
    state = load_state(part.name)
    lines = ledger_lines(part.name)
    print(f"part {part.name}: round {state['round']}; incumbents " +
          ", ".join(f"bucket {b}: {c['id']} ({c['why']})" for b, c in state["incumbent"].items()))
    print(f"trusted phases {state.get('trusted_phases')}; promotions since the last milestone {state.get('promotions_since_milestone')}"
          + (" -> server + e2e runs due" if state.get("promotions_since_milestone", 0) >= MILESTONE_PROMOTIONS else ""))
    verdicts = [l for l in lines if l.get("kind") == "verdict"]
    for l in verdicts[-10:]:
        print(f"  round {l['round']} {l['candidate']} {l['verdict']:<12} {l.get('why')}")
    if state.get("questions"):
        print("open questions for the user:", *state["questions"], sep="\n  ")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["init", "rebase", "round", "ingest", "status"])
    p.add_argument("round_dir", nargs="?")
    p.add_argument("--part", required=True)
    p.add_argument("--plan", default=None)
    p.add_argument("--bucket", type=int, default=None)
    p.add_argument("--generate", default="search,explore,passes")
    p.add_argument("--search-time", type=int, default=300)
    p.add_argument("--explore-seeds", type=int, default=2)
    p.add_argument("--random-seeds", type=int, default=2)
    p.add_argument("--costs", default=None, help="round: the cost file on the box (default ~/mk/mk_loop/<part>/costs.json, pushed "
                   "from the state dir's costs.json)")
    p.add_argument("--costs-seed", default=None, help="init: a cost file whose records the loop starts from")
    p.add_argument("--why", default="the graph changed", help="rebase: why")
    p.add_argument("--graph-option", action="append", default=[], metavar="NAME=VALUE", help="rebase: the part's graph options")
    p.add_argument("--variant", action="append", default=[], metavar="NODE:NAME=VALUE[,NAME=VALUE]",
                   help="round: a candidate = the incumbent with this template variant (a unit.py KEEP); repeatable")
    p.add_argument("--manifest", default=None, help="the sync manifest on the box (preflight)")
    a = p.parse_args(argv)
    if a.action == "init":
        init(a.part, a.plan, a.bucket, a.costs_seed)
    elif a.action == "rebase":
        opts = {k: json.loads(v) if v[:1] in "0123456789-[{tf" else v for k, v in (kv.split("=", 1) for kv in a.graph_option)}
        rebase(a.part, a.plan, a.bucket, a.why, opts)
    elif a.action == "round":
        vs = [{"node": int(v.split(":", 1)[0]), "variant": {k: int(x) for k, x in (kv.split("=") for kv in v.split(":", 1)[1].split(","))}}
              for v in a.variant]
        new_round(a.part, a.generate.split(","), a.search_time, a.explore_seeds, a.random_seeds, a.costs, a.manifest, a.bucket, vs)
    elif a.action == "ingest":
        ingest(a.part, a.round_dir)
    else:
        status(a.part)


if __name__ == "__main__":
    main()
