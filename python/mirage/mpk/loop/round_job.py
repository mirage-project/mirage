"""One round on the GPU box (docs/megakernel_dev_loop.md section 3, steps 1 and 3-8), from the round's spec.json (driver.py writes
it; driver.py ingest reads what this writes). Every run is a run.py process of its own (one built layer per process).

  1. preflight      preflight_box.sh with the sync manifest (when spec "manifest" is set): md5s, no other GPU processes, fabric;
                    lints tree (findings not in the baseline fail the round); the incumbent's check run and its build lints
  2. generate       from the incumbent, the generators the spec names:
                      search    search.search_plan with the loop's model (model.Model), from the incumbent, untrusted phases fixed
                      explore   explore.perturb_plan / random plans: measured with the timing build only, never compared
                      passes    the incumbent with each other choice of each pass (passes.ablation_candidates)
                      variants  the incumbent with a template variant the kernel sub-loop kept (unit.py KEEP)
                      graph_options  the graph with other part options (a new node, a hand-off kind, padding), placed
                                by the part's start plan (hooks.start_plan)
                      options   the incumbent with one node's split changed, re-placed on the least loaded SMs
                                (explore.option_plans; laddered: what the model is not trusted to rank yet)
                    each candidate through the passes (passes.apply_all) and lint_plan; a failed lint drops it (logged)
  3. ladder         harness: incumbent and candidate alternated (the candidate's first build through lints build: a fail is
                    FAIL-GATE), `harness_rounds` rounds; a candidate worse by > noise in every
                    round stops (LOSS); the others go on to the window (part.yaml window), `window_rounds` alternated rounds,
                    stopped as soon as 2 rounds are not wins; WIN = better by >= noise in >= 3 of 4 rounds. A part without a
                    window stops at the harness (verdict HARNESS-ONLY: never promoted)
  4. record         the timing build (--record, --predict) of the incumbent and of every candidate that was not a LOSS, and of
                    the exploration candidates: cost records, the model's prediction next to the measured stamps
  5. results        <round>/results.json; the last line of the log is ROUND-DONE <ok | reason>

python -m mirage.mpk.loop.round_job ROUND_DIR/spec.json
"""
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from typing import List, Optional

from . import candidate as cand_mod
from . import part as part_mod

HERE = os.path.dirname(os.path.abspath(__file__))


class Round:
    def __init__(self, spec_path: str):
        with open(spec_path) as f:
            self.spec = json.load(f)
        self.dir = os.path.dirname(os.path.abspath(spec_path))
        self.repo = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
        self.part = part_mod.load(os.path.join(self.repo, self.spec["part_dir"]), with_hooks=False)
        self.ladder = {"noise_us": 0.3, "harness_rounds": 2, "window_rounds": 4, "reps": 20, **self.spec.get("ladder", {})}
        self.results = {"round": self.spec["round"], "candidates": [], "ablations": [], "runs": [], "dropped": [], "log": []}
        self.costs = os.path.expanduser(self.spec["costs"])

    # ---- bookkeeping ----
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.results["log"].append(line)

    def save(self) -> None:
        with open(os.path.join(self.dir, "results.json"), "w") as f:
            json.dump(self.results, f, indent=1)

    def cand_path(self, cand: dict) -> str:
        path = os.path.join(self.dir, "cand", cand["id"], "candidate.json")
        if not os.path.exists(path):
            cand_mod.save(cand, path)
        return path

    # ---- one harness run (run.py) ----
    def run(self, cand: dict, name: str, record: bool = False) -> dict:
        out = os.path.join(self.dir, "runs", name)
        cmd = [sys.executable, "-m", "mirage.mpk.loop.run", "--part", self.part.dir, "--candidate", self.cand_path(cand),
               "--reps", str(self.ladder["reps"]), "--out", out]
        if self.spec.get("input_set"):
            cmd += ["--input-set", self.spec["input_set"]]
        if record:
            cmd += ["--record", self.costs, "--predict", self.costs, "--run-name", f"r{self.spec['round']}_{name}"]
        t0 = time.time()
        # a run takes ~15 s (compile included); a deadlocked plan's kernel keeps its process from exiting after run.py's own
        # launch timeout (KDA round 17: 315 s until killed), so the process is killed here
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=self.spec.get("run_timeout_s", 240))
            text, code = p.stdout + p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:   # killed: recorded as a failed run
            text = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            text, code = text + f"\nrun killed after {e.timeout:g} s\n", -9
        with open(out + ".txt", "w") as f:
            f.write(text)
        res = {"name": name, "candidate": cand["id"], "exit": code, "s": round(time.time() - t0, 1), "record": record}
        try:
            with open(os.path.join(out, "result.json")) as f:
                r = json.load(f)
            res.update(check=r.get("check"), median_us=statistics.median(r["spans_us"]) if r.get("spans_us") else None,
                       min_us=min(r["spans_us"]) if r.get("spans_us") else None, layer_md5=r.get("layer_md5"), code=r.get("code"))
        except (OSError, ValueError, KeyError):
            res.update(check=False, median_us=None, error=text[-600:])
        self.results["runs"].append(res)
        self.log(f"run {name}: exit {res['exit']} check {res.get('check')} median {res.get('median_us')} us ({res['s']} s)")
        self.save()
        return res

    def lint_build(self, run: dict, parent: Optional[dict] = None) -> dict:
        """lint_build + lint_sass on a run's build (lints.py build); returns its summary, with "fail" set when a check failed."""
        out = os.path.join(self.dir, "runs", run["name"])
        cmd = [sys.executable, "-m", "mirage.mpk.loop.lints", "build", out]
        if parent and parent.get("path"):
            cmd += ["--parent", parent["path"]]
        p = subprocess.run(cmd, capture_output=True, text=True)
        path = os.path.join(out, "sass_summary.json")
        summary = {"path": path if os.path.exists(path) else None, "fail": p.returncode != 0, "text": p.stdout[-1500:]}
        if summary["fail"]:
            self.log(f"lint build {run['name']}: FAIL " + " | ".join(l for l in p.stdout.splitlines() if l.startswith("FAIL"))[:400])
        return summary

    def triage(self, run_dir: str) -> dict:
        """triage.py on a timing run: the class totals (median over its launches) and the first launch's path."""
        from .triage import attribute, critical_path, load_run
        with open(os.path.join(run_dir, "result.json")) as f:
            r = json.load(f)
        schedules, task_times, timing = load_run(run_dir)
        pool = (r.get("pool_nodes") or [-1])[0]
        totals = [attribute(critical_path(schedules, t, tm, r.get("exchange_nodes", []), pool))["classes"]
                  for t, tm in zip(task_times, timing)]
        keys = sorted({k for t in totals for k in t})
        med = {k: round(statistics.median(t.get(k, 0.0) for t in totals), 3) for k in keys}
        self.log("triage of the incumbent (us on the critical path): " + json.dumps(dict(sorted(med.items(), key=lambda kv: -kv[1]))))
        from .triage import stage_split   # each node's stage stamps, GPU 0, latest over its tasks (us after the start barrier)
        stages = {}
        for name in sorted({n["name"] for n in schedules[0]["nodes"].values()}):
            rows = [x for x in stage_split(schedules, task_times, timing, name) if x["gpu"] == 0]
            if rows:
                stages[name] = {k: round(max(x[k] for x in rows), 3) for k in ("start", "inputs", "s0", "s1", "end")}
        self.log("stages (GPU 0, latest over tasks): " + json.dumps(stages))
        return {"classes": med, "stages": stages,
                "path": critical_path(schedules, task_times[0], timing[0], r.get("exchange_nodes", []), pool)}

    # ---- the server window (part.yaml window) ----
    def window(self, cand: dict, name: str) -> dict:
        w = self.part.cfg.get("window")
        out = os.path.join(self.dir, "window", name)
        os.makedirs(out, exist_ok=True)
        plan_file = os.path.join(out, "plan.json")
        with open(plan_file, "w") as f:
            json.dump(cand["plan"], f)
        fill = {"python": sys.executable, "torchrun": os.path.join(os.path.dirname(sys.executable), "torchrun"),
                "input": self.part.input_dir(cand["bucket"], self.spec.get("input_set")), "plan": plan_file, "out": out,
                "flags": " ".join(f"--nvcc-flag={shlex.quote(f)}" for f in cand.get("nvcc_flags", [])),
                "reps": str(self.ladder["reps"])}
        res = {"name": name, "candidate": cand["id"], "kind": "window"}
        for step in ("build", "run"):
            cmd = w[step].format(**fill)
            p = subprocess.run(cmd, shell=True, cwd=self.part.dir, capture_output=True, text=True, timeout=1800)
            with open(os.path.join(out, f"{step}.txt"), "w") as f:
                f.write(p.stdout + p.stderr)
            if p.returncode != 0:
                res.update(exit=p.returncode, step=step, median_us=None, check=False, error=(p.stdout + p.stderr)[-600:])
                self.log(f"window {name}: {step} failed (exit {p.returncode})")
                self.results["runs"].append(res)
                return res
        text = open(os.path.join(out, "run.txt")).read()
        m, c = re.search(w["result"], text), re.search(w["check"], text)
        ok = bool(c) and c.group(1) == c.group(2)
        res.update(exit=0, median_us=float(m.group(1)) if m else None, check=ok)
        self.results["runs"].append(res)
        self.log(f"window {name}: check {ok} median {res['median_us']} us")
        self.save()
        return res

    # ---- the ladder ----
    def harness_step(self, inc: dict, cand: dict) -> dict:
        noise, deltas = self.ladder["noise_us"], []
        for r in range(self.ladder["harness_rounds"]):
            a = self.run(inc, f"h{r}_inc_{cand['id']}")
            b = self.run(cand, f"h{r}_{cand['id']}")
            if not b.get("check"):
                return {"step": "harness", "verdict": "FAIL-GATE", "deltas": deltas, "reason": "check failed"}
            if r == 0 and self.lint_build(b, getattr(self, "inc_build", None))["fail"]:
                return {"step": "harness", "verdict": "FAIL-GATE", "deltas": deltas, "reason": "lint build / SASS"}
            if a.get("median_us") is None:
                return {"step": "harness", "verdict": "DIAG", "deltas": deltas, "reason": "incumbent run failed"}
            deltas.append(round(b["median_us"] - a["median_us"], 2))
        verdict = "LOSS" if all(d > noise for d in deltas) else "UP"
        return {"step": "harness", "verdict": verdict, "deltas": deltas}

    def window_step(self, inc: dict, cand: dict) -> dict:
        noise, deltas, wins, not_wins = self.ladder["noise_us"], [], 0, 0
        for r in range(self.ladder["window_rounds"]):
            a = self.window(inc, f"w{r}_inc_{cand['id']}")
            b = self.window(cand, f"w{r}_{cand['id']}")
            if not b.get("check"):
                return {"step": "window", "verdict": "FAIL-GATE", "deltas": deltas, "reason": "window check failed"}
            if a.get("median_us") is None or b.get("median_us") is None:
                return {"step": "window", "verdict": "DIAG", "deltas": deltas, "reason": "no window time"}
            d = round(b["median_us"] - a["median_us"], 2)
            deltas.append(d)
            wins, not_wins = wins + (d <= -noise), not_wins + (d > -noise)
            if not_wins >= 2:
                break
        verdict = "WIN" if wins >= 3 else ("LOSS" if all(d > noise for d in deltas) else "NULL")
        return {"step": "window", "verdict": verdict, "deltas": deltas}

    # ---- candidates ----
    def generate(self) -> List[dict]:
        """The round's candidates (passes applied, lint_plan passed), from the incumbent."""
        import torch
        from . import lints, passes
        gen, inc = self.spec.get("generate", {}), self.spec["incumbent"]
        hooks = part_mod.load_hooks(self.part.dir, self.part.cfg)
        tokens = inc["bucket"]
        ws, mpk = hooks.build(self.part.input_dir(tokens, self.spec.get("input_set")), self.part.tp, tokens,
                              self.part.graph_options(inc.get("graph_options")))
        num_sms = mpk._num_sms()
        from ..compiler import graph_nodes
        # the graph's grids as built: every plan below names every changeable node's grid explicitly (random_plan / the
        # perturbations re-grid the graph in place; a plan that left a grid implicit would take whatever grid was left)
        orig = {n.graph_idx: (tuple(n.grid), list(n.params)) for n in graph_nodes(mpk)}
        changeable = [n.graph_idx for n in graph_nodes(mpk) if n.info["changeable_grid_dims"]]

        def restore():
            for idx, (grid, params) in orig.items():
                mpk.kn_graph.regrid(idx, grid, params)

        def pin(plan):
            return dict(plan, grids={**{k: orig[k][0] for k in changeable}, **{int(k): tuple(v) for k, v in plan["grids"].items()}})
        inc_plan = pin(cand_mod.plan_from_doc(inc["plan"]))
        made = []   # (why, plan, pass choices)
        cands = []  # finished candidates (graph-option ones are made directly; the rest below from `made`)
        has_records = os.path.exists(self.costs) and json.load(open(self.costs)).get("records")
        if gen.get("search") and not has_records:
            self.log(f"search skipped: no cost records in {self.costs} yet (the exploration and timing builds of this round make them)")
        if gen.get("search") and has_records:
            from .. import search
            from .model import Model
            model = Model(self.costs, code=self.spec.get("code", "current"), gpu=torch.cuda.get_device_name(0),
                          stat=self.spec.get("model_stat", "p75"))
            kw = {}
            if self.spec.get("fixed_nodes"):
                kw["fixed_nodes"] = [int(k) for k in self.spec["fixed_nodes"]]
            try:
                plan, summary = search.search_plan(mpk, model, time_limit_s=gen["search"].get("time_s", 300), start_from=inc_plan,
                                                   **kw)
            except ValueError as e:   # a node with no records yet (a new task type): this round's timing builds make them
                self.log(f"search skipped: {e}")
                self.results["search"] = {"status": "skipped", "reason": str(e)}
            else:
                self.results["search"] = {k: summary.get(k) for k in ("status", "predicted_us", "bound_us", "solve_s", "grids")}
                self.results["search"]["fallbacks"] = getattr(model, "fallbacks", [])
                made.append(("search", plan, inc.get("passes", {})))
        if gen.get("explore"):
            from .explore import perturb_plan, random_plan
            ex = gen["explore"]
            for seed in ex.get("random", []):
                restore()
                made.append((f"explore random:{seed}", random_plan(mpk, self.part.tp, num_sms, seed), {}))
            for kind in ex.get("kinds", []):
                for seed in ex.get("seeds", []):
                    try:
                        restore()
                        made.append((f"explore {kind}:{seed}", perturb_plan(mpk, inc_plan, seed, kind), inc.get("passes", {})))
                    except ValueError as e:   # e.g. "option" on a graph whose nodes have one grid each
                        self.results["dropped"].append({"why": f"explore {kind}:{seed}", "reason": str(e)[:300]})
        if gen.get("options"):   # the incumbent with one node's split changed (measured on the ladder, not by the model)
            from .explore import option_plans
            restore()
            for plan in option_plans(mpk, inc_plan, gen["options"].get("per_node", 3), gen["options"].get("nodes")):
                why = plan.pop("why")
                plan.update({k: v for k, v in inc_plan.items() if k not in ("grids", "lists")})   # keep its variants / choices
                made.append((why, plan, inc.get("passes", {})))
        for vs in gen.get("variants", []):   # a variant the kernel sub-loop kept (unit.py KEEP): the incumbent with it
            plan = cand_mod.plan_from_doc(inc["plan"])
            variants = dict(plan.get("variants") or {})
            variants[int(vs["node"])] = dict(variants.get(int(vs["node"]), {}), **vs["variant"])
            plan["variants"] = variants
            made.append((f"variant node {vs['node']} {vs['variant']}", plan, inc.get("passes", {})))
        for change in gen.get("graph_options", []):   # a graph-level choice (a new node, another hand-off, padding): the graph
            # with these options, its plan from the part's start plan (hooks.start_plan; the placement starts over)
            opts = self.part.graph_options({**(inc.get("graph_options") or {}), **change})
            ws2, mpk2 = hooks.build(self.part.input_dir(tokens, self.spec.get("input_set")), self.part.tp, tokens, opts)
            try:
                plan = hooks.start_plan(mpk2, self.part.tp, mpk2._num_sms())
                bad = [f for f in lints.lint_plan(mpk2, plan) if f.level == "fail"]
                if bad:
                    self.results["dropped"].append({"why": f"graph {change}", "reason": "lint_plan: " + "; ".join(f.msg for f in bad)[:400]})
                    continue
                cands.append(cand_mod.make(self.part.name, tokens, plan, opts, list(inc.get("nvcc_flags", [])),
                                           inc.get("passes") or {}, parent=inc["id"], why=f"graph {change}"))
            finally:
                ws2 = mpk2 = None
                torch.cuda.empty_cache()
        if gen.get("passes"):
            has_window = bool(self.part.cfg.get("window"))
            for name, choice, _, _, used in passes.ablation_candidates(mpk, inc_plan, inc.get("passes") or {}):
                if passes.BY_NAME[name].measured_by == "window" and not has_window:
                    continue   # only a window sees this pass's effect (e.g. the PDL trigger), and the part has none yet
                made.append((f"pass {name}={choice}", inc_plan, used))
        for why, plan, choices in made:
            restore()
            plan = pin(plan)
            try:
                plan2, flags, used = passes.apply_all(mpk, plan, choices)
            except passes.Infeasible as e:
                self.results["dropped"].append({"why": why, "reason": f"infeasible: {e}"})
                continue
            bad = [f for f in lints.lint_plan(mpk, plan2) if f.level == "fail"]
            if bad:
                self.results["dropped"].append({"why": why, "reason": "lint_plan: " + "; ".join(f"{f.rule} {f.msg}" for f in bad)[:500]})
                continue
            c = cand_mod.make(self.part.name, tokens, plan2, inc.get("graph_options"), list(inc.get("nvcc_flags", [])) + flags,
                              used, parent=inc["id"], why=why)
            if c["id"] != inc["id"] and all(c["id"] != x["id"] for x in cands):
                cands.append(c)
        restore()
        ws = mpk = None   # the tensors freed before the runs (each run.py process loads its own)
        torch.cuda.empty_cache()
        self.log(f"generated {len(cands)} candidates, dropped {len(self.results['dropped'])}")
        return cands

    def main(self) -> str:
        if self.spec.get("manifest"):
            p = subprocess.run(["bash", os.path.join(HERE, "preflight_box.sh"), os.path.expanduser(self.spec["manifest"])],
                               capture_output=True, text=True)
            self.results["preflight"] = p.stdout[-2000:]
            self.log(p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "preflight: no output")
            if p.returncode != 0:
                return "preflight"
        from . import lints
        tree = lints.apply_baseline(lints.lint_source(lints.tree_paths()))
        self.results["lint_tree"] = [str(f) for f in tree]
        if lints.failed(tree):
            self.log("lint tree: " + "; ".join(str(f) for f in tree if f.level == "fail")[:500])
            return "lint-tree"
        inc = self.spec["incumbent"]
        base = self.run(inc, "incumbent_check")
        if not base.get("check"):
            return "incumbent-check"
        self.inc_build = self.lint_build(base)
        cands = self.generate()
        has_window = bool(self.part.cfg.get("window"))
        for c in cands:
            entry = {"id": c["id"], "why": c["why"], "passes": c["passes"], "steps": []}
            self.results["candidates"].append(entry)
            if c["why"].startswith("explore"):
                entry["verdict"] = "EXPLORE"
                continue
            h = self.harness_step(inc, c)
            entry["steps"].append(h)
            entry["verdict"] = h["verdict"]
            if h["verdict"] == "UP" and not has_window and self.part.cfg.get("promote_on") == "harness":
                # a part without a server window yet (part.yaml promote_on: harness): faster by >= noise in every harness round
                entry["verdict"] = "WIN" if all(d <= -self.ladder["noise_us"] for d in h["deltas"]) else "NULL"
            elif h["verdict"] == "UP":
                if has_window:
                    w = self.window_step(inc, c)
                    entry["steps"].append(w)
                    entry["verdict"] = w["verdict"]
                else:
                    entry["verdict"] = "HARNESS-ONLY"
            if c["why"].startswith("pass "):
                d = entry["steps"][-1]["deltas"]
                name, choice = c["why"][5:].split("=", 1)
                self.results["ablations"].append({"pass": name, "choice": choice, "step": entry["steps"][-1]["step"],
                                                  "delta_us": statistics.median(d) if d else None, "candidate": c["id"]})
            self.save()
        # timing builds: the incumbent, every candidate that was not a LOSS / FAIL, the exploration candidates
        rec = self.run(inc, "record_incumbent", record=True)
        if rec.get("check"):
            self.results["triage"] = self.triage(os.path.join(self.dir, "runs", "record_incumbent"))
        for c, e in zip(cands, self.results["candidates"]):
            if e["verdict"] not in ("LOSS", "FAIL-GATE"):
                self.run(c, f"record_{c['id']}", record=True)
        return "ok"


KEEPALIVE = ("~/stop_keepalive.sh", "~/start_keepalive.sh")   # the box's keepalive load must be off while we measure


def box_env() -> None:
    """The box's build environment for every run this job starts (as the job scripts set it): nvcc on PATH, CUDA_HOME, and no
    MIRAGE_ROOT (the editable install's tree is the one synced)."""
    cuda = "/usr/local/cuda"
    if os.path.isdir(cuda):
        os.environ["PATH"] = f"{cuda}/bin:" + os.environ.get("PATH", "")
        os.environ.setdefault("CUDA_HOME", cuda)
    os.environ.pop("MIRAGE_ROOT", None)


def main(argv: Optional[list] = None):
    spec = (argv or sys.argv[1:])[0]
    box_env()
    r = Round(spec)
    stop, start = (os.path.expanduser(k) for k in KEEPALIVE)
    if os.path.exists(stop):
        subprocess.run(["bash", stop])
        r.log("keepalive off")
    try:
        status = r.main()
    except Exception as e:   # the round's reason goes to the log's last line and results.json
        import traceback
        traceback.print_exc()
        status = f"error {type(e).__name__}: {str(e)[:200]}"
    finally:
        if os.path.exists(start):
            subprocess.run(["bash", start])
            r.log("keepalive on")
    r.results["status"] = status
    if os.path.exists(r.costs):   # the cost file with this round's records, fetched with the round (driver.py ingest)
        import shutil
        shutil.copy(r.costs, os.path.join(r.dir, "costs_after.json"))
    r.save()
    print(f"ROUND-DONE {status}", flush=True)


if __name__ == "__main__":
    main()
