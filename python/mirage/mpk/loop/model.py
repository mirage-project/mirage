"""The loop's cost model (docs/megakernel_dev_loop.md section 5): durations from cost records, the model's prediction of a plan, and
its error per phase, measured candidate by candidate.

  Model(paths, code, gpu, stat)         the durations search.search_plan reads (task_us, first_us, pool_us), from the records' p75
                                        (rule D4) instead of the median; a key with no record falls back to the nearest key with
                                        records, and the fallback is listed (Model.fallbacks), so a prediction built on it is labelled
                                        exploration, not exploitation. search.Costs (demo.py --search) is unchanged.
  predict(pk, plan, model)              the solver's timeline of a given plan (search_plan(given=plan)) + the fallbacks it used
  node_spans_predicted(summary)         per GPU set per node: (first start, last end) of the prediction, us
  node_spans_measured(schedules, ...)   per GPU per node: (first start, last end) of a timing build, us after the start barrier
  phase_error(pred, meas, ...)          per phase: predicted vs measured phase end, per GPU set; per node first start / last end
  ErrorTracker                          the last 10 candidates' phase errors (state.json); trusted_phases: mean |error| < 1 us (E3)
  fixed_for_untrusted(plan, nodes)      the plan restricted to some nodes: what the search keeps fixed for untrusted phases

Nothing here imports torch or the Mirage extension; search.py is imported only by predict and when a key is needed.
"""
import json
import os
import statistics
from typing import Dict, Iterable, List, Optional, Tuple


def _search():
    from .. import search
    return search


class Model:
    """Durations for search.search_plan from cost record files (cost_records.py), of one kernel version (code: static_schedule.code_tag;
    None: any) on one GPU type (gpu; None: any).

    stat "p75": a record's value is its p75_us (records written before p75_us existed: their median_us); "median": median_us (then
    the values are search.Costs's). Several records with the same task type, key and first-on-SM context (one per run): the median of
    their values, as search.Costs.

    fallback True: a key with no record takes the nearest key's value, and the use is appended to self.fallbacks as (node name,
    graph_idx, grid asked, producer grids asked, what was used). Nearest = among the keys with the same task type, params and
    shapes: the grid whose number of tasks is closest to the asked grid's, then the producer grids ("inputs") whose numbers of tasks
    are closest (ties: the smaller sum of per-axis differences). Example: the router gemm_tile asked at (7, 28, 1) = 196 tasks,
    with records at (7, 14, 1) = 98 and (14, 14, 1) = 196: (14, 14, 1); topk_route asked after a router of (7, 28, 1), with records
    after (7, 14, 1) and (7, 8, 1): after (7, 14, 1). The same total work spread over the same number of tasks gives tasks of
    about the same size, which is the closest guess the records allow; it is still a guess, hence the label. Rule D6 forbids the search from picking an option without records: for a search
    use fallback False (then a missing key gives None, as search.Costs)."""

    def __init__(self, paths, code: Optional[str] = None, gpu: Optional[str] = None, stat: str = "p75",
                 fallback: bool = True):
        if stat not in ("p75", "median"):
            raise ValueError(f"stat {stat}: p75 or median")
        self.stat, self.fallback = stat, fallback
        self.fallbacks: List[Tuple[str, int, tuple, tuple, str]] = []
        self.records = []
        for path in ([paths] if isinstance(paths, str) else list(paths or [])):
            if path and os.path.exists(path):
                with open(path) as f:
                    self.records += [r for r in json.load(f)["records"] if self._code_ok(r, code)
                                     and (gpu is None or (r.get("machine") or {}).get("gpu") == gpu)]
        self._by_type: Dict[str, List[dict]] = {}
        for r in self.records:
            self._by_type.setdefault(r["task_type"], []).append(r)

    _task_tags: Dict[str, str] = {}

    @classmethod
    def _code_ok(cls, r: dict, code: Optional[str]) -> bool:
        """code None: every record; "current": records whose task_code is the current code of their task type
        (static_schedule.task_code_tag; records without one are dropped); else records of that global code tag."""
        if code is None:
            return True
        if code != "current":
            return r.get("code") == code
        if r["task_type"] not in cls._task_tags:
            from ..static_schedule import task_code_tag
            cls._task_tags[r["task_type"]] = task_code_tag(r["task_type"])
        return r.get("task_code") == cls._task_tags[r["task_type"]]

    def value(self, r: dict) -> float:
        return r.get("p75_us", r["median_us"]) if self.stat == "p75" else r["median_us"]

    def stat_of(self, task_type: str, key: dict, first: Optional[bool] = None) -> Optional[float]:
        """The median over the matching records of their value; first True / False: only records of tasks first / not first on
        their SM."""
        values = [self.value(r) for r in self._by_type.get(task_type, ()) if r["key"] == key
                  and (first is None or (r.get("context") or {}).get("first_on_sm") is first)]
        return statistics.median(values) if values else None

    def _behind(self, task_type: str, key: dict) -> Optional[float]:
        us = self.stat_of(task_type, key, first=False)
        return self.stat_of(task_type, key) if us is None else us

    def nearest_key(self, task_type: str, key: dict) -> Optional[dict]:
        """The key with records nearest to `key` (see the class comment), or None."""
        def tasks(grid):
            return grid[0] * grid[1] * grid[2] if grid else 0

        def inputs_apart(k):
            a, b = k.get("inputs") or {}, key.get("inputs") or {}
            if set(a) != set(b):
                return float("inf")
            return sum(abs(tasks(a[i]) - tasks(b[i])) for i in a)
        same = [r["key"] for r in self._by_type.get(task_type, ())
                if r["key"].get("params") == key.get("params") and r["key"].get("shapes") == key.get("shapes") and r["key"] != key]
        if not same:
            return None
        best = min(same, key=lambda k: (abs(tasks(k["grid"]) - tasks(key["grid"])), inputs_apart(k),
                                        sum(abs(a - b) for a, b in zip(k["grid"], key["grid"])), json.dumps(k, sort_keys=True)))
        return None if inputs_apart(best) == float("inf") else best

    def task_us(self, node, grid: tuple, producer_grids: list) -> Optional[float]:
        """A task's time behind another task on its SM (else of all its records), as search.Costs.task_us, from this model's stat;
        no record: the cluster variant's (a node that can pair), then (fallback) the nearest key's."""
        s = _search()
        key = s.cost_key(node, grid, producer_grids)
        us = self._behind(node.name, key)
        if us is None and s.can_pair(node, grid):
            us = self._behind(node.name, dict(key, params=list(node.info["cluster_pair_params"])))
        if us is None and self.fallback:
            near = self.nearest_key(node.name, key)
            if near is not None:
                us = self._behind(node.name, near)
                used = f"grid {tuple(near['grid'])}" + (f" inputs {near['inputs']}" if near.get("inputs") else "")
                self.fallbacks.append((node.name, node.graph_idx, tuple(grid),
                                       tuple(tuple(g) if g is not None else None for g in producer_grids), used))
        return None if us is None else max(us, 0.0)

    def first_us(self, node, grid: tuple, producer_grids: list) -> Optional[float]:
        """A task's time first on its SM; None: no such record (no fallback: the solver then uses task_us)."""
        return self.stat_of(node.name, _search().cost_key(node, grid, producer_grids), first=True)

    def pool_us(self, node) -> Optional[float]:
        """A one-task-per-SM node's total work; None for other nodes."""
        if not node.info["one_task_per_sm"]:
            return None
        return self.stat_of(node.name, _search().cost_key(node, node.grid))


def predict(pk, plan: dict, model: Model, time_limit_s: float = 60.0) -> dict:
    """search.search_plan's summary for the given plan (its grids and lists fixed: the solver only computes its times), plus
    "fallbacks": the fallbacks of the plan's own options (search_plan asks for every option of every node; the others are
    dropped here)."""
    s = _search()
    from ..compiler import graph_nodes, producer_of
    model.fallbacks = []
    _, summary = s.search_plan(pk, model, time_limit_s=time_limit_s, given=plan)
    nodes = graph_nodes(pk)
    grid_of = {n.graph_idx: tuple(plan["grids"].get(n.graph_idx, n.grid)) for n in nodes}
    producers_of = {n.graph_idx: tuple(grid_of[p.graph_idx] if p else None for p in (producer_of(t, nodes) for t in n.inputs))
                    for n in nodes}
    summary["fallbacks"] = sorted({f for f in model.fallbacks if grid_of.get(f[1]) == f[2] and producers_of.get(f[1]) == f[3]},
                                  key=str)
    return summary


def node_spans_predicted(summary: dict) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """Per GPU set, per node: (first start, last end) in us of search_plan's timeline (search.prediction); a pool: (open, end).
    {"gpus_0_2_4_6": {26: (0.0, 9.1), ..., 38: (14.2, 51.0)}, ...}"""
    out = {}
    for gpu_set, tl in summary["timeline"].items():
        spans: Dict[int, Tuple[float, float]] = {}
        for sm in tl["sms"]:
            for t in sm:
                a, b = spans.get(t["node"], (float("inf"), float("-inf")))
                spans[t["node"]] = (min(a, t["start"]), max(b, t["end"]))
        for p in tl.get("pools", []):
            spans[int(str(p["id"]).split("@")[0])] = (p["open"], p["end"])
        out[gpu_set] = spans
    return out


def node_spans_measured(schedules: List[dict], task_times, timing) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """Per GPU, per node: (first start, last end) in us after the GPU's start barrier, the median over launches.
    task_times[launch][gpu][sm][entry] = (start, end, stage 0, stage 1) ns (run.py's task_times.npy, Harness.task_times);
    timing[launch][gpu] = (start barrier passed, end) ns. A start is the stamp at the SM's list entry: a task that waits for its
    inputs inside its body has started by then, so a first start can be earlier than the prediction's (which starts a task when its
    inputs are ready); last ends compare directly."""
    per: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    for times, tim in zip(task_times, timing):
        for g, s in enumerate(schedules):
            bar = tim[g][0]
            spans: Dict[int, Tuple[float, float]] = {}
            for sm, sm_list in enumerate(s["worker_task_queues"]):
                for i, t in enumerate(sm_list):
                    start, end = times[g][sm][i][0], times[g][sm][i][1]
                    if not start or not end:
                        break
                    node = s["all_tasks"][t]["node"]
                    a, b = spans.get(node, (float("inf"), float("-inf")))
                    spans[node] = (min(a, (start - bar) / 1e3), max(b, (end - bar) / 1e3))
            for node, v in spans.items():
                per.setdefault((g, node), []).append(v)
    out: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for (g, node), vs in per.items():
        out.setdefault(g, {})[node] = (statistics.median(v[0] for v in vs), statistics.median(v[1] for v in vs))
    return out


def node_output_names(pk) -> Dict[int, str]:
    """graph_idx -> the name of the node's first output tensor (the names part.yaml's phases use)."""
    from ..compiler import graph_nodes
    return {n.graph_idx: pk._tensor_names[n.outputs[0].guid] for n in graph_nodes(pk)}


def phase_error(pred: Dict[str, Dict[int, Tuple[float, float]]], meas: Dict[int, Dict[int, Tuple[float, float]]],
                phase_of: Dict[str, str], node_names: Dict[int, str], gpu_sets: Dict[str, List[int]]) -> dict:
    """pred: node_spans_predicted; meas: node_spans_measured; phase_of: output tensor name -> phase (part.Part.phase_of); node_names:
    node_output_names; gpu_sets: {GPU set: its GPUs} (summary["timeline"][set]["gpus"]). Measured values of a GPU set = the max over
    its GPUs. Errors are measured - predicted (positive: the model is early).
    Returns {"phases": {phase: {gpu set: {"pred", "meas", "err"}}}, "nodes": {graph_idx: {gpu set: {"first_err", "last_err"}}},
    "phase_err": {phase: the error of largest size over the GPU sets}}.
    Example: phase tail = nodes 39..42; set gpus_0_2_4_6 predicted latest end 61.2, GPUs 0, 2, 4, 6 measured 66.1, 66.8, 66.5,
    66.3 -> meas 66.8, err 5.6."""
    phases: Dict[str, Dict[str, dict]] = {}
    nodes: Dict[int, Dict[str, dict]] = {}
    for gpu_set, gpus in gpu_sets.items():
        p = pred.get(gpu_set, {})
        m: Dict[int, Tuple[float, float]] = {}
        for g in gpus:
            for node, (a, b) in meas.get(g, {}).items():
                a0, b0 = m.get(node, (float("-inf"), float("-inf")))
                m[node] = (max(a0, a), max(b0, b))
        for node in sorted(set(p) & set(m)):
            nodes.setdefault(node, {})[gpu_set] = {"first_err": round(m[node][0] - p[node][0], 3),
                                                   "last_err": round(m[node][1] - p[node][1], 3)}
        for phase in sorted(set(phase_of.values())):
            members = [n for n, name in node_names.items() if phase_of.get(name) == phase]
            pe = [p[n][1] for n in members if n in p]
            me = [m[n][1] for n in members if n in m]
            if pe and me:
                phases.setdefault(phase, {})[gpu_set] = {"pred": round(max(pe), 3), "meas": round(max(me), 3),
                                                         "err": round(max(me) - max(pe), 3)}
    worst = {ph: max((v["err"] for v in per.values()), key=abs) for ph, per in phases.items()}
    return {"phases": phases, "nodes": nodes, "phase_err": worst}


class ErrorTracker:
    """The model's phase errors on the last `window` measured candidates (rule E3: a phase is optimized by the search only while
    its mean absolute error is under 1 us). JSON-able (to_dict / from_dict) for state.json.
    t.add("a1b2c3", {"front": 0.4, "experts": -2.1, "tail": 5.6}); t.trusted_phases() -> ["front"]."""

    def __init__(self, window: int = 10, entries: Optional[List[dict]] = None):
        self.window = window
        self.entries: List[dict] = list(entries or [])[-window:]

    def add(self, candidate: str, phase_err: Dict[str, float]) -> None:
        self.entries = [e for e in self.entries if e["candidate"] != candidate]
        self.entries.append({"candidate": candidate, "errors": dict(phase_err)})
        self.entries = self.entries[-self.window:]

    def mean_abs(self) -> Dict[str, float]:
        acc: Dict[str, List[float]] = {}
        for e in self.entries:
            for ph, err in e["errors"].items():
                acc.setdefault(ph, []).append(abs(err))
        return {ph: round(sum(v) / len(v), 3) for ph, v in acc.items()}

    def trusted_phases(self, limit_us: float = 1.0, phases: Iterable[str] = ()) -> List[str]:
        """Phases whose mean absolute error is under limit_us; a phase in `phases` with no measurement yet is not trusted."""
        mean = self.mean_abs()
        return sorted(ph for ph in set(mean) | set(phases) if ph in mean and mean[ph] < limit_us)

    def to_dict(self) -> dict:
        return {"window": self.window, "entries": self.entries, "mean_abs": self.mean_abs()}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ErrorTracker":
        d = d or {}
        return cls(d.get("window", 10), d.get("entries"))


def nodes_of_phases(phases: Iterable[str], phase_of: Dict[str, str], node_names: Dict[int, str]) -> List[int]:
    want = set(phases)
    return sorted(n for n, name in node_names.items() if phase_of.get(name) in want)


def fixed_for_untrusted(plan: dict, nodes: Iterable[int]) -> dict:
    """The plan restricted to `nodes` (the nodes of the untrusted phases): their grids, and per GPU per SM their entries in the
    plan's order. A search from this partial would fix, for these nodes only, their option (the grid), each task's SM and the order
    of consecutive fixed tasks on an SM (solver.solve's fixed_options / fixed_lists already accept any subset of the tasks), and
    leave every other node free. search.search_plan has no input for this yet: it needs a `fixed_nodes` parameter that builds
    fixed_options and fixed_lists from this partial (as it does for `given`) and runs the free search with them.
    {"grids": {38: (2, 8, 1)}, "lists": [[[(38, (0, 0, 0))], [], ...], ...]}"""
    keep = set(nodes)
    return {"grids": {k: v for k, v in plan["grids"].items() if k in keep},
            "lists": [[[(node, tuple(pos)) for node, pos in sm_list if node in keep] for sm_list in gpu_lists]
                      for gpu_lists in plan["lists"]]}


def run_error(run_dir: str, phase_of: Dict[str, str]) -> dict:
    """The model's error on one measured run (a run.py --record --predict directory: result.json, prediction.json, schedules,
    task_times.npy, timing.npy): {"candidate", "phase_err" (phase_error's), "phase_nodes" {phase: [graph_idx]}, "predicted_us",
    "measured_span_us", "fallbacks"}."""
    import json
    import os
    from .triage import load_run
    with open(os.path.join(run_dir, "result.json")) as f:
        result = json.load(f)
    with open(os.path.join(run_dir, "prediction.json")) as f:
        pred = json.load(f)
    schedules, task_times, timing = load_run(run_dir)
    names = {int(k): v for k, v in result["node_names"].items()}
    gpu_sets = {g: tl["gpus"] for g, tl in pred["timeline"].items()}
    err = phase_error(node_spans_predicted(pred), node_spans_measured(schedules, task_times, timing), phase_of, names, gpu_sets)
    phases = sorted(set(phase_of.values()))
    return {"candidate": result.get("candidate"), "phase_err": err["phase_err"], "nodes": err["nodes"],
            "phase_nodes": {ph: nodes_of_phases([ph], phase_of, names) for ph in phases},
            "predicted_us": pred["predicted_us"], "measured_span_us": statistics.median(result["spans_us"]),
            "fallbacks": pred.get("fallbacks", [])}
