#!/usr/bin/env python3
"""M4-I5 -- the LATENCY floor of the compiled task graph.

Width can only ever remove the part of the step that comes from too few tasks.
What it cannot remove is the dependency chain.  This walks the compiled DAG and
returns the longest weighted chain, which is a hard floor on the decode step at
ANY width:

    task T is gated by event `T.dependent_event`; on finishing it increments
    `T.trigger_event`; an event fires when `num_triggers` producers have
    finished (task_register.cc / runtime.cc).  So
        ready(E)  = the num_triggers-th smallest finish time among E's producers
                    -- bounded above by max(finish) and below by the
                    num_triggers-th order statistic; both are reported
        start(T)  = ready(T.dependent_event)
        finish(T) = start(T) + duration(T)

Durations come from the measured trace (mean live and mean dead per task type,
so the same numbers the width table reports), and can be rescaled per task type
to price a split -- `--split 241:4` divides task 241's live duration by 4.

Two chains are reported:
  * `cp_max`   -- every event waits for ALL its producers.  This is what MPK
                  actually does for a full-fan-in event and is the honest floor.
  * `cp_ntrig` -- every event waits only for its first `num_triggers` producers.
                  A lower bound, printed so the spread is visible.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("width_json")
    ap.add_argument("--split", action="append", default=[],
                    help="task_type:k -- divide that type's live duration by k")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    w = json.load(open(a.width_json))
    dur_live, dur_dead, live_frac = {}, {}, {}
    for r in w["stages"]:
        tt = r["task_type"]
        dur_live[tt] = r["t_live_us"] or r["t_all_us"]
        dur_dead[tt] = r["t_dead_us"] or r["t_all_us"]
        live_frac[tt] = (r["live_per_step"] / r["n_per_step"]
                         if r["n_per_step"] else 0.0)
    scale = {}
    for s in a.split:
        k, v = s.split(":")
        scale[int(k)] = float(v)

    g = json.load(open(a.graph))
    tasks, events = g["all_tasks"], g["all_events"]
    n_t, n_e = len(tasks), len(events)

    # expected duration of a task = live_frac*t_live/k + (1-live_frac)*t_dead
    dur = np.zeros(n_t)
    tt_arr = np.array([t["task_type"] for t in tasks], dtype=np.int32)
    for tt in np.unique(tt_arr):
        tt = int(tt)
        k = scale.get(tt, 1.0)
        f = live_frac.get(tt, 0.0)
        d = f * dur_live.get(tt, 0.0) / k + (1 - f) * dur_dead.get(tt, 0.0)
        dur[tt_arr == tt] = d

    dep = np.array([t["dependent_event"] for t in tasks], dtype=np.int64)
    trg = np.array([t["trigger_event"] for t in tasks], dtype=np.int64)
    SENTINEL = 9223372036854775806
    ntrig = np.array([e["num_triggers"] for e in events], dtype=np.int64)

    # Task ids are emitted in graph order, so id order is topological: assert it.
    prod = [[] for _ in range(n_e)]
    for i in range(n_t):
        if 0 <= trg[i] < n_e:
            prod[trg[i]].append(i)
    bad = 0
    for e in range(n_e):
        if prod[e]:
            first_consumer = min((i for i in range(n_t) if dep[i] == e),
                                 default=None)
            break
    # cheap topological check: no task depends on an event whose producers all
    # have a larger task id
    for i in range(n_t):
        e = dep[i]
        if 0 <= e < n_e and prod[e] and min(prod[e]) > i:
            bad += 1

    ev_ready_max = np.zeros(n_e)
    ev_ready_nt = np.zeros(n_e)
    finish = np.zeros(n_t)
    ev_fin = [[] for _ in range(n_e)]
    for i in range(n_t):
        e = dep[i]
        st_max = ev_ready_max[e] if 0 <= e < n_e else 0.0
        st_nt = ev_ready_nt[e] if 0 <= e < n_e else 0.0
        finish[i] = st_max + dur[i]
        te = trg[i]
        if 0 <= te < n_e:
            ev_ready_max[te] = max(ev_ready_max[te], finish[i])
            ev_fin[te].append(st_nt + dur[i])
            k = int(ntrig[te])
            if len(ev_fin[te]) >= k:
                ev_ready_nt[te] = float(np.partition(np.array(ev_fin[te]),
                                                     k - 1)[k - 1])
            else:
                ev_ready_nt[te] = float(max(ev_fin[te]))

    out = dict(
        batch_size=w["batch_size"], window=w["window"],
        step_measured_us=w["step_us"],
        work_bound_us=w["machine"]["work_bound_us"],
        n_tasks=n_t, n_events=n_e,
        topological_violations=int(bad),
        splits={str(k): v for k, v in scale.items()},
        cp_max_us=round(float(finish.max()), 1),
        cp_ntrig_us=round(float(ev_ready_nt.max()), 1),
        cp_max_pct_of_step=round(100 * float(finish.max()) / w["step_us"], 1),
    )
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
