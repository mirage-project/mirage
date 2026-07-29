#!/usr/bin/env python3
"""M3-I7 -- pick, and justify, the analysis window for the per-stage table.

The per-stage MPK column is a ONE-STEP measurement, so which step it is decides
what the whole comparison means. Two things have to line up:

  * the step must be a real decode step at the stated batch size, i.e. inside a
    regime with NO prefill activity and as many live requests as the geometry
    can offer; and
  * `concurrency.py` (per-stage wall spans) and `parse_profile.py` (step_us,
    the denominator) must measure the SAME iterations. concurrency.py has no
    window flag -- it takes the midpoint of `schedule_sim.steady_window`'s raw
    window -- so parse_profile's `--warm-iters` is what has to move.

This script replays the exact admission model (`schedule_sim.simulate`, the same
one the runtime is compared against), enumerates every prefill-free regime, and
emits the per-bs `warm_iters` that centres a `--span`-iteration parse window on
concurrency.py's midpoint. It prints the enumeration so the choice is auditable
rather than asserted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))       # .../accept/opt
import schedule_sim as SIM                        # noqa: E402


def runs(sim):
    out, i, n = [], 0, len(sim["iters"])
    while i < n:
        k = SIM.regime_key(sim["iters"][i])
        j = i
        while j < n and SIM.regime_key(sim["iters"][j]) == k:
            j += 1
        out.append((i, j, k))
        i = j
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--msl", type=int, required=True)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--mbt", type=int, default=16)
    ap.add_argument("--batch-sizes", default="1,8,16")
    ap.add_argument("--span", type=int, default=96)
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    plan = {}
    for bs in [int(x) for x in a.batch_sizes.split(",")]:
        sim = SIM.simulate([a.prompt_len] * bs, a.mbt, a.msl)
        rr = runs(sim)
        free = [(i, j, k) for i, j, k in rr if k[1] == 0 and j - i >= 5]
        lo, hi = SIM.steady_window(sim)
        mid = lo + (hi - lo) // 2            # exactly concurrency.py's choice
        span = min(a.span, hi - lo)
        wlo = max(lo, min(mid - span // 2, hi - span))
        warm = wlo - lo
        # context of the earliest-admitted and latest-admitted live request at mid
        it = sim["iters"][mid]
        plan[str(bs)] = dict(
            n_iterations=sim["n_iterations"],
            raw_steady_window=[lo, hi],
            raw_regime_live_prefill_decode_tokens=list(SIM.regime_key(sim["iters"][lo])),
            concurrency_iteration=mid,
            span=span, warm_iters=warm,
            parse_window=[wlo, wlo + span],
            n_live_at_mid=it["n_live"],
            prefill_free_runs=[dict(start=i, stop=j, length=j - i,
                                    regime=list(k)) for i, j, k in free],
        )
        print(f"bs{bs}: n_it={sim['n_iterations']:5d} raw_steady={[lo, hi]} "
              f"regime={SIM.regime_key(sim['iters'][lo])} conc_mid={mid} "
              f"-> parse [{wlo},{wlo + span}) warm={warm} span={span}")
        for i, j, k in free:
            print(f"      prefill-free [{i},{j}) len={j - i:4d} "
                  f"(live,prefill,decode,tokens)={k}")
    if a.out:
        Path(a.out).write_text(json.dumps(plan, indent=1) + "\n")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
