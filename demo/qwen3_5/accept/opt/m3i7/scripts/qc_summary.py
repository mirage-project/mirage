#!/usr/bin/env python3
"""M3-I7 -- collapse the per-bs anchor-QC records into the verdict that goes
into ferret_targets.json's `basis.anchor_qc`.

A per-stage table is only as good as the window it was measured in, and the
window is only valid if the task-type counts the trace implies match the counts
the compiled task graph actually contains. anchor_qc.py already computes that;
this turns its three files into one machine-readable verdict so a consumer of
ferret_targets.json can see, without opening anything else, which batch sizes
the numbers are load-bearing for.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qc-dir", required=True)
    ap.add_argument("--prefix", default="armL")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    out = {}
    for p in sorted(glob.glob(os.path.join(a.qc_dir, f"{a.prefix}_bs*_rep0_qc.json"))):
        bs = re.search(r"_bs(\d+)_", os.path.basename(p)).group(1)
        d = json.loads(Path(p).read_text())
        q, w = d["anchor_qc"], d["windowing_crosscheck"]
        rot = d.get("rotation_validation", {})
        out[bs] = {
            "window": w["steady_window"],
            "window_step_us": w["steady_window_step_us"],
            "wave_average_step_us": w["full_span_step_us"],
            "window_vs_wave_pct": w["pct_diff"],
            "max_frac_err_over_all_types": q["max_frac_err_over_all_types"],
            "threshold": q["threshold"],
            "window_valid": q["window_valid"],
            "n_task_types_mismatched": q["n_task_types_mismatched_static_count"],
            "trace_iterations": q["n_iterations_full_span"] + 1,
            "call_site_rotation_aligned": {k: v.get("aligned") for k, v in rot.items()},
            "verdict": "PASS" if q["window_valid"] else "FAIL",
        }
    passing = [bs for bs, v in out.items() if v["verdict"] == "PASS"]
    failing = [bs for bs, v in out.items() if v["verdict"] != "PASS"]
    out["_summary"] = {
        "pass": passing, "fail": failing,
        "meaning": ("anchor QC compares the per-step task-type counts the trace implies "
                    "against the counts the compiled task graph contains; a window that "
                    "fails is not a step at the stated batch size. Rows for a FAILing bs "
                    "are retained for continuity and flagged, not used for ranking (the "
                    "rank is computed on bs1)."),
    }
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: (v if k == "_summary" else v["verdict"]) for k, v in out.items()},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
