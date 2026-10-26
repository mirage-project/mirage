"""A candidate: what one measurement measures (docs/megakernel_dev_loop.md principle 1), as one file, candidate.json:

  id             a hash of everything below that changes the built kernel (part, bucket, plan, graph_options, nvcc_flags,
                 passes), so the same candidate made twice gets the same id
  part, bucket   the part's name, its token bucket
  plan           compiler.compile_plan's plan: grids, lists, and the optional choices compile_plan reads: cluster_launch,
                 dry ({graph_idx: false} turns a node's dry pass off), variants ({graph_idx: {name: value}})
  graph_options  the part's graph options (part.yaml graph_options; e.g. MoE rmsnorm_recompute)
  nvcc_flags     extra build flags (e.g. -DSTATIC_PDL_TRIGGER=1)
  passes         {pass name: its choice} (passes.py), as applied
  parent, why    the candidate it was made from, and why (a triage class, an exploration target, a pass ablation)
"""
import copy
import hashlib
import json
import os
from typing import Optional

ID_FIELDS = ("part", "bucket", "plan", "graph_options", "nvcc_flags", "passes")


def plan_doc(plan: dict) -> dict:
    """A plan as JSON (int keys -> str, tuples -> lists), with its optional choices."""
    doc = {"grids": {str(k): list(v) for k, v in plan["grids"].items()},
           "lists": [[[[node, list(pos)] for node, pos in sm_list] for sm_list in gpu_lists] for gpu_lists in plan["lists"]]}
    if "cluster_launch" in plan:
        doc["cluster_launch"] = bool(plan["cluster_launch"])
    if plan.get("dry"):
        doc["dry"] = {str(k): bool(v) for k, v in plan["dry"].items()}
    if plan.get("variants"):
        doc["variants"] = {str(k): dict(v) for k, v in plan["variants"].items()}
    return doc


def plan_from_doc(doc: dict) -> dict:
    plan = {"grids": {int(k): tuple(v) for k, v in doc["grids"].items()},
            "lists": [[[(node, tuple(pos)) for node, pos in sm_list] for sm_list in gpu_lists] for gpu_lists in doc["lists"]]}
    if "cluster_launch" in doc:
        plan["cluster_launch"] = bool(doc["cluster_launch"])
    if doc.get("dry"):
        plan["dry"] = {int(k): bool(v) for k, v in doc["dry"].items()}
    if doc.get("variants"):
        plan["variants"] = {int(k): dict(v) for k, v in doc["variants"].items()}
    return plan


def make(part: str, bucket: int, plan: dict, graph_options: Optional[dict] = None, nvcc_flags=(), passes: Optional[dict] = None,
         parent: Optional[str] = None, why: str = "") -> dict:
    cand = {"part": part, "bucket": int(bucket), "plan": plan_doc(plan), "graph_options": dict(graph_options or {}),
            "nvcc_flags": list(nvcc_flags), "passes": dict(passes or {}), "parent": parent, "why": why}
    cand["id"] = cand_id(cand)
    return cand


def cand_id(cand: dict) -> str:
    body = json.dumps({k: cand.get(k) for k in ID_FIELDS}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(body.encode()).hexdigest()[:10]


def derive(cand: dict, why: str, plan: Optional[dict] = None, **changes) -> dict:
    """A new candidate from `cand`: its plan replaced (plan, as compile_plan takes it) and/or top-level fields changed; parent =
    cand's id."""
    new = copy.deepcopy(cand)
    if plan is not None:
        new["plan"] = plan_doc(plan)
    for k, v in changes.items():
        if k not in ID_FIELDS:
            raise KeyError(f"candidate field {k} is not one of {ID_FIELDS}")
        new[k] = v
    new["parent"], new["why"] = cand["id"], why
    new["id"] = cand_id(new)
    return new


def save(cand: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cand, f)


def load(path: str) -> dict:
    with open(path) as f:
        cand = json.load(f)
    if cand.get("id") != cand_id(cand):
        raise ValueError(f"{path}: its id {cand.get('id')} is not the hash of its fields ({cand_id(cand)}): edited by hand?")
    return cand
