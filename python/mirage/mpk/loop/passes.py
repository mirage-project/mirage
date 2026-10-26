"""The post-processing passes (docs/megakernel_dev_loop.md section 8), each with the same protocol, run in a fixed order on every
candidate before it is measured. A pass changes only its own choices (and the list numbering); the choices go into the candidate
(candidate.json "passes") and into the context of its cost records.

  Pass.choices(pk, plan)            the choices this pass can make for this plan; the first is the default
  Pass.apply(pk, plan, choice)      -> (plan, nvcc_flags) with the choice applied, or raises Infeasible(reason)
  Pass.features(choice)             the context it adds to cost records
  Pass.constraints(pk)              soft constraints for the search (none of the built passes has one yet)
  estimate(ledger_lines, name, choice)  the pass's measured effect: the median of its ablations in the ledger (>= 3), else None

The order (each pass sees the earlier ones' output) and what each does today:
  1. smem_plan       one shared-memory size for every SM (the most any node needs: static_schedule.smem_code); a per-SM plan needs
                     kernel support that does not exist yet -> only "max"
  2. dry_pass        a node whose task type has a dry pass runs it on its first task on each SM ("on"); "off:<graph_idx>" turns
                     one node's off (plan "dry", compiler.compile_plan)
  3. consumer_place  a consumer behind its last producer on the same SM: not built (no choice but "none")
  4. cluster_pairing sum_rmsnorm's token halves on the 2 CTAs of a cluster (compiler.pair_clusters): "on" when the pairs fit,
                     "off" = the plain launch (plan "cluster_launch")
  5. handoff_kind    one hand-off kind per task type today (counters / 0xFF sentinels, fixed in the templates): only "fixed"
  6. prefetch        no smem / L2 prefetch in the templates: only "none"
  7. pdl_trigger     when the kernel lets the next kernel launch (config.cuh STATIC_PDL_TRIGGER, the plug-in build only): 2
                     (default), 0, 1; a harness run (no STATIC_RESET_IN_KERNEL) ignores it, so only the window measures it

Ablations: for a promoted candidate, ablation_candidates makes one candidate per pass with that pass at another choice; the
round measures each pair and writes the difference to the ledger (kind "ablation"), which estimate reads.
"""
import copy
import statistics
from typing import List, Optional, Tuple


class Infeasible(Exception):
    """A pass's choice cannot be applied to this plan (the reason is the message)."""


class Pass:
    name = ""
    measured_by = "harness"   # the cheapest ladder step that sees this pass's effect: harness or window

    def choices(self, pk, plan: dict) -> List[str]:
        raise NotImplementedError

    def apply(self, pk, plan: dict, choice: str) -> Tuple[dict, List[str]]:
        raise NotImplementedError

    def features(self, choice: str) -> dict:
        return {self.name: choice}

    def constraints(self, pk) -> list:
        return []


class SmemPlan(Pass):
    name = "smem_plan"

    def choices(self, pk, plan):
        return ["max"]

    def apply(self, pk, plan, choice):
        if choice != "max":
            raise Infeasible("a per-SM shared-memory plan needs kernel support (static_schedule.smem_code gives one size)")
        return plan, []


class DryPass(Pass):
    name = "dry_pass"

    def _dry_nodes(self, pk) -> List[int]:
        from ..compiler import graph_nodes
        return [n.graph_idx for n in graph_nodes(pk) if n.info["dry"]]

    def choices(self, pk, plan):
        return ["on"] + [f"off:{k}" for k in self._dry_nodes(pk)]

    def apply(self, pk, plan, choice):
        plan = copy.deepcopy(plan)
        plan.pop("dry", None)
        if choice == "on":
            return plan, []
        k = int(choice.split(":", 1)[1])
        if k not in self._dry_nodes(pk):
            raise Infeasible(f"node {k} has no dry pass")
        plan["dry"] = {k: False}
        return plan, []


class ConsumerPlace(Pass):
    name = "consumer_place"

    def choices(self, pk, plan):
        return ["none"]

    def apply(self, pk, plan, choice):
        if choice != "none":
            raise Infeasible("placing a consumer behind its last producer is not built")
        return plan, []


class ClusterPairing(Pass):
    name = "cluster_pairing"

    def choices(self, pk, plan):
        from ..compiler import can_pair, graph_nodes
        nodes = graph_nodes(pk)
        grids = {int(k): tuple(v) for k, v in plan["grids"].items()}
        return ["on", "off"] if any(can_pair(n, grids.get(n.graph_idx)) for n in nodes) else ["off"]

    def apply(self, pk, plan, choice):
        plan = copy.deepcopy(plan)
        if choice == "off":
            plan["cluster_launch"] = False
            return plan, []
        from ..compiler import apply_grids, can_pair, pair_clusters
        nodes = apply_grids(pk, {int(k): v for k, v in plan["grids"].items()})
        paired = {n.graph_idx for n in nodes if can_pair(n)}
        if not paired:
            raise Infeasible("no node can run its pairs on a cluster with these grids")
        try:   # compile_plan pairs them itself; this only tells whether it can
            for gpu, lists in enumerate(plan["lists"]):
                pair_clusters(nodes, paired, lists, pk._num_sms(), gpu)
        except ValueError as e:
            raise Infeasible(f"the cluster pairs do not fit: {e}")
        plan.pop("cluster_launch", None)
        return plan, []


class HandoffKind(Pass):
    name = "handoff_kind"

    def choices(self, pk, plan):
        return ["fixed"]

    def apply(self, pk, plan, choice):
        if choice != "fixed":
            raise Infeasible("one hand-off kind per task type today (fixed in the templates)")
        return plan, []


class Prefetch(Pass):
    name = "prefetch"

    def choices(self, pk, plan):
        return ["none"]

    def apply(self, pk, plan, choice):
        if choice != "none":
            raise Infeasible("no smem or L2 prefetch in the templates")
        return plan, []


class PdlTrigger(Pass):
    name = "pdl_trigger"
    measured_by = "window"

    def choices(self, pk, plan):
        return ["2", "0", "1"]

    def apply(self, pk, plan, choice):
        if choice not in ("0", "1", "2"):
            raise Infeasible(f"STATIC_PDL_TRIGGER is 0, 1 or 2, not {choice}")
        return plan, ([] if choice == "2" else [f"-DSTATIC_PDL_TRIGGER={choice}"])


PASSES: List[Pass] = [SmemPlan(), DryPass(), ConsumerPlace(), ClusterPairing(), HandoffKind(), Prefetch(), PdlTrigger()]
BY_NAME = {p.name: p for p in PASSES}


def apply_all(pk, plan: dict, choices: Optional[dict] = None) -> Tuple[dict, List[str], dict]:
    """Every pass in order, each with its choice (choices[name], default its first). Returns (plan, nvcc flags, {pass: choice});
    raises Infeasible naming the pass."""
    choices = dict(choices or {})
    unknown = set(choices) - set(BY_NAME)
    if unknown:
        raise KeyError(f"no passes {sorted(unknown)} (passes: {list(BY_NAME)})")
    flags, used = [], {}
    for p in PASSES:
        choice = choices.get(p.name) or p.choices(pk, plan)[0]
        try:
            plan, more = p.apply(pk, plan, choice)
        except Infeasible as e:
            raise Infeasible(f"{p.name}={choice}: {e}")
        flags += more
        used[p.name] = choice
    return plan, flags, used


def features(used: dict) -> dict:
    out = {}
    for name, choice in used.items():
        out.update(BY_NAME[name].features(choice))
    return out


def ablation_candidates(pk, plan: dict, used: dict) -> List[Tuple[str, str, dict, List[str], dict]]:
    """For each pass with more than one choice: (pass, the other choice, plan, nvcc flags, all choices) with only that pass changed
    from `used`. Infeasible choices are left out."""
    out = []
    for p in PASSES:
        current = used.get(p.name) or p.choices(pk, plan)[0]   # a pass the candidate does not name runs its default choice
        for choice in p.choices(pk, plan):
            if choice == current:
                continue
            try:
                new_plan, flags, new_used = apply_all(pk, plan, dict(used, **{p.name: choice}))
            except Infeasible:
                continue
            out.append((p.name, choice, new_plan, flags, new_used))
    return out


def estimate(ledger_lines: List[dict], name: str, choice: str, min_ablations: int = 3) -> Optional[float]:
    """The median measured effect (us; > 0 = the choice is slower than the incumbent's) of `choice` for pass `name` from the
    ledger's ablation lines, or None with fewer than min_ablations."""
    deltas = [l["delta_us"] for l in ledger_lines
              if l.get("kind") == "ablation" and l.get("pass") == name and l.get("choice") == choice and l.get("delta_us") is not None]
    return statistics.median(deltas) if len(deltas) >= min_ablations else None
