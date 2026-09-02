"""Choose where the task boundaries go, instead of writing them by hand."""
from __future__ import annotations

import dataclasses
import itertools
from typing import Iterator, Optional

from .group import Group, make_group
from .node import ModelGraph, OPAQUE, is_opaque


MAX_GROUP_OPS = 6

MAX_GROUP_INPUTS = 3


def _runs(n: int, forced_cuts: set[int], max_len: int) -> Iterator[list[tuple[int, int]]]:
    """Every way to cut [0, n) into contiguous runs of at most max_len,
    always cutting at the given positions. Yields (start, stop) pairs."""
    optional = [i for i in range(1, n) if i not in forced_cuts]
    for k in range(len(optional) + 1):
        for extra in itertools.combinations(optional, k):
            cuts = sorted(forced_cuts | set(extra) | {0, n})
            runs = list(zip(cuts, cuts[1:]))
            if all(b - a <= max_len for a, b in runs):
                yield runs


def enumerate_partitions(
    graph: ModelGraph,
    node_ids: Optional[list[int]] = None,
    *,
    max_group_ops: int = MAX_GROUP_OPS,
) -> Iterator[list[Group]]:
    """Candidate partitions of `node_ids` (default: the whole graph)."""
    ids = list(range(len(graph))) if node_ids is None else sorted(node_ids)
    forced = set()
    for pos, i in enumerate(ids):
        if is_opaque(graph.nodes[i].op):
            forced.add(pos)
            forced.add(pos + 1)
    forced.discard(0)
    forced.discard(len(ids))

    for runs in _runs(len(ids), forced, max_group_ops):
        try:
            yield [make_group(graph, ids[a:b], _tag(graph, ids[a:b]))
                   for a, b in runs]
        except ValueError:
            continue


def _tag(graph: ModelGraph, ids) -> str:
    ops = [graph.nodes[i].op for i in ids]
    if len(ops) == 1 and is_opaque(ops[0]):
        return ops[0].split(":", 1)[1]
    return "_".join(ops)



@dataclasses.dataclass
class Rejection:
    stage: str
    reason: str


def _edges(graph: ModelGraph, partition: list[Group]) -> dict[int, set[int]]:
    """Producer group -> consumer groups, after residual stripping."""
    produced_by = {}
    for gi, g in enumerate(partition):
        for i in g.nodes:
            produced_by[graph.nodes[i].output.name] = gi

    succ: dict[int, set[int]] = {gi: set() for gi in range(len(partition))}
    for gi, g in enumerate(partition):
        for v in g.external_inputs:
            pg = produced_by.get(v.name)
            if pg is not None and pg != gi:
                succ[pg].add(gi)

    # reachable-in-two-or-more-hops, then drop those direct edges
    stripped = {gi: set(cs) for gi, cs in succ.items()}
    for u, cs in succ.items():
        seen, frontier = set(), set()
        for m in cs:                       # one hop
            frontier |= succ[m]            # two hops
        while frontier:
            seen |= frontier
            nxt = set()
            for m in frontier:
                nxt |= succ[m] - seen
            frontier = nxt
        stripped[u] -= seen                # v also reachable the long way
    return stripped


def default_partition(graph: ModelGraph, opaque_runs=(),
                      fuse: bool = True) -> list[Group]:
    """One group per node, plus two fusions: a silu/rms_norm whose only
    consumer is a mul, and every node between an `opaque_runs` pair."""
    fused = {}
    for i, n in enumerate(graph.nodes):
        # fuse=False gives one task per node, for partition comparisons.
        if fuse and n.op in ("silu", "rms_norm"):
            cons = graph.consumers(n.output)
            if len(cons) == 1 and graph.nodes[cons[0]].op == "mul":
                tag = "silu_mul" if n.op == "silu" else "rmsnorm"
                fused[i] = ([i, cons[0]], tag)
            continue
        for start, end, tag in opaque_runs:
            if n.op == OPAQUE + start:
                stop = next(j for j in range(i + 1, len(graph.nodes))
                            if graph.nodes[j].op == OPAQUE + end)
                fused[i + 1] = (list(range(i + 1, stop)), tag)

    absorbed = {i for ids, _ in fused.values() for i in ids} - set(fused)
    groups = []
    for i, n in enumerate(graph.nodes):
        if i in absorbed:
            continue
        if i in fused:
            groups.append(make_group(graph, *fused[i]))
        elif is_opaque(n.op):
            groups.append(make_group(graph, [i], n.op.split(":", 1)[1]))
        else:
            groups.append(make_group(graph, [i], n.op))
    return groups


def check_fork_join(graph: ModelGraph, partition: list[Group]) -> Optional[str]:
    """MPK gives a task one trigger_event and one dependent_event."""
    succ = _edges(graph, partition)
    pred: dict[int, set[int]] = {gi: set() for gi in range(len(partition))}
    for u, cs in succ.items():
        for v in cs:
            pred[v].add(u)

    is_fork_producer = {gi: len(cs) > 1 for gi, cs in succ.items()}
    is_join_consumer = {gi: len(ps) > 1 for gi, ps in pred.items()}

    for gi, g in enumerate(partition):
        if is_join_consumer[gi] and any(is_fork_producer[p] for p in pred[gi]):
            return f"group {g.tag!r} is a join-consumer fed by a fork-producer (case 2)"
        if is_fork_producer[gi] and any(is_join_consumer[c] for c in succ[gi]):
            return f"group {g.tag!r} is a fork-producer feeding a join-consumer (case 3)"
    return None


def check_shapes(group: Group) -> Optional[str]:
    """What search and the Blackwell backend will refuse on shape alone."""
    if len(group.external_inputs) > MAX_GROUP_INPUTS:
        return (f"{len(group.external_inputs)} inputs, search fuses at most "
                f"{MAX_GROUP_INPUTS}")
    n = group.output.dims[-1]
    if n % 64:
        return f"output last dim {n} is not a multiple of 64"
    return None


def group_signature(graph: ModelGraph, group: Group) -> tuple:
    """What makes two groups the same search: ops and shapes, not node ids."""
    return (tuple(graph.nodes[i].op for i in group.nodes),
            tuple(v.dims for v in group.external_inputs),
            group.output.dims)


_PROBE_SRC = """
import json, sys
from mirage.mpk.lowering import task_search
from mirage.mpk.lowering.task_search import TaskSpec, TensorSpec

spec_dims = json.loads(sys.argv[1])
ops = json.loads(sys.argv[2])
attrs = json.loads(sys.argv[3])
grid = tuple(json.loads(sys.argv[4]))
out_dims = tuple(json.loads(sys.argv[5]))
forloop = json.loads(sys.argv[6])
max_ops = json.loads(sys.argv[7])

def build(kn, t):
    env = list(t)
    out = None
    for op, a in zip(ops, attrs):
        n = len(env)
        args = [env[i] for i in a["in"]]
        # JSON has no tuples; kn.rms_norm's normalized_shape must be one.
        kw = {k: tuple(v) if isinstance(v, list) else v
              for k, v in a.get("kw", {}).items()}
        out = getattr(kn, a["fn"])(*args, **kw)
        env.append(out)
    return out

spec = TaskSpec("probe", build, [TensorSpec(tuple(d)) for d in spec_dims])
# Ask EXACTLY what lower_group asks -- same grid, same forloop, same op
# budget. A probe that searches a different configuration answers a different
# question and calls groups buildable that the real lowering rejects.
try:
    scheds = task_search.search_task_schedules(
        spec, grid_dim=grid, forloop_range=forloop, max_ops=max_ops,
        wide_inputs=len(spec_dims) > 3)
except Exception:
    try:
        scheds = task_search.search_task_schedules(
            spec, grid_dim=grid, forloop_range=None, max_ops=max_ops,
            wide_inputs=len(spec_dims) > 3)
    except Exception as e:
        print(chr(10) + "@@PROBE@@ NO search " + type(e).__name__ + ": "
              + str(e)[:200].replace(chr(10), " "))
        sys.exit(0)

# Search returning a schedule is only the FIRST gate. The transpiler still has
# to emit a body for it, and it rejects some -- "an operand layout the MMA
# cannot read" -- at register_generated_task. A probe that stops at search
# calls those groups buildable and they are not, so go all the way to
# registration. Any schedule that registers makes the group buildable.
import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

nw, ns = mirage.get_configurations_from_gpu(0)
p = PersistentKernel.get_default_init_parameters()
p.update(test_mode=True, num_workers=nw, num_local_schedulers=ns, mpi_rank=0,
         world_size=1, max_num_batched_tokens=8, max_num_batched_requests=8)
last = None
for i, sched in enumerate(scheds):
    try:
        pk = PersistentKernel(**p)
        ins = [pk.new_tensor(dims=tuple(d), name="probe_in%d" % j)
               for j, d in enumerate(spec_dims)]
        out = pk.new_tensor(dims=out_dims, name="probe_out")
        task_search.register_searched_task(pk, sched, inputs=ins, output=out)
        print(chr(10) + "@@PROBE@@ OK")
        sys.exit(0)
    except Exception as e:
        last = type(e).__name__ + ": " + str(e)[:160].replace(chr(10), " ")
print(chr(10) + "@@PROBE@@ NO register " + str(last))
"""


class Schedulable:
    """Memoised 'can this group be BUILT?', probed out of process.

    Buildable means both gates pass: search returns a schedule, and the
    transpiler accepts it at registration. Checking only the first calls
    groups buildable that fail later with an MMA operand-layout error.
    """

    def __init__(self, graph: ModelGraph, grid_for=None, forloop_for=None,
                 verbose: bool = False,
                 timeout: int = 900, require_fused_only: bool = True):
        from .task_search import default_forloop, default_grid
        self.graph = graph
        self.grid_for = grid_for or default_grid
        self.forloop_for = forloop_for or default_forloop
        self.verbose = verbose
        self.timeout = timeout
        self.require_fused_only = require_fused_only
        self.cache: dict[tuple, Optional[str]] = {}

    def _probe(self, group: Group, grid) -> Optional[str]:
        import json as _json
        import subprocess
        import sys as _sys

        from .node import OPS

        slot = {v.name: i for i, v in enumerate(group.external_inputs)}
        ops, attrs = [], []
        for i in group.nodes:
            n = self.graph.nodes[i]
            attrs.append({"fn": n.op,
                          "in": [slot[v.name] for v in n.inputs],
                          "kw": {k: list(v) if isinstance(v, tuple) else v
                                 for k, v in n.attrs.items()}})
            ops.append(n.op)
            slot[n.output.name] = len(group.external_inputs) + len(ops) - 1

        args = [_json.dumps([list(v.dims) for v in group.external_inputs]),
                _json.dumps(ops), _json.dumps(attrs), _json.dumps(list(grid)),
                _json.dumps(list(group.output.dims)),
                _json.dumps(self.forloop_for(group)),
                _json.dumps(max(9, len(group.nodes)
                                + len(group.external_inputs) + 4))]
        try:
            proc = subprocess.run([_sys.executable, "-c", _PROBE_SRC] + args,
                                  capture_output=True, text=True,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return f"search timed out after {self.timeout}s"
        marker = "@@PROBE@@ "
        idx = proc.stdout.rfind(marker)
        line = (proc.stdout[idx + len(marker):].split("\n", 1)[0].strip()
                if idx >= 0 else None)
        if line is None:
            tail = (proc.stdout + proc.stderr).strip().splitlines()
            return ("search crashed: "
                    + (tail[-1][:120] if tail else f"rc={proc.returncode}"))
        return None if line == "OK" else line[3:]

    def __call__(self, group: Group) -> Optional[str]:
        if is_opaque(self.graph.nodes[group.nodes[0]].op):
            return None                      # a hand-written task, not searched
        if self.require_fused_only and len(group.nodes) == 1:
            return None                      # today's baseline; can fall back
        sig = group_signature(self.graph, group)
        if sig in self.cache:
            return self.cache[sig]

        why = check_shapes(group)
        if why is None:
            why = self._probe(group, tuple(self.grid_for(group)))
        if self.verbose:
            print(f"[partition] {group.tag}: "
                  f"{'ok' if why is None else 'no -- ' + why}", flush=True)
        self.cache[sig] = why
        return why


def feasible_partitions(
    graph: ModelGraph,
    node_ids: Optional[list[int]] = None,
    *,
    schedulable: Optional[Schedulable] = None,
    max_group_ops: int = MAX_GROUP_OPS,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> tuple[list[list[Group]], dict[str, int]]:
    """Candidates that survive every check, plus a tally of what killed the
    rest. Pass schedulable=None to skip the expensive stage."""
    stats = {"enumerated": 0, "fork_join": 0, "shapes": 0, "schedulable": 0,
             "kept": 0}
    kept = []
    for partition in enumerate_partitions(graph, node_ids,
                                          max_group_ops=max_group_ops):
        stats["enumerated"] += 1

        why = check_fork_join(graph, partition)
        if why:
            stats["fork_join"] += 1
            continue

        bad = next((check_shapes(g) for g in partition
                    if not is_opaque(graph.nodes[g.nodes[0]].op)
                    and check_shapes(g)), None)
        if bad:
            stats["shapes"] += 1
            continue

        if schedulable is not None:
            bad = next((schedulable(g) for g in partition if schedulable(g)), None)
            if bad:
                stats["schedulable"] += 1
                continue

        kept.append(partition)
        stats["kept"] += 1
        if limit and len(kept) >= limit:
            break
    if verbose:
        print(f"[partition] {stats}", flush=True)
    return kept, stats


def assign_buffers(graph: ModelGraph, partition: list[Group],
                   pinned: Optional[dict] = None,
                   reuse: bool = True) -> dict:
    """Map each group output to a buffer name, reusing buffers by liveness."""
    pinned = pinned or {}
    owner = {i: gi for gi, g in enumerate(partition) for i in g.nodes}

    # last group that reads each value
    last_read: dict[str, int] = {}
    for gi, g in enumerate(partition):
        for v in g.external_inputs:
            last_read[v.name] = max(last_read.get(v.name, -1), gi)
    for v in graph.outputs:
        last_read[v.name] = len(partition)          # lives to the end

    free: dict[tuple, list[str]] = {}
    busy: list[tuple[int, tuple, str]] = []          # (free_after, key, buf)
    assignment: dict[str, str] = {}
    n = 0

    for gi, g in enumerate(partition):
        # release buffers whose last reader has run
        for entry in [b for b in busy if b[0] < gi]:
            busy.remove(entry)
            free.setdefault(entry[1], []).append(entry[2])

        name = g.output.name

        if name in pinned:
            assignment[name] = name
            continue
        key = tuple(g.output.dims)
        # Reuse frees a buffer once the partition INDEX of its last reader has
        # passed, which assumes groups run in order. MPK runs a DAG, so two
        # groups sharing a recycled buffer may overlap. reuse=False pins one
        # buffer per group to test that.
        pool = free.get(key) if reuse else None
        if pool:
            buf = pool.pop()
        else:
            n += 1
            buf = f"mg_buf{n}_{'x'.join(str(d) for d in key)}"
        assignment[name] = buf
        busy.append((last_read.get(name, gi), key, buf))
    return assignment


def check_covers(graph: ModelGraph, partition: list[Group]) -> None:
    """Every node in exactly one group. A gap aborts inside MPK's lowering
    with no useful message, so catch it here."""
    seen: dict[int, str] = {}
    for g in partition:
        for i in g.nodes:
            if i in seen:
                raise ValueError(
                    f"node {i} ({graph.nodes[i]!r}) is in both {seen[i]!r} "
                    f"and {g.tag!r}")
            seen[i] = g.tag or str(g.nodes)
    missing = [i for i in range(len(graph)) if i not in seen]
    if missing:
        raise ValueError(
            f"partition does not cover {len(missing)} node(s): "
            + ", ".join(f"[{i}] {graph.nodes[i]!r}" for i in missing[:4]))
