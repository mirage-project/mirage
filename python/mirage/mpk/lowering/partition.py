"""Choose where the task boundaries go, instead of writing them by hand.

Today the five MPK_COMPILED_MLP_IMPL variants ARE the partition space: someone
wrote each one. The spread between them is large -- measured on Qwen3-0.6B
decode, up+SiLU fusion ran 12.3% better than three-task and SiLU+down 26.5%
worse -- so it is worth searching rather than guessing.

Enumeration is over contiguous runs of the topological order. That is not
every set partition, but it is every partition whose groups are convex, which
is the only kind that lowers: a group with a node outside it on a path between
two nodes inside would need its own input to come from its own output.

The expensive part is asking search whether a group is schedulable at all.
Candidates share groups heavily -- most differ in one cut -- so that answer is
memoised per group signature, which is what makes the space tractable.
"""
from __future__ import annotations

import dataclasses
import itertools
from typing import Iterator, Optional

from .group import Group, make_group
from .node import ModelGraph, is_opaque


# A group of this many muGraph ops is already past what search will schedule
# (max_num_threadblock_graph_op is 9 and counts TB_INPUT_OPs), so do not
# enumerate beyond it.
MAX_GROUP_OPS = 6

# task_search pins block_dim, and search's own cap on how many tensors one
# threadblock graph may read.
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
    """Candidate partitions of `node_ids` (default: the whole graph).

    An opaque node cannot be fused, so it forces a cut on both sides.
    """
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
            # More than one live output, or an opaque node grouped with
            # something. Not a legal task -- skip the candidate.
            continue


def _tag(graph: ModelGraph, ids) -> str:
    ops = [graph.nodes[i].op for i in ids]
    if len(ops) == 1 and is_opaque(ops[0]):
        return ops[0].split(":", 1)[1]
    return "_".join(ops)


# ---------------------------------------------------------------------------
# Feasibility, cheapest test first.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Rejection:
    stage: str
    reason: str


def _edges(graph: ModelGraph, partition: list[Group]) -> dict[int, set[int]]:
    """Producer group -> consumer groups, after residual stripping.

    build_annotated_graph drops a direct edge u->v when a path u->...->v of
    length >= 2 also exists (step (c)); the residual add in a transformer layer
    is exactly that shape. Skipping this step makes every partition look like a
    join+fork violation, including the one MPK runs today.
    """
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


def check_fork_join(graph: ModelGraph, partition: list[Group]) -> Optional[str]:
    """MPK gives a task one trigger_event and one dependent_event.

    build_annotated_graph rejects a layer that is both a join-consumer and a
    fork-consumer -- "a task cannot have two trigger_events" -- and the mirror
    case on the producer side (annotated_graph.cc:600-678). Pure graph shape,
    so it costs nothing and runs first.
    """
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
    """What makes two groups the same question for search. Deliberately not
    the node ids: the same op sequence on the same shapes in a different layer
    is the same search, and Qwen3 has 28 identical layers."""
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
try:
    task_search.search_task_schedules(spec, grid_dim=grid)
    print(chr(10) + "@@PROBE@@ OK")
except Exception as e:
    print(chr(10) + "@@PROBE@@ NO " + type(e).__name__ + ": "
          + str(e)[:200].replace(chr(10), " "))
"""


class Schedulable:
    """Memoised 'can search schedule this group at all?', probed out of process.

    Out of process because search can ABORT rather than raise: some group
    shapes make it throw a nlohmann::json type_error, which reaches
    std::terminate and takes the interpreter with it. In-process probing meant
    one bad shape killed a whole filter run. A crashed probe is simply recorded
    as unschedulable.

    Memoised per signature, not per group: Qwen3 has 28 identical layers and
    most candidates differ by one cut, so a 160-candidate space asks search a
    few dozen distinct questions.
    """

    def __init__(self, graph: ModelGraph, grid_for=None, verbose: bool = False,
                 timeout: int = 900, require_fused_only: bool = True):
        from .task_search import default_grid
        self.graph = graph
        self.grid_for = grid_for or default_grid
        self.verbose = verbose
        self.timeout = timeout
        # A single-node group is what MPK runs today, as a hand-written task,
        # so it is lowerable whether or not search can schedule it -- lower()
        # falls back. Only a FUSED group has to be searchable, because fusing
        # is the thing being proposed. Set False to demand search for every
        # group, which is stricter than the baseline.
        self.require_fused_only = require_fused_only
        self.cache: dict[tuple, Optional[str]] = {}

    def _probe(self, group: Group, grid) -> Optional[str]:
        import json as _json
        import subprocess
        import sys as _sys

        from .node import OPS

        # Describe the group so the child can rebuild the build lambda without
        # importing this graph: op fn names, and which env slots feed each.
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
                _json.dumps(ops), _json.dumps(attrs), _json.dumps(list(grid))]
        try:
            proc = subprocess.run([_sys.executable, "-c", _PROBE_SRC] + args,
                                  capture_output=True, text=True,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return f"search timed out after {self.timeout}s"
        # search prints progress with no trailing newline, so the marker can
        # land mid-line -- find it anywhere, not at a line start.
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
                   pinned: Optional[dict] = None, alias=None) -> dict:
    """Map each group output to a buffer name, reusing buffers by liveness.

    MPK has no intermediate DTensors, so every boundary needs a pre-allocated
    tensor. Allocating one per boundary would grow with depth -- 28 layers x
    several boundaries -- which is why Qwen3Builder pre-allocates rmsnorm_out /
    attn_in / mlp_mid ONCE and rewrites them every layer
    (builder.py:114-250). This reproduces that automatically.

    Safe because build_annotated_graph binds a read to whichever producer most
    recently wrote the buffer (annotated_graph.cc:144-158): as long as
    registration follows topological order and a buffer is only reused after
    its last reader, the most-recent writer is the right one.

    An ALIASED group writes in place into one of its inputs, so its output
    physically lives in THAT buffer. Two things follow, and getting either
    wrong produces a model that runs and computes nonsense: the output must be
    assigned the same buffer, and that buffer must stay live until the
    OUTPUT's last reader -- not the input's. Releasing it at the aliased group
    hands a still-live tensor to a later group.

    Returns value name -> buffer name. Pinned values keep their own buffer.
    """
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
        ali = alias(g) if alias is not None else None
        if ali is not None and name not in pinned:
            src = g.external_inputs[ali].name
            buf = assignment.get(src, src)
            assignment[name] = buf
            keep = last_read.get(name, gi)
            for k, entry in enumerate(busy):
                if entry[2] == buf:
                    busy[k] = (max(entry[0], keep), entry[1], buf)
                    break
            else:
                busy.append((keep, tuple(g.output.dims), buf))
            continue

        if name in pinned:
            assignment[name] = name
            continue
        key = tuple(g.output.dims)
        pool = free.get(key)
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
