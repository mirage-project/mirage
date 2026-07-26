"""KV cache planning for models with more than one kind of KV state.

A hybrid-attention model stores several KV *streams* — e.g. full attention,
sliding-window, compressed latent — whose per-token byte costs differ. MPK
serves all of them from ONE physical page pool: every page is the same number
of bytes and is handed out at runtime from a single free list, so capacity
moves freely between streams instead of being statically partitioned.

Vocabulary:

- stream (``KVSpec``): one kind of KV state, declared by the model builder.
- page: one fixed-size row of physical memory (``target_page_bytes``).
- block_size: raw tokens one page holds for a given stream. Streams with
  smaller or compressed entries pack more tokens per page, so each stream
  consumes pages at its own rate.
- group: one page table. Layers that advance through pages in lockstep share
  a group; a stream with many layers is split into several same-shaped groups.
- slot: index of one physical tensor. Groups are chunked to a common size so
  that layer i of every group can share the tensor at slot i (pages are
  byte-identical across groups, which makes the sharing safe).

Typical flow:

    plan = plan_kv_groups([KVSpec(...), KVSpec(...)])
    cache = plan.allocate_slots(entry_shape, max_num_pages)
    pk = PersistentKernel(
        kv_groups=plan.group_specs(),
        meta_tensors={**other_meta, **plan.build_meta_tensors(...)})
    for layer_id in range(num_layers):
        group_id, slot_id = plan.layer_info(layer_id)
        pk.attach_input(cache[slot_id], ...)
        # pass group_id to the layer
"""

from dataclasses import dataclass
from functools import reduce
from math import gcd
from typing import Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class KVSpec:
    """One KV stream declared by a model builder.

    per_entry_bytes: bytes of one stored entry (one raw token when
        uncompressed; one compressed record otherwise).
    layer_ids: layers carrying this stream. A layer may appear in several
        specs (e.g. a main cache plus an indexer cache on the same layer).
    compress_ratio: raw tokens folded into one stored entry.
    window_size: sliding-window length in raw tokens. Reserved: the
        allocator currently grows window streams like full streams; only
        attention masking consumes this today.
    block_size_multiple_of: kernel constraint on ENTRIES per page (e.g. a
        TMA tile multiple).
    preferred_block_size: this stream's natural block size in raw tokens.
        The largest resulting page across specs becomes the shared page size.
    """
    name: str
    per_entry_bytes: int
    layer_ids: Tuple[int, ...]
    compress_ratio: int = 1
    window_size: Optional[int] = None
    block_size_multiple_of: int = 1
    preferred_block_size: Optional[int] = None

    def __post_init__(self):
        assert self.per_entry_bytes > 0 and self.compress_ratio >= 1
        assert len(self.layer_ids) > 0
        assert len(set(self.layer_ids)) == len(self.layer_ids), \
            f"spec {self.name}: duplicate layer ids"
        if self.preferred_block_size is not None:
            assert self.preferred_block_size % self.compress_ratio == 0, (
                f"spec {self.name}: preferred_block_size must be a multiple "
                f"of compress_ratio")


class KVUnificationError(Exception):
    """A stream cannot fit even one entry into the shared page size, so a
    single-page-size plan is impossible. Raised loudly rather than silently
    degrading; multi-bucket planning (several page sizes) is future work."""


@dataclass
class KVGroupSpec:
    """Per-group config consumed by PersistentKernel: the group's page table
    advances ``block_size`` raw tokens per page."""
    block_size: int


@dataclass
class KVCachePlan:
    """Planner output: a prescription only. The builder allocates tensors
    (``allocate_slots``) and wires layers (``layer_info``); the plan itself
    holds no GPU state."""

    @dataclass
    class Group:
        """One page table: a chunk of one stream's layers, padded with None
        up to the plan's slot count."""
        group_id: int
        spec_name: str
        layer_ids: Tuple[Optional[int], ...]
        block_size: int          # raw tokens per page
        entries_per_page: int    # = block_size / compress_ratio
        padding_bytes_per_page: int

    target_page_bytes: int
    num_slots: int
    groups: Tuple["KVCachePlan.Group", ...]

    # ── what PersistentKernel consumes ────────────────────────────────────

    def group_specs(self):
        """The kv_groups= argument for PersistentKernel."""
        return [KVGroupSpec(block_size=g.block_size) for g in self.groups]

    def build_meta_tensors(self, max_num_pages: int,
                           max_num_batched_requests: int,
                           dtype=torch.int32, device: str = "cuda"):
        """Page-table buffers (indptr / indices / last_page_len) for every
        group, keyed as PersistentKernel.__init__ expects. Merge into the
        meta_tensors dict BEFORE constructing PersistentKernel."""
        out = {}
        for g in range(len(self.groups)):
            out[f"paged_kv_indptr_buffer_{g}"] = torch.zeros(
                max_num_batched_requests + 1, dtype=dtype, device=device)
            out[f"paged_kv_indices_buffer_{g}"] = torch.zeros(
                max_num_pages, dtype=dtype, device=device)
            out[f"paged_kv_last_page_len_buffer_{g}"] = torch.zeros(
                max_num_batched_requests, dtype=dtype, device=device)
        return out

    # ── per-layer wiring ──────────────────────────────────────────────────

    def layer_info(self, layer_id: int) -> Tuple[int, int]:
        """(group_id, slot_id) for one model layer."""
        for g in self.groups:
            if layer_id in g.layer_ids:
                return g.group_id, g.layer_ids.index(layer_id)
        raise KeyError(f"layer {layer_id} not covered by any group")

    def layer_assignments(self, num_layers: Optional[int] = None):
        """[(group_id, slot_id), ...] indexed by layer_id; asserts every
        layer below num_layers is covered."""
        if num_layers is None:
            real_ids = [lid for g in self.groups for lid in g.layer_ids
                       if lid is not None]
            num_layers = max(real_ids) + 1
        out = [None] * num_layers
        for g in self.groups:
            for slot_id, lid in enumerate(g.layer_ids):
                if lid is not None:
                    out[lid] = (g.group_id, slot_id)
        missing = [i for i, v in enumerate(out) if v is None]
        assert not missing, f"layers {missing} not covered by any group"
        return out

    def layer_group_ids(self, num_layers: Optional[int] = None):
        """[group_id, ...] indexed by layer_id — enough when each layer has
        a dedicated tensor and only the group routing matters."""
        return [g for g, _slot in self.layer_assignments(num_layers)]

    def slot_assignment(self):
        """slot index -> [(group_id, layer_id), ...] sharing that slot."""
        return [
            [(g.group_id, g.layer_ids[s]) for g in self.groups
             if g.layer_ids[s] is not None]
            for s in range(self.num_slots)
        ]

    # ── allocation ────────────────────────────────────────────────────────

    def allocate_slots(self, entry_shape: Sequence[int], max_num_pages: int,
                       dtype=torch.bfloat16, device: str = "cuda"):
        """One stacked tensor [num_slots, max_num_pages, block_size,
        *entry_shape]; the builder attaches slice [slot_id] per layer. Slot
        sharing needs no special mode — two groups' layers simply slice the
        same slot_id. Requires a uniform block_size across groups (always
        true for a single-page-size plan)."""
        block_sizes = {g.block_size for g in self.groups}
        assert len(block_sizes) == 1, (
            f"allocate_slots needs one uniform block_size, got {block_sizes}")
        (block_size,) = block_sizes
        return torch.zeros(self.num_slots, max_num_pages, block_size,
                           *entry_shape, dtype=dtype, device=device)


# ── planner ───────────────────────────────────────────────────────────────


def plan_kv_groups(
    specs,
    target_page_bytes: Optional[int] = None,
    default_block_size: int = 64,
) -> KVCachePlan:
    """Turn KVSpec declarations into a KVCachePlan.

    1. Pick the shared page size: by default the largest natural page across
       specs (preferred_block_size worth of entries), so the densest stream
       keeps its preferred block size.
    2. For every stream, pack as many entries per page as fit (floor,
       honoring block_size_multiple_of; leftover bytes are bounded
       intra-page padding). A stream whose single entry does not fit raises
       KVUnificationError.
    3. Chunk each stream's layers into groups of ``_group_size`` layers so
       all groups share one slot layout.
    """
    specs = list(specs)
    assert specs, "need at least one KVSpec"
    names = [s.name for s in specs]
    assert len(set(names)) == len(names), f"duplicate spec names: {names}"

    if target_page_bytes is None:
        target_page_bytes = max(
            ((s.preferred_block_size or default_block_size)
             // s.compress_ratio) * s.per_entry_bytes
            for s in specs)

    per_spec = {s.name: _fit_block_size(s, target_page_bytes) for s in specs}
    group_size = _group_size([len(s.layer_ids) for s in specs])

    groups = []
    for s in specs:
        block_size, entries, padding = per_spec[s.name]
        layers = list(s.layer_ids)
        for start in range(0, len(layers), group_size):
            chunk = layers[start:start + group_size]
            chunk += [None] * (group_size - len(chunk))
            groups.append(KVCachePlan.Group(
                group_id=len(groups),
                spec_name=s.name,
                layer_ids=tuple(chunk),
                block_size=block_size,
                entries_per_page=entries,
                padding_bytes_per_page=padding,
            ))

    return KVCachePlan(
        target_page_bytes=target_page_bytes,
        num_slots=group_size,
        groups=tuple(groups),
    )


def plan_uniform_kv_groups(num_layers: int, per_entry_bytes: int,
                          preferred_block_size: int) -> KVCachePlan:
    """Single-stream shorthand: one spec covering every layer. Model
    constructors use this when the caller passes no explicit plan."""
    spec = KVSpec("uniform", per_entry_bytes=per_entry_bytes,
                 layer_ids=tuple(range(num_layers)),
                 preferred_block_size=preferred_block_size)
    return plan_kv_groups([spec])


def _fit_block_size(spec: KVSpec, target_page_bytes: int):
    """Largest entry count fitting one page, floored to the kernel's entry
    multiple. Returns (block_size_tokens, entries, padding_bytes)."""
    entries = target_page_bytes // spec.per_entry_bytes
    entries -= entries % spec.block_size_multiple_of
    if entries <= 0:
        raise KVUnificationError(
            f"spec {spec.name}: one entry ({spec.per_entry_bytes} B, "
            f"multiple_of {spec.block_size_multiple_of}) does not fit a "
            f"{target_page_bytes} B page")
    block_size = entries * spec.compress_ratio
    padding = target_page_bytes - entries * spec.per_entry_bytes
    return block_size, entries, padding


def _group_size(layer_counts):
    """Slots per group. A group with k real layers padded to S slots strands
    (S-k)/S of every page it holds, so:

    - near-equal counts (hi < 1.5 * lo): pad the smaller stream up — bounded
      waste, fewest groups;
    - otherwise, a usable gcd (>= lo/2): split with zero padding;
    - degenerate gcd: fall back to the smallest count (only the ragged last
      chunk gets padded)."""
    lo, hi = min(layer_counts), max(layer_counts)
    if hi < lo * 1.5:
        return hi
    g = reduce(gcd, layer_counts)
    if g == lo or (g > 1 and g >= lo // 2):
        return g
    return lo


# ── debug ─────────────────────────────────────────────────────────────────


class KVEventLog:
    """Record-and-verify instrumentation for the runtime page allocator.

    Constructing one wires a ``kv_event_log`` meta tensor into the kernel
    (before pk.compile(); the scheduler then logs every page ALLOC/FREE 
    plus one ITER marker per scheduling pass). After the kernel ran, 
    ``verify()`` replays the log and asserts allocator invariants.

    Log format: log[0] = event count; event i is 4 ints at [4i+1 .. 4i+4] =
    (type, group_id, request_slot, page_id), type 1=ALLOC, 2=FREE, 3=ITER.
    """

    def __init__(self, pk, plan: KVCachePlan, capacity: int = 65536,
                 device: str = "cuda"):
        self.num_groups = len(plan.groups)
        self.log = torch.zeros(capacity, dtype=torch.int32, device=device)
        pk.meta_tensors["kv_event_log"] = self.log

    def verify(self):
        """Replay the log; assert no double-alloc, no free of an unowned
        page, and zero pages still live at the end. Returns
        {"iterations": int, "per_group": [{"allocs": int, "frees": int}]}."""
        return self.replay(self.log, self.num_groups)

    @staticmethod
    def replay(log: torch.Tensor, num_groups: int):
        """Standalone replay of a raw log tensor (see class docstring for
        the format); same checks and return value as verify()."""
        events = log.cpu().tolist()
        count = events[0]
        live = [set() for _ in range(num_groups)]
        stats = [{"allocs": 0, "frees": 0} for _ in range(num_groups)]
        iterations = 0
        for i in range(count):
            etype, g, _req, page = events[4 * i + 1: 4 * i + 5]
            if etype == 3:
                iterations += 1
                continue
            assert 0 <= g < num_groups, f"event {i}: group_id {g} out of range"
            if etype == 1:
                assert page not in live[g], (
                    f"group {g}: page {page} allocated twice with no free "
                    "in between")
                live[g].add(page)
                stats[g]["allocs"] += 1
            elif etype == 2:
                assert page in live[g], (
                    f"group {g}: page {page} freed but was never allocated "
                    "(or already freed)")
                live[g].discard(page)
                stats[g]["frees"] += 1
            else:
                raise ValueError(f"event {i}: unknown event type {etype}")
        leaked = {g: sorted(pages) for g, pages in enumerate(live) if pages}
        assert not leaked, f"pages leaked at end of log: {leaked}"
        return {"iterations": iterations, "per_group": stats}
