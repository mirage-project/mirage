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
    pool, views = plan.allocate_pool(
        {"stream": [("k", entry_shape, torch.bfloat16),
                    ("v", entry_shape, torch.bfloat16)]}, max_num_pages)
    pk = PersistentKernel(
        kv_groups=plan.group_specs(),
        meta_tensors={**other_meta, **plan.build_meta_tensors(...)})
    for layer_id in range(num_layers):
        group_id, slot_id = plan.layer_info(layer_id)
        pk.attach_input(views[group_id]["k"][slot_id], ...)
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
    window_size: sliding-window length in raw tokens. The scheduler recycles
        pages that have fallen out of the window, so a window stream holds
        roughly the window instead of the whole sequence. Declaring it
        promises that every layer of this stream masks with that window.
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


# Tokens per KV tile in the windowed attention kernel (attention_sm100.cuh's
# KV_TILE_SIZE, mirrored by MPK_KV_WINDOW_TILE in runtime_header.h). A windowed
# task starts loading at a tile boundary, not at the exact window edge, so a
# page only counts as dead once it is entirely below that boundary.
KV_WINDOW_TILE = 64


@dataclass
class KVGroupSpec:
    """Per-group config consumed by PersistentKernel: the group's page table
    advances ``block_size`` raw tokens per page.

    ``window_size`` (0 = full attention) lets the scheduler recycle pages that
    have fallen out of the window mid-request. Declaring it is a PROMISE that
    every layer reading this group's page table masks with that same window —
    a full-attention layer on a windowed group would read a recycled page."""
    block_size: int
    window_size: int = 0


def pages_per_request(block_size: int, window_size: int, max_seq_length: int,
                      max_num_batched_tokens: int = 1) -> int:
    """Worst-case pages one request holds in a group at any single step.

    Without a window that is the whole sequence. With one, the scheduler
    returns the pages below the window's tile boundary, so what stays is the
    window plus the partial pages at each end.

    The two sides are evaluated at different points on purpose, matching the
    scheduler: pages are allocated for the batch's LAST token but recycled
    against its FIRST one, so a batch of ``max_num_batched_tokens`` can hold
    up to that much extra."""
    worst = 0
    for boundary in range(0, max_seq_length, block_size):
        for pos in (boundary, min(boundary + block_size - 1,
                                  max_seq_length - 1)):
            span = (pos + 1 + block_size - 1) // block_size
            freed = 0
            if window_size > 0:
                step = max(pos + 1 - max_num_batched_tokens, 0)
                live_from = max(step - window_size + 1, 0)
                freed = ((live_from // KV_WINDOW_TILE) * KV_WINDOW_TILE
                         // block_size)
            worst = max(worst, span - freed)
    return worst


@dataclass
class KVCachePlan:
    """Planner output: a prescription only. The builder allocates tensors
    (``allocate_pool``) and wires layers (``layer_info``); the plan itself
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
        window_size: int = 0     # 0 = full attention

    target_page_bytes: int
    num_slots: int
    groups: Tuple["KVCachePlan.Group", ...]
    # Set once by resolve_pool_size. The page tables and the pool must be
    # sized by the SAME number, and they are built in different places (the
    # demo builds the meta tensors, the model builder allocates the pool), so
    # recording it here lets both read it and lets a mismatch be caught.
    max_num_pages: Optional[int] = None

    # ── what PersistentKernel consumes ────────────────────────────────────

    def group_specs(self):
        """The kv_groups= argument for PersistentKernel."""
        return [KVGroupSpec(block_size=g.block_size, window_size=g.window_size)
                for g in self.groups]

    # ── sizing the pool ───────────────────────────────────────────────────

    @property
    def page_id_bytes(self) -> int:
        """Bytes one page id costs. Drawing it takes that page at every slot,
        not one layer's page."""
        return self.num_slots * self.target_page_bytes

    def pages_for_budget(self, budget_bytes: int) -> int:
        """How many page ids fit in a byte budget, rounded down."""
        assert budget_bytes >= 0
        return budget_bytes // self.page_id_bytes

    def _pool_pages(self, given: Optional[int]) -> int:
        """The page count to size a pool-shaped thing with. Accepts the
        caller's number, the one resolve_pool_size recorded, or both when they
        agree — a disagreement means the page tables and the pool would be
        sized differently, which only shows up as corruption at runtime."""
        if given is None and self.max_num_pages is None:
            raise ValueError(
                "no pool size: pass max_num_pages, or call resolve_pool_size "
                "on this plan first")
        if given is not None and self.max_num_pages is not None:
            if given != self.max_num_pages:
                raise ValueError(
                    f"pool size disagreement: {given} pages passed here but "
                    f"the plan was sized to {self.max_num_pages}")
        return given if given is not None else self.max_num_pages

    def budget_bytes(self, num_pages: int) -> int:
        """Bytes a pool of ``num_pages`` ids occupies — the inverse, for
        reporting a page count back to the user in the units they care
        about."""
        return num_pages * self.page_id_bytes

    def pages_needed(self, max_num_batched_requests: int, max_seq_length: int,
                     max_num_batched_tokens: int = 1) -> int:
        """Floor: page ids the batch holds at once in the worst case. Below
        this the free list wraps and re-hands a page that is still in use."""
        return max_num_batched_requests * sum(
            pages_per_request(g.block_size, g.window_size, max_seq_length,
                              max_num_batched_tokens)
            for g in self.groups)

    def pool_size(self, budget_bytes: int, max_num_batched_requests: int,
                  max_seq_length: int, max_num_batched_tokens: int = 1) -> int:
        """Page count for a byte budget, checked against the floor."""
        pages = self.pages_for_budget(budget_bytes)
        floor = self.pages_needed(max_num_batched_requests, max_seq_length,
                                  max_num_batched_tokens)
        if pages < floor:
            raise ValueError(
                f"KV budget too small: {format_bytes(budget_bytes)} gives {pages} "
                f"page(s), but {max_num_batched_requests} request(s) at "
                f"{max_seq_length} tokens need {floor} "
                f"({format_bytes(self.budget_bytes(floor))}). Raise the budget, raise "
                f"--page-size, or lower max_seq_length.")
        return pages


    def build_meta_tensors(self, max_num_pages: Optional[int] = None,
                           max_num_batched_requests: int = 1,
                           max_seq_length: Optional[int] = None,
                           dtype=torch.int32, device: str = "cuda"):
        """Page-table buffers (indptr / indices / last_page_len) for every
        group, keyed as PersistentKernel.__init__ expects. Merge into the
        meta_tensors dict BEFORE constructing PersistentKernel.

        The indices buffer is indexed by ABSOLUTE page number within a
        request, so it must cover the whole sequence even when a windowed
        group holds far fewer pages than that (recycled slots stay in place,
        holding -1). Pass ``max_seq_length`` to size it for that; without it
        the buffer only fits ``max_num_pages`` entries, which is enough
        exactly when no group recycles."""
        max_num_pages = self._pool_pages(max_num_pages)
        out = {}
        for g_id, g in enumerate(self.groups):
            span = max_num_pages
            if max_seq_length is not None:
                span = max(max_num_pages,
                           max_num_batched_requests
                           * ((max_seq_length + g.block_size - 1)
                              // g.block_size))
            out[f"paged_kv_indptr_buffer_{g_id}"] = torch.zeros(
                max_num_batched_requests + 1, dtype=dtype, device=device)
            out[f"paged_kv_indices_buffer_{g_id}"] = torch.zeros(
                span, dtype=dtype, device=device)
            out[f"paged_kv_last_page_len_buffer_{g_id}"] = torch.zeros(
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

    def allocate_pool(self, entry_layouts, max_num_pages: Optional[int] = None,
                      device: str = "cuda"):
        """The entire KV cache as ONE allocation, plus typed views.

        Physical layout: ``[num_slots, max_num_pages, target_page_bytes]``
        raw bytes. A page id from the unified free list denotes page ``p``
        of EVERY slot — that page at every layer — and only one group holds
        a given page id at a time, so the views never alias live data and
        no capacity is stranded.

        A stream may carve its page into several COMPONENTS (a GQA stream
        stores K and V; a latent stream stores one blob). Components are
        laid out component-major inside the page: all of the page's entries
        for component 0, then all for component 1, ... — so each component's
        inner shape is exactly what a per-layer cache tensor looks like
        today, and only the page stride differs.

        entry_layouts: ``{spec_name: [(component_name, entry_shape, dtype),
            ...]}``. The components' per-entry bytes must sum to at most the
            stream's per_entry_bytes.
        Returns ``(pool, views)``; ``views[group_id][component_name]`` is
            shaped ``[num_slots, max_num_pages, entries_per_page,
            *entry_shape]``. The views alias ``pool`` and keep its storage
            alive on their own; ``pool`` is returned for accounting.

        EVERY view has page stride == target_page_bytes, which is >= its
        packed entry span whenever the stream has more than one component or
        any intra-page padding. Kernels must therefore address a page by its
        stride and never assume pages sit back to back — see
        ``page_stride_elems``."""
        max_num_pages = self._pool_pages(max_num_pages)
        pool = torch.zeros(self.num_slots, max_num_pages,
                           self.target_page_bytes, dtype=torch.uint8,
                           device=device)
        views = {}
        for g in self.groups:
            if g.spec_name not in entry_layouts:
                raise KeyError(
                    f"no entry layout given for stream '{g.spec_name}'")
            byte_off = 0
            comps = {}
            for cname, entry_shape, dtype in entry_layouts[g.spec_name]:
                entry_elems = 1
                for d in entry_shape:
                    entry_elems *= d
                itemsize = torch.empty(0, dtype=dtype).element_size()
                assert self.target_page_bytes % itemsize == 0, (
                    f"page of {self.target_page_bytes} B does not divide "
                    f"into {itemsize} B elements ('{g.spec_name}.{cname}')")
                assert byte_off % itemsize == 0, (
                    f"component '{g.spec_name}.{cname}' starts at byte "
                    f"{byte_off}, not a multiple of its {itemsize} B element")
                span = g.entries_per_page * entry_elems
                byte_end = byte_off + span * itemsize
                assert byte_end <= self.target_page_bytes, (
                    f"stream '{g.spec_name}' components exceed the "
                    f"{self.target_page_bytes} B page at '{cname}' "
                    f"({byte_end} B)")
                elem_off = byte_off // itemsize
                comps[cname] = pool.view(dtype)[
                    ..., elem_off:elem_off + span].view(
                    self.num_slots, max_num_pages, g.entries_per_page,
                    *entry_shape)
                byte_off = byte_end
            views[g.group_id] = comps
        return pool, views

    def page_stride_elems(self, dtype) -> int:
        """Elements between consecutive pages of a pool view of ``dtype`` —
        what a kernel must multiply a page id by. Always the full page, not
        the view's packed entry span."""
        itemsize = torch.empty(0, dtype=dtype).element_size()
        assert self.target_page_bytes % itemsize == 0
        return self.target_page_bytes // itemsize


# ── planner ───────────────────────────────────────────────────────────────


def format_bytes(nbytes: int) -> str:
    """Human-readable byte count, so page counts and budgets can be reported
    in the same units the user typed."""
    for unit, scale in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if nbytes >= scale:
            return f"{nbytes / scale:.2f} {unit}"
    return f"{nbytes} B"


def resolve_kv_budget(spec, device: int = 0) -> int:
    """Turn a user-facing budget into bytes.

    Accepts a raw byte count(``"24GiB"`` / ``"512MiB"`` / ``12345678``), or
     a fraction of the device's TOTAL memory (``0.6``, or ``"60%"``).

    The fraction is of TOTAL, not free. Whether the budget actually fits is
    a separate check at allocation time."""
    if isinstance(spec, (int,)) and not isinstance(spec, bool):
        return int(spec)
    if isinstance(spec, float):
        frac = spec
    else:
        text = str(spec).strip()
        units = {"KIB": 1024, "MIB": 1024**2, "GIB": 1024**3, "TIB": 1024**4,
                 "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4,
                 "B": 1}
        upper = text.upper()
        for suffix, scale in sorted(units.items(), key=lambda kv: -len(kv[0])):
            if upper.endswith(suffix):
                return int(float(text[:-len(suffix)]) * scale)
        frac = float(text[:-1]) / 100 if upper.endswith("%") else float(text)
    if not 0 < frac <= 1:
        raise ValueError(
            f"KV budget fraction must be in (0, 1], got {frac}. Use a suffix "
            f"like '24GiB' for an absolute size.")
    total = torch.cuda.get_device_properties(device).total_memory
    return int(total * frac)


def resolve_pool_size(plan: "KVCachePlan", *, kv_budget=None,
                      max_num_pages: Optional[int] = None,
                      max_seq_length: int,
                      max_num_batched_requests: int = 1,
                      max_num_batched_tokens: int = 1,
                      device: int = 0, verbose: bool = True) -> int:
    """The page count to build the pool with, from a byte budget or an
    explicit count. Every demo needs this, so it lives here rather than being
    re-derived per model.

    Exactly one of ``kv_budget`` / ``max_num_pages`` may be given. A byte
    budget is the better knob: a page id costs slots x page_bytes, so the same
    page COUNT is 24 MiB at one block size and 1536 MiB at another. The count
    is kept for tests that need a fixed, machine-independent number.

    Raises ValueError with both units when the pool would be below what the
    batch needs, or larger than what is free right now."""
    if (kv_budget is None) == (max_num_pages is None):
        raise ValueError("give exactly one of kv_budget / max_num_pages")

    if max_num_pages is not None:
        pages = max_num_pages
        floor = plan.pages_needed(max_num_batched_requests, max_seq_length,
                                  max_num_batched_tokens)
        if pages < floor:
            raise ValueError(
                f"max_num_pages {pages} is below the {floor} this batch needs "
                f"({format_bytes(plan.budget_bytes(floor))})")
        source = "explicit page count"
    else:
        budget = resolve_kv_budget(kv_budget, device)
        pages = plan.pool_size(budget, max_num_batched_requests,
                               max_seq_length, max_num_batched_tokens)
        source = f"budget {kv_budget}"

    plan.max_num_pages = pages          # both sizing sites read it from here
    used = plan.budget_bytes(pages)
    free, total = torch.cuda.mem_get_info(device)
    if verbose:
        print(f"KV pool: {pages} pages x "
              f"{format_bytes(plan.page_id_bytes)} = {format_bytes(used)}  "
              f"({source}; device has {format_bytes(free)} free of "
              f"{format_bytes(total)})")
    if used > free:
        raise ValueError(
            f"the KV pool alone ({format_bytes(used)}) exceeds free memory "
            f"({format_bytes(free)}), before the model weights")
    return pages


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
        # TODO(padding): this maximises block sizes but ignores how evenly
        # each stream's entries tile the page, so a stream whose entry size
        # does not divide the target pays for it. Padding is already
        # measured per group, so a later pass can score several candidate
        # page sizes and pick the one with the least total padding.

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
                window_size=s.window_size or 0,
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
