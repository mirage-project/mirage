"""KV cache planner: hybrid KV streams share one pool, one physical page size.

Pool shape: ``[num_slots, max_num_pages, target_page_bytes]``.

- stream (KVSpec): one type of cache element, declared by the model builder.
- page: one row of physical memory.
- block_size: raw tokens one page holds for a given stream.
- group: partitioned (by layers) or grouped KVSpec that share one page.
- slot: index of one physical tensor. Layer i of every group shares the
  tensor at slot i.

Usage:
    plan = plan_kv_groups([KVSpec(...), KVSpec(...)])
    pool, views = plan.allocate_pool(entry_layouts, max_num_pages)
    pk = PersistentKernel(kv_groups=plan.group_specs(),
                          meta_tensors={**plan.build_meta_tensors(...)})
    group_id, slot_id = plan.layer_info(layer_id)
"""

import warnings
from dataclasses import dataclass, fields, replace
from enum import Enum
from functools import reduce
from math import gcd
from typing import Optional, Sequence, Tuple

import torch


class KVKind(Enum):
    """How the planner treats a stream. The planner branches on this and on
    nothing else -- a stream's NAME never changes how it is planned."""
    PAGED = "paged"       # grows with tokens, gets a group and a page table
    FLAT = "flat"         # read as one [capacity, width] array, no page table
    BOUNDED = "bounded"   # fixed one entry per request, ever -- still gets a
                          # group and shares the pool's free list, but never
                          # more than 1 entry/page regardless of what the
                          # shared page size turns out to be


@dataclass(frozen=True)
class KVSpec:
    """One KV stream, in the planner's own vocabulary (see ``KVStream`` for
    the model-builder-facing form this is normally derived from).

    per_entry_bytes: bytes of one stored entry.
    layer_ids: layers carrying this stream.
    compress_ratio: raw tokens folded into one stored entry. For a bounded
        spec, the whole run folds into its one entry.
    window_size: sliding-window length in raw tokens.
    block_size_multiple_of: restriction on block size, in raw tokens (e.g.
        the attention kernel's KV tile). None takes default_kv_tile().
    bounded: never more than 1 entry, regardless of the shared page size.
        See ``KVKind.BOUNDED``.
    """
    name: str
    per_entry_bytes: int
    layer_ids: Tuple[int, ...]
    compress_ratio: int = 1
    window_size: Optional[int] = None
    block_size_multiple_of: Optional[int] = None
    bounded: bool = False

    def __post_init__(self):
        assert self.per_entry_bytes > 0 and self.compress_ratio >= 1
        assert len(self.layer_ids) > 0
        assert len(set(self.layer_ids)) == len(self.layer_ids), \
            f"spec {self.name}: duplicate layer ids"
        assert not (self.bounded and self.window_size), (
            f"spec {self.name}: bounded and window_size are exclusive -- a "
            f"fixed one-entry state is never recycled by a sliding window")


def _itemsize(dtype) -> int:
    return torch.empty(0, dtype=dtype).element_size()


@dataclass(frozen=True)
class KVStream:
    """One KV stream: what a page holds, for which layers. The declaration
    surface a model builder uses; ``build_kv_cache`` turns it into a
    ``KVSpec`` (paged) or ``FlatStream`` (unpaged).

    ``components`` is the per-token payload -- for GQA the K and V halves,
    each ``(entry_name, entry_shape, dtype)``. ``per_entry_bytes`` derives
    from it.

    ``kind=KVKind.FLAT`` means the reader addresses the cache as one flat
    ``[capacity, width]`` array: no group, no page table, no say in the
    shared page size. The plan still owns and budgets its storage. See
    ``FlatStream``.

    A layer carrying several streams gets each one's kwargs namespaced by its
    ``name`` (``**kv_swa`` supplies ``swa_k_cache``, ...) -- names are
    required unique, so this cannot collide; a layer with one stream gets the
    bare ``k_cache``/``group_id``.
    """
    name: str
    layers: Tuple[int, ...]
    components: Tuple[Tuple[str, Tuple[int, ...], "torch.dtype"], ...]
    window: int = 0
    compress_ratio: int = 1
    block_size_multiple_of: Optional[int] = None
    kind: KVKind = KVKind.PAGED

    @property
    def paged(self) -> bool:
        """Not FLAT: gets a group, a page table, a slot in the shared pool."""
        return self.kind is not KVKind.FLAT

    @property
    def per_entry_bytes(self) -> int:
        return sum(reduce(lambda a, b: a * b, shape, 1) * _itemsize(dtype)
                   for _, shape, dtype in self.components)

    def _check_components(self):
        assert self.components, f"stream {self.name}: no components declared"
        names = [c[0] for c in self.components]
        assert len(set(names)) == len(names), (
            f"stream {self.name}: duplicate component names {names}")

    def _flat(self, capacity: int) -> "FlatStream":
        """The unpaged form. Every paging knob must be at its default, since
        accepting one on an unpaged stream would silently ignore it."""
        self._check_components()
        for field_name, value in (("window", self.window),
                                  ("compress_ratio", self.compress_ratio),
                                  ("block_size_multiple_of",
                                   self.block_size_multiple_of)):
            default = KVStream.__dataclass_fields__[field_name].default
            if value != default:
                raise ValueError(
                    f"stream '{self.name}' is kind=KVKind.FLAT but sets "
                    f"{field_name}={value!r}; that describes a page, and an "
                    f"unpaged stream has none.")
        return FlatStream(name=self.name, layers=tuple(self.layers),
                          components=tuple(tuple(c) for c in self.components),
                          capacity=capacity)

    def _spec(self) -> KVSpec:
        assert self.paged, f"stream {self.name}: not paged, use _flat()"
        self._check_components()
        if self.kind is KVKind.BOUNDED and self.window:
            raise ValueError(
                f"stream '{self.name}' is kind=KVKind.BOUNDED but sets "
                f"window={self.window!r}; a fixed one-entry state is never "
                f"recycled by a sliding window.")
        return KVSpec(name=self.name,
                      per_entry_bytes=self.per_entry_bytes,
                      layer_ids=tuple(self.layers),
                      compress_ratio=self.compress_ratio,
                      window_size=self.window or None,
                      block_size_multiple_of=self.block_size_multiple_of,
                      bounded=self.kind is KVKind.BOUNDED)


def _hashable(v):
    """A dict key from a declared field, whether it came in as a list or a
    tuple. ``components`` is routinely a list, and shapes inside it tuples."""
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    return v


def _merge_identical_streams(streams):
    """Fold streams that would lay a page out identically into one stream.

    A group IS a page table, and the pool carries one slot count per group,
    so a stream with very few layers drags that count down for everyone it
    stays apart from (DeepSeek-V3's 61 attention + 1 MTP layer: gcd(61, 1)=1,
    so unmerged this is 62 single-slot groups instead of one group of 62).

    Key = every KVStream field describing the PAGE, read off the dataclass so
    a field added later is included by default. ``components`` must be in the
    key, otherwise two streams can agree on ``per_entry_bytes`` while laid out
    differently. Two streams with the same key covering the same layer are
    refused rather than merged.

    Returns ``(merged_streams, {declared name: merged name})``; the map is how
    ``attach`` finds the group a declared stream ended up in.
    """
    key_fields = [f.name for f in fields(KVStream)
                  if f.name not in ("name", "layers")]
    order, folded = [], {}
    for s in streams:
        key = tuple(_hashable(getattr(s, f)) for f in key_fields)
        if key not in folded:
            order.append(key)
            folded[key] = [s, [], []]
        exemplar, names, layers = folded[key]
        overlap = set(layers) & set(s.layers)
        if overlap:
            raise ValueError(
                f"streams {names + [s.name]!r} declare the identical page "
                f"layout and all cover layer(s) {sorted(overlap)}.")
        names.append(s.name)
        layers.extend(s.layers)

    out, merged_name = [], {}
    for key in order:
        exemplar, names, layers = folded[key]
        if len(names) == 1:
            out.append(exemplar)
            merged_name[names[0]] = exemplar.name
            continue
        fused = replace(exemplar, name="+".join(names),
                        layers=tuple(sorted(layers)))
        out.append(fused)
        for n in names:
            merged_name[n] = fused.name
    return out, merged_name


@dataclass(frozen=True)
class FlatStream:
    """A stream whose reader wants one contiguous ``[capacity, width]`` array.

    Outside the pool by necessity: a constant-stride reader cannot be handed
    its storage a page at a time from a shared free list. Still inside the
    plan, which allocates it, budgets it, and is the only way to reach it.
    ``capacity`` is in tokens -- a flat cache is indexed by absolute position,
    so it holds the whole run.
    """
    name: str
    layers: Tuple[int, ...]
    components: Tuple[Tuple[str, Tuple[int, ...], "torch.dtype"], ...]
    capacity: int

    @property
    def per_entry_bytes(self) -> int:
        return sum(reduce(lambda a, b: a * b, shape, 1) * _itemsize(dtype)
                   for _, shape, dtype in self.components)

    @property
    def nbytes(self) -> int:
        return len(self.layers) * self.capacity * self.per_entry_bytes


def _check_stream_names(streams) -> None:
    """Stream names are unique. A repeat silently loses a layout (``_layouts``
    is keyed by name), and since ``attach()`` namespaces a shared layer's
    kwargs by name, a repeat would also collide there -- unique names rule
    both out at once."""
    seen = set()
    for st in streams:
        if st.name in seen:
            raise ValueError(
                f"two streams are both named {st.name!r}; stream names must "
                f"be unique")
        seen.add(st.name)


def build_kv_cache(streams, *,
                   kv_budget=None,
                   max_num_pages: Optional[int] = None,
                   max_seq_length: Optional[int] = None,
                   max_num_batched_requests: int = 1,
                   max_num_batched_tokens: int = 1,
                   device: str = "cuda",
                   verbose: bool = True,
                   **plan_kwargs) -> "KVCachePlan":
    """Declare the KV streams, plan the page geometry, size the pool and
    allocate it -- the whole cache, in one call.

    Give exactly one of ``kv_budget`` (bytes, the better knob) or
    ``max_num_pages``. ``max_seq_length`` is required with a budget (that is
    what the pool is sized to hold); with an explicit page count it is
    optional, and adds the floor check that the pool holds one request of
    that length.

    The returned plan owns the pool; ``attach(mpk, layer)`` is the only way
    to a cache tensor, so no caller can hold one that skipped the
    pool-identity check.

    A ``kind=KVKind.FLAT`` stream is split off here and never reaches
    ``plan_kv_groups``: no say in the shared page size, no page table, but
    the plan still owns and budgets its storage. With no paged stream at all
    the plan has zero groups.

    Several streams may cover one layer -- a windowed, a compressed and an
    indexer cache on one layer is a real shape (DeepSeek V4) -- and
    ``attach`` namespaces each by name so they cannot collide.
    """
    streams = list(streams)
    _check_stream_names(streams)
    flat_streams = [st for st in streams if not st.paged]
    paged_streams, merged_name = _merge_identical_streams(
        [st for st in streams if st.paged])
    if kv_budget is not None and max_seq_length is None:
        raise ValueError(
            "max_seq_length is required with kv_budget: a budget is sized to "
            "hold a request of some length, and there is nothing to size to "
            "without it")
    if flat_streams:
        if max_seq_length is None:
            raise ValueError(
                f"max_seq_length is required by unpaged stream(s) "
                f"{[st.name for st in flat_streams]}: a flat cache is indexed "
                f"by absolute position, so its capacity IS the run length")
        if max_num_batched_requests != 1:
            raise ValueError(
                f"unpaged stream(s) {[st.name for st in flat_streams]} with "
                f"max_num_batched_requests={max_num_batched_requests}: a flat "
                f"cache is indexed by absolute position with no request "
                f"dimension, so two requests would write the same rows. This "
                f"was always true of these kernels; declaring the stream is "
                f"what makes it checkable.")

    if paged_streams:
        plan = plan_kv_groups([st._spec() for st in paged_streams],
                              **plan_kwargs)
    else:
        plan = _plan_with_no_paged_streams(**plan_kwargs)
    plan._layouts = {st.name: list(st.components) for st in paged_streams}
    plan.flat_streams = tuple(st._flat(max_seq_length) for st in flat_streams)
    # attach() resolves per DECLARED stream, not per merged group: a layer can
    # carry several streams, and each keeps its own components and kwargs.
    plan._declared = tuple(streams)
    plan._merged_name = merged_name
    _resolve_pool_size(plan, kv_budget=kv_budget, max_num_pages=max_num_pages,
                       max_seq_length=max_seq_length,
                       max_num_batched_requests=max_num_batched_requests,
                       max_num_batched_tokens=max_num_batched_tokens,
                       device=_device_index(device), verbose=verbose)
    return plan._materialize(device=device)


def _device_index(device) -> int:
    """The ordinal mem_get_info wants, from whatever form the caller gave.

    A bare "cuda" means the CURRENT device, not device 0 -- every rank of a
    multi-GPU run passes "cuda" and must measure its own card.
    """
    if isinstance(device, int):
        return device
    dev = torch.device(device)
    if dev.type != "cuda":
        return 0
    return dev.index if dev.index is not None else torch.cuda.current_device()


class KVUnificationError(ValueError):
    """A stream does not fit the shared page size, so a single-page-size plan
    is impossible. Multi-bucket planning might be a future work for models work
    better with multiple page sizes.

    A ValueError, not a bare Exception: it is a bad CLI/config choice (the
    page size a caller picked), the same class of mistake every demo's
    `except ValueError: raise SystemExit(...)` guard already exists to turn
    into a clean message instead of a raw traceback -- it should not need its
    own separate except clause at every call site to get that.
    """


# Tokens per KV tile in the windowed attention kernel. A windowed task starts
# loading at a tile boundary, so a page is dead only once entirely below it.
#
# The scheduler uses the same number under the name MPK_KV_WINDOW_TILE.
def _window_tile_from_header(fallback: int = 64) -> int:
    import re
    from pathlib import Path

    header = (Path(__file__).resolve().parents[3] / "include" / "mirage" /
              "persistent_kernel" / "runtime_header.h")
    try:
        m = re.search(r"^#define\s+MPK_KV_WINDOW_TILE\s+(\d+)\s*$",
                      header.read_text(), re.M)
    except OSError:
        return fallback
    return int(m.group(1)) if m else fallback


KV_WINDOW_TILE = _window_tile_from_header()


def default_kv_tile(target_cc: Optional[int] = None) -> int:
    """KV tile to assume for a spec that declares none."""
    if target_cc is None:
        try:
            props = torch.cuda.get_device_properties(0)
            target_cc = props.major * 10 + props.minor
        except Exception:
            return 64
    return 64 if target_cc >= 90 else 128


@dataclass
class KVGroupSpec:
    """Per-group config consumed by PersistentKernel: the group's page table
    advances ``block_size`` raw tokens per page.

    ``window_size`` (0 = full attention) lets the scheduler recycle pages that
    have fallen out of the window mid-request."""
    block_size: int
    window_size: int = 0


def pages_per_request(block_size: int, window_size: int, max_seq_length: int,
                      max_num_batched_tokens: int = 1) -> int:
    """Worst-case pages one request holds in a group at any single step.

    Matching the scheduler, pages are counted as allocated for the batch's
    LAST token and recycled against its FIRST, so a wide batch holds up to
    ``max_num_batched_tokens`` extra."""
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
    (``allocate_pool``) and wires layers (``layer_info``)."""

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
        tile: int = 64           # kernel KV tile block_size is a multiple of
        tile_declared: bool = False   # False = took the device default

    target_page_bytes: int
    num_slots: int
    groups: Tuple["KVCachePlan.Group", ...]
    # Set once by resolve_pool_size. The page tables and the pool are built
    # in different places and both read from here.
    max_num_pages: Optional[int] = None
    # Streams whose reader is not paged: no group, no page table, storage
    # owned here anyway. Empty for every model whose kernels index by page.
    flat_streams: Tuple["FlatStream", ...] = ()
    # Filled in by declare_kv / materialize: the component layouts the streams
    # declared, and the pool built from them. Holding the views here is what
    # lets attach() be the only way to reach a cache tensor.
    _layouts: Optional[dict] = None
    _pool: Optional["torch.Tensor"] = None
    _views: Optional[dict] = None
    _flat_pool: Optional["torch.Tensor"] = None
    _flat_views: Optional[dict] = None
    # The streams as DECLARED, before merging, plus where each one landed.
    # A layer may appear in several; attach() walks them all.
    _declared: Tuple["KVStream", ...] = ()
    _merged_name: Optional[dict] = None

    # ── what PersistentKernel consumes ────────────────────────────────────

    def group_specs(self):
        """The kv_groups= argument for PersistentKernel."""
        return [KVGroupSpec(block_size=g.block_size, window_size=g.window_size)
                for g in self.groups]

    def merged_with(self, stream_name: str) -> str:
        """The group a declared, paged stream ended up in after merging.
        Two streams sharing this value share one page table."""
        return (self._merged_name or {}).get(stream_name, stream_name)

    # ── sizing the pool ───────────────────────────────────────────────────

    @property
    def page_id_bytes(self) -> int:
        """Bytes one page id costs: that page at every slot."""
        return self.num_slots * self.target_page_bytes

    @property
    def flat_bytes(self) -> int:
        """Bytes the unpaged streams occupy."""
        return sum(st.nbytes for st in self.flat_streams)

    def pages_for_budget(self, budget_bytes: int) -> int:
        """How many page ids fit in a byte budget, rounded down. The unpaged
        streams are not negotiable, so they are spent first."""
        assert budget_bytes >= 0
        remaining = budget_bytes - self.flat_bytes
        if remaining < 0:
            raise ValueError(
                f"the unpaged streams alone need "
                f"{format_bytes(self.flat_bytes)}, more than the "
                f"{format_bytes(budget_bytes)} budget")
        if self.page_id_bytes == 0:
            return 0                 # no paged group; the floor sets the count
        return remaining // self.page_id_bytes

    def _pool_pages(self, given: Optional[int]) -> int:
        """The page count to size a pool-shaped thing with: the caller's
        number, the one resolve_pool_size recorded, or both when they agree."""
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
        """Bytes the whole cache occupies, which is what a budget must cover."""
        return num_pages * self.page_id_bytes + self.flat_bytes

    def pages_needed(self, max_num_batched_requests: int, max_seq_length: int,
                     max_num_batched_tokens: int = 1) -> int:
        """Floor: page ids the batch holds at once in the worst case. Below
        it the free list wraps and re-hands a live page.

        Assumes a windowed group recycles, which offline and online_pinned do
        but spec-decode does not. The planner does not know the mode, so under
        spec-decode this floor is too low and the real check is
        PersistentKernel._check_kv_capacity at graph-build time -- same
        arithmetic, mode-aware, later error."""
        return max_num_batched_requests * sum(
            pages_per_request(g.block_size, g.window_size, max_seq_length,
                              max_num_batched_tokens)
            for g in self.groups)

    def build_meta_tensors(self, *, max_seq_length: int,
                           max_num_pages: Optional[int] = None,
                           max_num_batched_requests: int = 1,
                           dtype=torch.int32, device: str = "cuda"):
        """Page-table buffers (indptr / indices / last_page_len) for every
        group. Merge into the meta_tensors dict before constructing
        PersistentKernel.

        The indices buffer is indexed by absolute page number within a
        request; a recycled slot keeps its place holding -1, so its span
        follows ``max_seq_length`` rather than the live page count.
        """
        max_num_pages = self._pool_pages(max_num_pages)
        out = {}
        for g_id, g in enumerate(self.groups):
            span = max(max_num_pages,
                       max_num_batched_requests
                       * ((max_seq_length + g.block_size - 1) // g.block_size))
            out[f"paged_kv_indptr_buffer_{g_id}"] = torch.zeros(
                max_num_batched_requests + 1, dtype=dtype, device=device)
            out[f"paged_kv_indices_buffer_{g_id}"] = torch.zeros(
                span, dtype=dtype, device=device)
            out[f"paged_kv_last_page_len_buffer_{g_id}"] = torch.zeros(
                max_num_batched_requests, dtype=dtype, device=device)
            # In-place compaction snapshots the index buffer, so it needs the
            # same span. PersistentKernel fills these in when they are absent,
            # but not in online_pinned mode, which allocates nothing itself --
            # and the span formula it would use is this one, so compute it
            # once here rather than keeping two copies in step.
            out[f"paged_kv_indices_snapshot_{g_id}"] = torch.zeros(
                span, dtype=dtype, device=device)
        return out

    # ── explaining the plan ───────────────────────────────────────────────

    def first_recycled_step(self, group) -> Optional[int]:
        """Step at which a windowed group first frees a page. A page dies once
        the window edge, rounded down to a tile, has passed it."""
        if group.window_size <= 0:
            return None
        tiles = -(-group.block_size // KV_WINDOW_TILE) * KV_WINDOW_TILE
        return group.window_size - 1 + tiles

    def describe(self, max_seq_length: Optional[int] = None) -> str:
        """How the shared page turned into each stream's block size. Streams
        with smaller entries pack more tokens into the same page."""
        flat_lines = [
            f"  unpaged '{st.name}': {len(st.layers)} layer(s) x "
            f"{st.capacity} tokens x {st.per_entry_bytes} B = "
            f"{format_bytes(st.nbytes)}  (no group, no page table -- the "
            f"kernel reads it flat)"
            for st in self.flat_streams]
        if not self.groups:
            return "\n".join(
                ["KV cache: no paged stream, so ZERO KV groups"] + flat_lines)
        lines = [
            f"KV page: {format_bytes(self.target_page_bytes)} x "
            f"{self.num_slots} slot(s) = {format_bytes(self.page_id_bytes)} "
            f"per page id"
        ]
        warnings = []
        for g in self.groups:
            src = "declared" if g.tile_declared else "device default"
            pad = (f", {g.padding_bytes_per_page} B padding"
                   if g.padding_bytes_per_page else "")
            note = ""
            if g.window_size:
                first = self.first_recycled_step(g)
                if max_seq_length is not None and first >= max_seq_length:
                    note = "  <-- window never recycles here"
                    warnings.append(
                        f"group {g.group_id} ('{g.spec_name}') declares a "
                        f"{g.window_size}-token window, but a {g.block_size}-"
                        f"token block only frees its first page at step "
                        f"{first}, past this {max_seq_length}-token run. The "
                        f"window is inert at this length -- not a leak, and "
                        f"not a reason to lower the block size, which would "
                        f"raise the page count.")
                else:
                    note = f", recycles from step {first}"
            lines.append(
                f"  group {g.group_id} '{g.spec_name}': block {g.block_size} "
                f"tokens ({g.entries_per_page} entries, tile {g.tile} "
                f"{src}){pad}{note}")
        lines += flat_lines
        for w in warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)

    def _layer_info(self, layer_id: int) -> Tuple[int, int]:
        """(group_id, slot_id) for a layer sitting in exactly ONE group. A
        layer in several has no single answer -- that caller wants
        ``attach()``, which returns every cache the layer carries."""
        hits = [(g.group_id, g.layer_ids.index(layer_id))
                for g in self.groups if layer_id in g.layer_ids]
        if len(hits) > 1:
            names = ", ".join(repr(self.groups[gid].spec_name)
                              for gid, _ in hits)
            raise KeyError(
                f"layer {layer_id} is in {len(hits)} groups ({names}), so it "
                f"has no single group id; use attach(), which returns every "
                f"cache the layer carries")
        if hits:
            return hits[0]
        for st in self.flat_streams:
            if layer_id in st.layers:
                raise KeyError(
                    f"layer {layer_id} belongs to unpaged stream "
                    f"'{st.name}', which has no group or page table; reach it "
                    f"through attach()")
        raise KeyError(f"layer {layer_id} not covered by any group")

    def _group_of(self, stream: "KVStream", layer_id: int) -> Tuple[int, int]:
        """(group_id, slot_id) holding this DECLARED stream's layer. The
        stream may have been folded into a merged one, so match on the
        merged name, not the declared one."""
        spec_name = (self._merged_name or {}).get(stream.name, stream.name)
        for g in self.groups:
            if g.spec_name == spec_name and layer_id in g.layer_ids:
                return g.group_id, g.layer_ids.index(layer_id)
        raise KeyError(
            f"stream '{stream.name}' (planned as '{spec_name}') has no group "
            f"holding layer {layer_id}")

    def _layer_streams(self, layer_id: int):
        """Every declared stream on this layer, as
        ``(stream, group_id | None, slot_id | None)``. Unpaged streams have no
        group, so both ids are None."""
        out = []
        for st in self._declared:
            if layer_id not in st.layers:
                continue
            if st.paged:
                group_id, slot_id = self._group_of(st, layer_id)
                out.append((st, group_id, slot_id))
            else:
                out.append((st, None, None))
        return out

    # ── allocation ────────────────────────────────────────────────────────

    def _materialize(self, *, max_num_pages: Optional[int] = None,
                     device: str = "cuda"):
        """Allocate the pool from the declared layouts and keep the views.

        Replaces allocate_pool plus carrying ``views`` around: the plan holds
        them, so no caller can end up with a cache tensor that never went
        through the identity check -- attach() is the only way back out."""
        if self._layouts is None:
            raise RuntimeError(
                "this plan was not built by declare_kv(), so it does not know "
                "the component layouts; use declare_kv([KVStream(...), ...])")
        self._pool, self._views = self._allocate_pool(
            self._layouts, max_num_pages, device)
        self._flat_pool, self._flat_views = self._allocate_flat(device)
        return self

    def zero_(self):
        """Clear the whole cache. One allocation backs every group and slot,
        so this is one memset rather than a walk over the views."""
        if self._pool is None:
            raise RuntimeError("materialize() has not run on this plan")
        self._pool.zero_()
        if self._flat_pool is not None:
            self._flat_pool.zero_()
        return self

    def views(self, group_id: int):
        """{component: (slots, pages, page size, *entry shape)} for one group.

        For code that indexes the cache directly rather than handing it to the
        megakernel -- demo/qwen3's eager PyTorch reference writes
        key_cache[layer, 0, step] itself, and needs the tensors, not an
        attachment. Anything feeding a paged-attention task wants attach()
        instead, which resolves the slot and checks the view is still on the
        pool."""
        if self._views is None:
            raise RuntimeError("materialize() has not run on this plan")
        return self._views[group_id]

    def attach(self, mpk, layer_id: int, prefix: str = "layer"):
        """Everything this layer's tasks need, for EVERY cache it carries:

            mpk.paged_attention_layer(..., **kv.attach(mpk, i))

        Folds group/slot resolution, the views walk and the pool-identity
        check (which catches a cache copied out by a stray ``.contiguous()``)
        into one accessor. A layer with several streams returns them all,
        each namespaced by its own name (names are required unique, so this
        cannot collide); a layer with one stream gets bare keys.
        ``window_size`` appears only where a window was declared; absent
        means 0, the default every layer function already takes.
        """
        if self._views is None:
            raise RuntimeError("materialize() has not run on this plan")
        entries = self._layer_streams(layer_id)
        if not entries:
            raise KeyError(f"layer {layer_id} is not covered by any KV stream")
        solo = len(entries) == 1
        out = {}
        for stream, group_id, slot_id in entries:
            key = (lambda s: s) if solo else (lambda s, n=stream.name: f"{n}_{s}")
            out[key("group_id")] = group_id
            if stream.window:
                out[key("window_size")] = stream.window
            for name, entry_shape, dtype in stream.components:
                if group_id is None:
                    view = self._flat_views[(stream.name, layer_id)][name]
                    entry_dims = view.shape[1:]
                else:
                    view = self._views[group_id][name][slot_id]
                    entry_dims = view.shape[2:]
                # Guards the planner, not the caller: the view is built from
                # this same declaration, so a mismatch means the pool was laid
                # out differently than the stream asked for.
                assert tuple(entry_dims) == tuple(entry_shape), (
                    f"{name} view has entry shape {tuple(entry_dims)} but "
                    f"stream '{stream.name}' declared {tuple(entry_shape)}")
                assert view.dtype == dtype, (
                    f"{name} view is {view.dtype}, declared {dtype}")
                label = f"{prefix}_{layer_id}_{key(name)}_cache"
                check = self._assert_in_flat if group_id is None else self._assert_in_pool
                out[key(f"{name}_cache")] = mpk.attach_input(
                    torch_tensor=check(view, label), name=label)
        return out

    def _allocate_flat(self, device: str = "cuda"):
        """The unpaged streams as ONE allocation, plus typed views keyed
        ``views[(stream name, layer_id)][component]``. Laid out stream / layer
        / component, so a view has the shape and stride of the tensor it
        replaces. The stream name is part of the key because one layer may
        carry more than one unpaged stream.
        """
        total = self.flat_bytes
        buf = torch.zeros(total, dtype=torch.uint8, device=device)
        self._flat_span = (buf.data_ptr(), buf.data_ptr() + total)
        views, off = {}, 0
        for st in self.flat_streams:
            for layer_id in st.layers:
                comps = {}
                for cname, entry_shape, dtype in st.components:
                    entry_elems = reduce(lambda a, b: a * b, entry_shape, 1)
                    itemsize = _itemsize(dtype)
                    span = st.capacity * entry_elems
                    assert off % itemsize == 0, (
                        f"component '{st.name}.{cname}' starts at byte {off}, "
                        f"not a multiple of its {itemsize} B element")
                    comps[cname] = buf.view(dtype)[
                        off // itemsize:off // itemsize + span].view(
                        st.capacity, *entry_shape)
                    off += span * itemsize
                views[(st.name, layer_id)] = comps
        assert off == total, f"flat allocation walked {off} of {total} B"
        return buf, views

    def _assert_in_flat(self, tensor, name: str = "tensor"):
        """The unpaged twin of _assert_in_pool: still on the plan's buffer,
        and still addressed one token row at a time."""
        if getattr(self, "_flat_span", None) is None:
            raise RuntimeError(
                "the unpaged streams have not been allocated on this plan")
        lo, hi = self._flat_span
        ptr = tensor.data_ptr()
        if not lo <= ptr < hi:
            raise AssertionError(
                f"{name} is not a view on the KV cache: storage 0x{ptr:x} is "
                f"outside [0x{lo:x}, 0x{hi:x})")
        want = reduce(lambda a, b: a * b, tensor.shape[1:], 1)
        got = tensor.stride(0)
        if got != want:
            raise AssertionError(
                f"{name} has row stride {got}, expected {want}: it is on the "
                f"cache but not addressed one token at a time")
        return tensor

    def _allocate_pool(self, entry_layouts,
                       max_num_pages: Optional[int] = None,
                       device: str = "cuda"):
        """The entire KV cache as ONE allocation, plus typed views.

        Shape: ``[num_slots, max_num_pages, target_page_bytes]``. A page id
        denotes page ``p`` of every slot, held by one group at a time. A
        stream may carve its page into several components (K and V), laid
        out component-major inside the page.

        entry_layouts: ``{spec_name: [(component_name, entry_shape, dtype),
            ...]}``; the components' per-entry bytes must sum to at most the
            stream's per_entry_bytes.
        Returns ``(pool, views)``; ``views[group_id][component_name]`` is
            shaped ``[num_slots, max_num_pages, entries_per_page,
            *entry_shape]`` and aliases ``pool``."""
        max_num_pages = self._pool_pages(max_num_pages)
        pool = torch.zeros(self.num_slots, max_num_pages,
                           self.target_page_bytes, dtype=torch.uint8,
                           device=device)
        self._pool_span = (pool.data_ptr(),
                           pool.data_ptr() + pool.numel() * pool.element_size())
        views = {}
        for g in self.groups:
            if g.spec_name is None:      # placeholder group, holds nothing
                views[g.group_id] = {}
                continue
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

    def _assert_in_pool(self, tensor, name: str = "tensor"):
        """Assert if a cache tensor is a view ON the pool, not a copy of one."""
        if getattr(self, "_pool_span", None) is None:
            raise RuntimeError("the pool has not been allocated on this plan")
        lo, hi = self._pool_span
        ptr = tensor.data_ptr()
        if not lo <= ptr < hi:
            raise AssertionError(
                f"{name} is not a view on the KV pool: storage 0x{ptr:x} is "
                f"outside [0x{lo:x}, 0x{hi:x})")
        want = self.elems_per_page(tensor.dtype)
        got = tensor.stride(0)
        if got != want:
            raise AssertionError(
                f"{name} has page stride {got}, expected {want}: it is on the "
                f"pool but not addressed a whole page at a time. "
                f"(Pass views[g][c][slot_id], not views[g][c].)")
        return tensor

    def elems_per_page(self, dtype) -> int:
        """How wide one page is, in elements of ``dtype``.

        Always the full page, not the view's packed entry span. NOT the
        kernel's PAGE_STRIDE in token, which is this divided by the width."""
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


def resolve_kv_budget(spec) -> int:
    """Turn a user-facing KV budget into bytes.

    Absolute sizes only: ``"24GiB"``, ``"512MiB"``, or a raw int. A bare
    number as a string is rejected.
    """
    if isinstance(spec, int) and not isinstance(spec, bool):
        return int(spec)
    text = str(spec).strip()
    units = {"KIB": 1024, "MIB": 1024**2, "GIB": 1024**3, "TIB": 1024**4,
             "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4,
             "B": 1}
    upper = text.upper()
    for suffix, scale in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if upper.endswith(suffix):
            return int(float(text[:-len(suffix)]) * scale)
    raise ValueError(
        f"KV budget {spec!r} needs a unit, e.g. '24GiB' or '512MiB'.")


def _resolve_pool_size(plan: "KVCachePlan", *, kv_budget=None,
                       max_num_pages: Optional[int] = None,
                       max_seq_length: Optional[int] = None,
                       max_num_batched_requests: int = 1,
                       max_num_batched_tokens: int = 1,
                       device: int = 0, verbose: bool = True) -> int:
    """The page count to build the pool with, from a byte budget or an
    explicit count. Exactly one of ``kv_budget`` / ``max_num_pages`` may be
    given. Without ``max_seq_length`` the floor check is skipped --
    ``build_kv_cache`` requires one alongside a budget for that reason.
    """
    if (kv_budget is None) == (max_num_pages is None):
        raise ValueError("give exactly one of kv_budget / max_num_pages")

    if max_num_pages is not None:
        pages, source = max_num_pages, "explicit page count"
    elif plan.page_id_bytes == 0:
        # No paged group: a page id holds nothing, so the budget buys no pages
        # and only has to cover the unpaged streams.
        plan.pages_for_budget(resolve_kv_budget(kv_budget))   # budget check
        pages = plan.pages_needed(max_num_batched_requests,
                                  max_seq_length or 1,
                                  max_num_batched_tokens)
        source = f"budget {kv_budget} (no paged streams)"
    else:
        pages = plan.pages_for_budget(resolve_kv_budget(kv_budget))
        source = f"budget {kv_budget}"

    if max_seq_length is not None:
        floor = plan.pages_needed(max_num_batched_requests, max_seq_length,
                                  max_num_batched_tokens)
        if pages < floor:
            raise ValueError(
                f"KV pool too small: {source} gives {pages} page(s), but "
                f"{max_num_batched_requests} request(s) at {max_seq_length} "
                f"tokens need {floor} "
                f"({format_bytes(plan.budget_bytes(floor))})")

    # MPK_MAX_NUM_PAGES cannot be zero even when nothing allocates a page.
    pages = max(pages, 1)
    plan.max_num_pages = pages          # both sizing sites read it from here
    if verbose:
        print(plan.describe(max_seq_length))
    used = plan.budget_bytes(pages)
    free, total = torch.cuda.mem_get_info(device)
    if verbose:
        flat = (f" + {format_bytes(plan.flat_bytes)} unpaged"
                if plan.flat_streams else "")
        print(f"KV pool: {pages} pages x "
              f"{format_bytes(plan.page_id_bytes)}{flat} = "
              f"{format_bytes(used)}  "
              f"({source}; device has {format_bytes(free)} free of "
              f"{format_bytes(total)})")
    if used > free:
        raise ValueError(
            f"the KV cache alone ({format_bytes(used)}) exceeds free memory "
            f"({format_bytes(free)}), before the model weights")
    return pages


def _plan_with_no_paged_streams(target_page_bytes: Optional[int] = None,
                                block_size: int = 64,
                                target_cc: Optional[int] = None
                                ) -> KVCachePlan:
    """The plan for a model with no paged KV: zero groups, zero bytes.

    The runtime compiles at MPK_NUM_KV_GROUPS == 0; device-side arrays that
    would go zero-length size with MPK_NUM_KV_GROUPS_ARRAY instead.
    """
    if target_page_bytes is not None:
        raise ValueError(
            "target_page_bytes was given but no stream is paged, so there is "
            "no page to size")
    return KVCachePlan(target_page_bytes=0, num_slots=0, groups=())


def plan_kv_groups(
    specs,
    target_page_bytes: Optional[int] = None,
    block_size: Optional[int] = None,
    target_cc: Optional[int] = None,
) -> KVCachePlan:
    """Turn KVSpec declarations into a KVCachePlan.

    Give exactly one of ``target_page_bytes`` (bytes, exact) or ``block_size``
    (tokens); neither given defaults to ``block_size=64``. They drive
    different fitting rules for every spec that is not the one setting the
    page size:

    - ``target_page_bytes``: greedy packing (``_fit_block_size``) -- every
      spec gets as many entries as fit, floored to its own tile. Use this
      when the exact byte budget is already known (e.g. it must unify
      several ``compress_ratio``s that no single token count can at once).
    - ``block_size``: each spec computes its own tile-legal native size in
      isolation (``_native_fit``); the largest becomes ``target_page_bytes``.
      Every other spec then scales up by an exact integer ratio (zero
      padding) or, failing that, keeps its native size and pads the rest,
      never repacked (``_fit_to_anchor``) -- matching vLLM's
      ``unify_kv_cache_spec_page_size``. Coarser than the greedy path when a
      spec's ratio to the anchor isn't clean.

    A spec that cannot fit even one tile's worth of entries raises
    KVUnificationError; one that fits but pays padding warns instead of
    failing silently (``_warn_if_page_wastes_bytes``).

    Layers are then chunked into groups of ``_group_size`` layers so all
    groups share one slot layout with minimal waste.
    """
    specs = list(specs)
    assert specs, "need at least one KVSpec"
    names = [s.name for s in specs]
    assert len(set(names)) == len(names), f"duplicate spec names: {names}"
    if target_page_bytes is not None and block_size is not None:
        raise ValueError("give exactly one of block_size or target_page_bytes")

    tiles = {s.name: (s.block_size_multiple_of if s.block_size_multiple_of
                      else default_kv_tile(target_cc)) for s in specs}

    if target_page_bytes is None:
        native = {s.name: _native_fit(s, block_size or 64, tiles[s.name])
                 for s in specs}
        target_page_bytes = max(native_bytes for _, native_bytes in native.values())
        per_spec = {s.name: _fit_to_anchor(s, native[s.name], target_page_bytes)
                    for s in specs}
    else:
        per_spec = {s.name: _fit_block_size(s, target_page_bytes, tiles[s.name])
                    for s in specs}
    declared = {s.name: s.block_size_multiple_of is not None for s in specs}
    group_size = _group_size([len(s.layer_ids) for s in specs])
    _warn_if_group_size_starved(specs, group_size)
    _warn_if_page_wastes_bytes(specs, target_page_bytes, per_spec)

    groups = []
    for s in specs:
        spec_block_size, entries, padding = per_spec[s.name]
        layers = list(s.layer_ids)
        for start in range(0, len(layers), group_size):
            chunk = layers[start:start + group_size]
            chunk += [None] * (group_size - len(chunk))
            groups.append(KVCachePlan.Group(
                group_id=len(groups),
                spec_name=s.name,
                layer_ids=tuple(chunk),
                block_size=spec_block_size,
                entries_per_page=entries,
                padding_bytes_per_page=padding,
                window_size=s.window_size or 0,
                tile=tiles[s.name],
                tile_declared=declared[s.name],
            ))

    return KVCachePlan(
        target_page_bytes=target_page_bytes,
        num_slots=group_size,
        groups=tuple(groups),
    )


def _native_fit(spec: KVSpec, block_size: int, tile: int) -> Tuple[int, int]:
    """This spec's own tile-legal (entries, bytes) at ``block_size`` tokens,
    with no cross-spec reconciliation yet. Raises KVUnificationError if it
    can't fit one tile's worth of entries even in isolation. A bounded spec
    skips the tile arithmetic -- it is always exactly 1 entry.
    """
    if spec.bounded:
        return 1, spec.per_entry_bytes
    entries_per_tile = max(tile // spec.compress_ratio, 1)
    entries = block_size // spec.compress_ratio
    entries -= entries % entries_per_tile
    if entries <= 0:
        want = entries_per_tile * spec.per_entry_bytes
        raise KVUnificationError(
            f"spec '{spec.name}': {block_size} tokens give "
            f"{block_size // spec.compress_ratio} entries, short of the "
            f"{entries_per_tile} its {tile}-token tile needs "
            f"(>= {want} B/page at this compress_ratio)")
    return entries, entries * spec.per_entry_bytes


def _fit_to_anchor(spec: KVSpec, native: Tuple[int, int],
                   target_page_bytes: int):
    """Reconcile this spec's ``_native_fit`` result against the shared page:
    (block_size_tokens, entries, padding_bytes).

    Scales up by an exact integer ratio when one exists (zero padding);
    otherwise leaves the spec at its native size and pads the rest -- NEVER
    repacked here, unlike ``_fit_block_size``. A bounded spec always takes the
    pad branch: scaling its entries up would defeat the point of a fixed
    one-per-request state.
    """
    native_entries, native_bytes = native
    if spec.bounded:
        if target_page_bytes < native_bytes:
            raise KVUnificationError(
                f"spec '{spec.name}': a bounded state needs {native_bytes} B "
                f"but the shared page is only {target_page_bytes} B")
        return spec.compress_ratio, 1, target_page_bytes - native_bytes
    if target_page_bytes % native_bytes == 0:
        entries = native_entries * (target_page_bytes // native_bytes)
        padding = 0
    else:
        entries = native_entries
        padding = target_page_bytes - native_bytes
    block_size = entries * spec.compress_ratio
    return block_size, entries, padding


def _fit_block_size(spec: KVSpec, target_page_bytes: int, tile: int):
    """Page capacity for a stream as (block_size_tokens, entries, padding_bytes).

    Fits what the page holds, floored to a tile multiple; the leftover is
    padding. Always repacks (unlike ``_fit_to_anchor``) -- used only for the
    ``target_page_bytes`` path, where the caller already picked an exact
    byte budget and wants it used well. A bounded spec is the one exception:
    it stays at 1 entry rather than being packed with copies nothing reads.
    """
    if spec.bounded:
        if target_page_bytes < spec.per_entry_bytes:
            raise KVUnificationError(
                f"spec '{spec.name}': a bounded state needs "
                f"{spec.per_entry_bytes} B but the shared page is only "
                f"{target_page_bytes} B")
        return (spec.compress_ratio, 1,
               target_page_bytes - spec.per_entry_bytes)
    # Entries must land on a tile boundary once converted back to tokens.
    entries_per_tile = max(tile // spec.compress_ratio, 1)
    entries = target_page_bytes // spec.per_entry_bytes
    entries -= entries % entries_per_tile
    if entries <= 0:
        want = entries_per_tile * spec.per_entry_bytes
        raise KVUnificationError(
            f"spec '{spec.name}': a {target_page_bytes} B page holds "
            f"{target_page_bytes // spec.per_entry_bytes} entries of "
            f"{spec.per_entry_bytes} B, short of the {entries_per_tile} its "
            f"{tile}-token tile needs (>= {want} B/page)")
    block_size = entries * spec.compress_ratio
    assert block_size % tile == 0, (
        f"spec '{spec.name}': derived block_size {block_size} is not a "
        f"multiple of the {tile}-token kernel tile")
    padding = target_page_bytes - entries * spec.per_entry_bytes
    return block_size, entries, padding


def _group_size(layer_counts):
    """Slots per group. A group with k real layers padded to S slots strands
    (S-k)/S of every page it holds, so:

    - near-equal counts (hi < 1.5 * lo): pad the smaller stream up;
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


def _warn_if_group_size_starved(specs, group_size: int) -> None:
    """Flag a spec that fragments because a much smaller one set group_size.

    Mirrors ``_group_size``'s branches to tell its two good outcomes (pad up
    to hi; split on a real gcd > 1) from its one bad one, with one deliberate
    difference: unlike ``_group_size``, this does NOT treat ``g == lo`` as
    clean, since when lo is small (e.g. 1) that holds trivially for any spec
    sizes -- gcd with 1 is always 1. That case is exactly what's worth
    flagging: a 1-layer speculative-decode draft next to a 48-layer target
    with a different page layout (so ``_merge_identical_streams`` can't fold
    them) forces group_size=1, fragmenting the target into 48 single-slot
    groups with zero padding, which a padding-only check would call clean.
    """
    counts = [len(s.layer_ids) for s in specs]
    lo, hi = min(counts), max(counts)
    if hi < lo * 1.5 or hi == lo:
        return  # padded up to hi, or already uniform -- no fragmentation
    g = reduce(gcd, counts)
    if g > 1 and (g == lo or g >= lo // 2):
        return  # a real (> 1) usable gcd: every spec divides it, zero waste

    setters = [s.name for s in specs if len(s.layer_ids) == group_size]
    setter = repr(setters[0]) if setters else "a smaller stream"
    for s in specs:
        count = len(s.layer_ids)
        if count <= group_size:
            continue
        chunks = -(-count // group_size)
        padding = chunks * group_size - count
        warnings.warn(
            f"KV plan: stream {s.name!r} ({count} layers) shares no clean "
            f"multiple with the {group_size}-layer group size that "
            f"{setter} set, so it fragments into {chunks} group(s)"
            + (f" ({padding} padded layer(s) on the last one)"
               if padding else " with no per-page padding, but "
               f"{chunks}x the group-table and scheduler overhead a single "
               f"group would cost") +
            ". A small extra stream (e.g. a speculative-decode draft) "
            "forces this on an otherwise-uniform target when its page "
            "layout doesn't match closely enough to merge via "
            "_merge_identical_streams.", stacklevel=3)


def _warn_if_page_wastes_bytes(specs, target_page_bytes: int,
                               per_spec: dict) -> None:
    """Flag a spec whose entries do not tile the shared page exactly.

    A non-exact spec is floored to a tile multiple and the remainder reported
    as ``padding_bytes_per_page``, charged on EVERY page its group ever
    holds -- so even a "small" per-page percentage compounds across the
    pool. Warn unconditionally (no threshold) and let the caller judge, same
    philosophy as ``_warn_if_group_size_starved``.
    """
    for s in specs:
        _, entries, padding = per_spec[s.name]
        if padding <= 0:
            continue
        pct = 100 * padding / target_page_bytes
        warnings.warn(
            f"KV plan: stream {s.name!r} wastes {padding} B ({pct:.1f}%) of "
            f"every {target_page_bytes} B page -- its {entries} entries of "
            f"{s.per_entry_bytes} B do not tile the page exactly. Pick a "
            f"block_size/target_page_bytes divisible by this spec's entry "
            f"size (or its own block_size_multiple_of tile) if the waste "
            f"matters.", stacklevel=3)


# ── debug ─────────────────────────────────────────────────────────────────


class KVEventLog:
    """Record-and-verify instrumentation for the runtime page allocator.

    Constructing one wires a ``kv_event_log`` meta tensor into the kernel
    before compilation. After the kernel ran, ``verify()`` replays the log
    and asserts allocator invariants.

    Log format: log[0] = event count; event i is 4 ints at [4i+1 .. 4i+4] =
    (type, group_id, request_slot, page_id), type 1=ALLOC, 2=FREE, 3=ITER,
    4=MOVE. A MOVE carries no group and reuses the last two fields for the
    request's old and new batch slot.
    """

    def __init__(self, pk, plan: KVCachePlan, capacity: int = 65536,
                 device: str = "cuda"):
        self.num_groups = len(plan.groups)
        self.log = torch.zeros(capacity, dtype=torch.int32, device=device)
        pk.meta_tensors["kv_event_log"] = self.log

    def verify(self):
        """Replay the log; assert no double-alloc, no free of an unowned
        page, and zero pages still live at the end. Returns
        {"iterations": int, "compactions": int,
         "per_group": [{"allocs": int, "frees": int}]}."""
        return self.replay(self.log, self.num_groups)

    @staticmethod
    def replay(log: torch.Tensor, num_groups: int):
        """Standalone replay of a raw log tensor; same as verify()."""
        events = log.cpu().tolist()
        count = events[0]
        live = [set() for _ in range(num_groups)]
        stats = [{"allocs": 0, "frees": 0} for _ in range(num_groups)]
        iterations = 0
        compactions = 0
        for i in range(count):
            etype, g, _req, page = events[4 * i + 1: 4 * i + 5]
            if etype == 3:
                iterations += 1
                continue
            if etype == 4:
                compactions += 1
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
        return {"iterations": iterations, "compactions": compactions,
                "per_group": stats}
