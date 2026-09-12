"""KVCache: the object ``build_kv_cache()`` returns.

Pool shape: ``[num_slots, max_num_pages, target_page_bytes]``.
- page: one row of physical memory.
- block_size: raw tokens one page holds for a given stream.
- group: one page table -- a chunk of one stream's layers sharing one page.
- slot: index of one physical tensor; layer i of every group shares slot i.

Owns the pool once materialized; ``attach(mpk, layer)`` is the only way to a
cache tensor. See ``kv_planner.py`` for how a KVCache gets built.
"""

from dataclasses import dataclass
from functools import reduce
from typing import Optional, Tuple

import torch

from .kv_stream import KVStream, FlatStream, _itemsize


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


def format_bytes(nbytes: int) -> str:
    """Human-readable byte count, so page counts and budgets can be reported
    in the same units the user typed."""
    for unit, scale in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if nbytes >= scale:
            return f"{nbytes / scale:.2f} {unit}"
    return f"{nbytes} B"


@dataclass
class KVGroupConfig:
    """Per-group config for PersistentKernel: the page table advances
    ``block_size`` tokens per page. ``window_size=0`` is full attention;
    nonzero lets the scheduler recycle out-of-window pages."""
    block_size: int
    window_size: int = 0


def pages_per_request(block_size: int, window_size: int, max_seq_length: int,
                      max_num_batched_tokens: int = 1) -> int:
    """Worst-case pages one request holds in a group at any single step.

    Pages are counted as allocated for the batch's LAST token and recycled
    against its FIRST. A batch holds up to ``max_num_batched_tokens`` extra."""
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
class KVCache:
    """Planner output. Owns the pool once materialized; ``attach(mpk,
    layer)`` is the only way to a cache tensor."""

    @dataclass
    class Group:
        """One page table: a chunk of one stream's layers, padded with None
        up to the plan's slot count."""
        group_id: int
        stream_name: str
        layer_ids: Tuple[Optional[int], ...]
        block_size: int          # raw tokens per page
        entries_per_page: int    # = block_size / compress_ratio
        padding_bytes_per_page: int
        window_size: int = 0     # 0 = full attention
        tile: int = 64           # kernel KV tile block_size is a multiple of
        tile_declared: bool = False   # False = took the device default

    target_page_bytes: int
    num_slots: int
    groups: Tuple["KVCache.Group", ...]
    max_num_pages: Optional[int] = None    # set by _resolve_pool_size
    # Unpaged streams: no group/page table, storage owned here anyway.
    flat_streams: Tuple["FlatStream", ...] = ()
    # Filled in by build_kv_cache/_materialize; lets attach() be the only
    # way to reach a cache tensor.
    _layouts: Optional[dict] = None
    _pool: Optional["torch.Tensor"] = None
    _views: Optional[dict] = None
    _flat_pool: Optional["torch.Tensor"] = None
    _flat_views: Optional[dict] = None
    # Streams as DECLARED, before merging; attach() walks all that cover a
    # given layer.
    _declared: Tuple["KVStream", ...] = ()
    _merged_name: Optional[dict] = None

    # ── what PersistentKernel consumes ────────────────────────────────────

    def group_specs(self):
        """The kv_groups= argument for PersistentKernel."""
        return [KVGroupConfig(block_size=g.block_size, window_size=g.window_size)
                for g in self.groups]

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
        """Page count to use: the caller's number, the plan's recorded one,
        or both if they agree."""
        if given is None and self.max_num_pages is None:
            raise ValueError(
                "no pool size: pass max_num_pages here, or size this plan "
                "through build_kv_cache() first")
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
        """Floor: page ids the batch holds at once, worst case. Assumes
        windowed groups recycle (spec-decode does not); The authoritative
        check is at ``PersistentKernel._check_kv_capacity``."""
        return max_num_batched_requests * sum(
            pages_per_request(g.block_size, g.window_size, max_seq_length,
                              max_num_batched_tokens)
            for g in self.groups)

    def build_meta_tensors(self, *, max_seq_length: int,
                           max_num_pages: Optional[int] = None,
                           max_num_batched_requests: int = 1,
                           dtype=torch.int32, device: str = "cuda"):
        """Page-table buffers (indptr / indices / last_page_len) per group,
        for PersistentKernel's meta_tensors.

        Indices are indexed by absolute page number; a recycled slot keeps
        -1, so the span follows max_seq_length, not the live page count.
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
            # Compaction's scratch copy of the index buffer; same span since
            # online_pinned mode allocates nothing of its own.
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
                        f"group {g.group_id} ('{g.stream_name}') declares a "
                        f"{g.window_size}-token window, but a {g.block_size}-"
                        f"token block only frees its first page at step "
                        f"{first}, past this {max_seq_length}-token run. The "
                        f"window is inert at this length -- not a leak, and "
                        f"not a reason to lower the block size, which would "
                        f"raise the page count.")
                else:
                    note = f", recycles from step {first}"
            lines.append(
                f"  group {g.group_id} '{g.stream_name}': block {g.block_size} "
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
            names = ", ".join(repr(self.groups[gid].stream_name)
                              for gid, _ in hits)
            raise KeyError(
                f"layer {layer_id} is in {len(hits)} groups ({names}), so it "
                f"has no single group id; use attach()")
        if hits:
            return hits[0]
        for st in self.flat_streams:
            if layer_id in st.layers:
                raise KeyError(
                    f"layer {layer_id} belongs to unpaged stream "
                    f"'{st.name}', which has no group or page table")
        raise KeyError(f"layer {layer_id} not covered by any group")

    def _group_of(self, stream: "KVStream", layer_id: int) -> Tuple[int, int]:
        """(group_id, slot_id) holding this DECLARED stream's layer. The stream
        may have been folded into a merged one, match on the merged name."""
        merged = (self._merged_name or {}).get(stream.name, stream.name)
        for g in self.groups:
            if g.stream_name == merged and layer_id in g.layer_ids:
                return g.group_id, g.layer_ids.index(layer_id)
        raise KeyError(
            f"stream '{stream.name}' (planned as '{merged}') has no group "
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
        """Allocate the pool and keep the views, attach() is the only way out."""
        if self._layouts is None:
            raise RuntimeError(
                "this plan has no component layouts to allocate")
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

        For code indexing the cache directly instead of through the
        megakernel. A paged-attention task should use attach() instead."""
        if self._views is None:
            raise RuntimeError("materialize() has not run on this plan")
        return self._views[group_id]

    def attach(self, mpk, layer_id: int, prefix: str = "layer"):
        """Everything layer `i`'s tasks need, for every cache it carries:

            mpk.paged_attention_layer(..., **kv.attach(mpk, i))

        Several streams on one layer come back namespaced by name; a lone
        stream gets bare keys. window_size appears only where declared.
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
                # Guards the planner: a mismatch means the pool was laid out
                # differently than the stream declared.
                assert tuple(entry_dims) == tuple(entry_shape)
                assert view.dtype == dtype
                label = f"{prefix}_{layer_id}_{key(name)}_cache"
                check = self._assert_in_flat if group_id is None else self._assert_in_pool
                out[key(f"{name}_cache")] = mpk.attach_input(
                    torch_tensor=check(view, label), name=label)
        return out

    def _allocate_flat(self, device: str = "cuda"):
        """The unpaged streams as ONE allocation, keyed
        ``views[(stream_name, layer_id)][component]`` -- a layer may carry
        more than one unpaged stream."""
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
        """The unpaged twin of _assert_in_pool."""
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
                f"{name} has row stride {got}, expected {want}.")
        return tensor

    def _allocate_pool(self, entry_layouts,
                       max_num_pages: Optional[int] = None,
                       device: str = "cuda"):
        """The entire KV cache as ONE allocation, plus typed views.

        Shape: ``[num_slots, max_num_pages, target_page_bytes]``; a stream
        may carve its page into several components (K and V), component-major.

        entry_layouts: ``{stream_name: [(component_name, entry_shape, dtype),
            ...]}``. Returns ``(pool, views)``; ``views[group_id][component]``
            is shaped ``[num_slots, max_num_pages, entries_per_page,
            *entry_shape]``.
        """
        max_num_pages = self._pool_pages(max_num_pages)
        pool = torch.zeros(self.num_slots, max_num_pages,
                           self.target_page_bytes, dtype=torch.uint8,
                           device=device)
        self._pool_span = (pool.data_ptr(),
                           pool.data_ptr() + pool.numel() * pool.element_size())
        views = {}
        for g in self.groups:
            if g.stream_name is None:    # placeholder group, holds nothing
                views[g.group_id] = {}
                continue
            if g.stream_name not in entry_layouts:
                raise KeyError(
                    f"no entry layout given for stream '{g.stream_name}'")
            byte_off = 0
            comps = {}
            for cname, entry_shape, dtype in entry_layouts[g.stream_name]:
                entry_elems = 1
                for d in entry_shape:
                    entry_elems *= d
                itemsize = torch.empty(0, dtype=dtype).element_size()
                assert self.target_page_bytes % itemsize == 0, (
                    f"page of {self.target_page_bytes} B does not divide "
                    f"into {itemsize} B elements ('{g.stream_name}.{cname}')")
                assert byte_off % itemsize == 0, (
                    f"component '{g.stream_name}.{cname}' starts at byte "
                    f"{byte_off}, not a multiple of its {itemsize} B element")
                span = g.entries_per_page * entry_elems
                byte_end = byte_off + span * itemsize
                assert byte_end <= self.target_page_bytes, (
                    f"stream '{g.stream_name}' components exceed the "
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
                f"{name} has page stride {got}, expected {want}.")
        return tensor

    def elems_per_page(self, dtype) -> int:
        """Page width in elements of ``dtype`` -- the full page, not the
        view's packed entry span or the kernel's PAGE_STRIDE."""
        itemsize = torch.empty(0, dtype=dtype).element_size()
        assert self.target_page_bytes % itemsize == 0
        return self.target_page_bytes // itemsize
