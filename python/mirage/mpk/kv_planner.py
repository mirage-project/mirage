"""KV cache planner: turn declared KVSpec streams into one KVCache.

Hybrid streams (different compress ratios, entry sizes, windows) share one
physical page size -- see ``plan_kv_groups()`` for how a shared
``target_page_bytes`` is picked and how each stream's block_size is fit to it.
"""

import warnings
from functools import reduce
from math import gcd
from typing import Optional, Tuple

import torch

from .kv_stream import (
    KVMode,
    KVSpec,
    KVStream,
    FlatStream,
    _merge_identical_streams,
    _check_stream_names,
)
from .kv_cache import (
    KV_WINDOW_TILE,
    KVGroupConfig,
    pages_per_request,
    format_bytes,
    KVCache,
)

__all__ = [
    "KVMode", "KVSpec", "KVStream", "FlatStream",
    "KVCache", "KVGroupConfig", "KVUnificationError", "KVEventLog",
    "build_kv_cache", "plan_kv_groups", "pages_per_request",
    "default_kv_tile", "resolve_kv_budget", "format_bytes", "KV_WINDOW_TILE",
]


def build_kv_cache(streams, *,
                   kv_budget=None,
                   max_num_pages: Optional[int] = None,
                   max_seq_length: Optional[int] = None,
                   max_num_batched_requests: int = 1,
                   max_num_batched_tokens: int = 1,
                   device: str = "cuda",
                   verbose: bool = True,
                   **plan_kwargs) -> "KVCache":
    """Declare the streams, plan the geometry, size and allocate the pool --
    the whole cache in one call.

    Give exactly one of ``kv_budget`` (bytes) or ``max_num_pages``.
    ``max_seq_length`` is required with a budget; with an explicit page count
    it only adds the one-request floor check.

    The returned plan owns the pool: ``attach(mpk, layer)`` is the only way
    to a cache tensor. Several streams may cover one layer (DeepSeek-V4's
    windowed + compressed + indexer), each namespaced by name.
    """
    streams = list(streams)
    _check_stream_names(streams)
    flat_streams = [st for st in streams if not st.paged]
    paged_streams, merged_name = _merge_identical_streams(
        [st for st in streams if st.paged])
    if kv_budget is not None and max_seq_length is None:
        raise ValueError(
            "max_seq_length is required with kv_budget: a budget is sized to "
            "hold a request of some length")
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
                f"cache has no request dimension, so two requests would write "
                f"the same rows")

    if paged_streams:
        plan = plan_kv_groups([st._spec() for st in paged_streams],
                              **plan_kwargs)
    else:
        plan = _plan_with_no_paged_streams(**plan_kwargs)
    plan._layouts = {st.name: list(st.components) for st in paged_streams}
    plan.flat_streams = tuple(st._flat(max_seq_length) for st in flat_streams)
    # attach() resolves per DECLARED stream, not per merged group: a layer may
    # carry several, each keeping its own components.
    plan._declared = tuple(streams)
    plan._merged_name = merged_name
    _resolve_pool_size(plan, kv_budget=kv_budget, max_num_pages=max_num_pages,
                       max_seq_length=max_seq_length,
                       max_num_batched_requests=max_num_batched_requests,
                       max_num_batched_tokens=max_num_batched_tokens,
                       device=_device_index(device), verbose=verbose)
    return plan._materialize(device=device)


def _device_index(device) -> int:
    if isinstance(device, int):
        return device
    dev = torch.device(device)
    if dev.type != "cuda":
        return 0
    return dev.index if dev.index is not None else torch.cuda.current_device()


class KVUnificationError(ValueError):
    """A stream does not fit the shared page size. A ValueError so demos'
    existing ``except ValueError: raise SystemExit(...)`` guard catches it
    without a separate except clause."""


def default_kv_tile(target_cc: Optional[int] = None) -> int:
    """KV tile to assume for a spec that declares none."""
    if target_cc is None:
        try:
            props = torch.cuda.get_device_properties(0)
            target_cc = props.major * 10 + props.minor
        except Exception:
            return 64
    return 64 if target_cc >= 90 else 128


def resolve_kv_budget(spec) -> int:
    """Turn a user-facing KV budget into bytes: ``"24GiB"``, ``"512MiB"``, or
    a raw int. A bare number as a string is rejected."""
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


def _resolve_pool_size(plan: "KVCache", *, kv_budget=None,
                       max_num_pages: Optional[int] = None,
                       max_seq_length: Optional[int] = None,
                       max_num_batched_requests: int = 1,
                       max_num_batched_tokens: int = 1,
                       device: int = 0, verbose: bool = True) -> int:
    """Page count to build the pool with: exactly one of ``kv_budget``/
    ``max_num_pages``. Without ``max_seq_length`` the floor check is
    skipped."""
    if (kv_budget is None) == (max_num_pages is None):
        raise ValueError("give exactly one of kv_budget / max_num_pages")

    if max_num_pages is not None:
        pages, source = max_num_pages, "explicit page count"
    elif plan.page_id_bytes == 0:
        # No paged group: the budget only has to cover the unpaged streams.
        plan.pages_for_budget(resolve_kv_budget(kv_budget))   # budget check
        pages = plan.pages_needed(max_num_batched_requests,
                                  max_seq_length or 1,
                                  max_num_batched_tokens)
        source = f"budget {kv_budget} (no paged streams)"
    else:
        pages = plan.pages_for_budget(resolve_kv_budget(kv_budget))
        source = f"budget {kv_budget}"

    if max_seq_length is not None:
        # Only one request is guaranteed to fit; more is a runtime choice.
        floor = plan.pages_needed(1, max_seq_length, max_num_batched_tokens)
        if pages < floor:
            raise ValueError(
                f"KV pool too small: {source} gives {pages} page(s), but "
                f"even one request at {max_seq_length} tokens needs {floor} "
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
                                ) -> KVCache:
    """Plan for a model with no paged KV: zero groups, zero bytes. Compiles
    at MPK_NUM_KV_GROUPS==0; device arrays that would go zero-length size
    with MPK_NUM_KV_GROUPS_ARRAY instead."""
    if target_page_bytes is not None:
        raise ValueError(
            "target_page_bytes was given but no stream is paged, so there is "
            "no page to size")
    return KVCache(target_page_bytes=0, num_slots=0, groups=())


def plan_kv_groups(
    specs,
    target_page_bytes: Optional[int] = None,
    block_size: Optional[int] = None,
    target_cc: Optional[int] = None,
) -> KVCache:
    """Turn KVSpec declarations into a KVCache.

    Give exactly one of ``target_page_bytes`` (bytes) or ``block_size``
    (tokens, default 64).
    - ``target_page_bytes``: greedy packing (``_pack_to_page_size``) -- every
      spec gets as many entries as fit, floored to its own tile.
    - ``block_size``: each spec computes its own tile-legal native size
      (``_native_fit``); the largest becomes ``target_page_bytes``, others
      scale by an exact ratio or pad to it (``_unify_to_page_size``).

    A spec that can't fit one tile raises KVUnificationError; one that fits
    but pads warns instead. Layers are then chunked into ``_group_size``-sized
    groups so every group shares one slot layout.
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
        per_spec = {s.name: _unify_to_page_size(s, native[s.name], target_page_bytes)
                    for s in specs}
    else:
        per_spec = {s.name: _pack_to_page_size(s, target_page_bytes, tiles[s.name])
                    for s in specs}
    declared = {s.name: s.block_size_multiple_of is not None for s in specs}
    group_size = _group_size([len(s.layer_ids) for s in specs])
    _warn_if_slots_starved(specs, group_size)
    _warn_if_page_wastes_bytes(specs, target_page_bytes, per_spec)

    groups = []
    for s in specs:
        spec_block_size, entries, padding = per_spec[s.name]
        layers = list(s.layer_ids)
        for start in range(0, len(layers), group_size):
            chunk = layers[start:start + group_size]
            chunk += [None] * (group_size - len(chunk))
            groups.append(KVCache.Group(
                group_id=len(groups),
                stream_name=s.name,
                layer_ids=tuple(chunk),
                block_size=spec_block_size,
                entries_per_page=entries,
                padding_bytes_per_page=padding,
                window_size=s.window_size or 0,
                tile=tiles[s.name],
                tile_declared=declared[s.name],
            ))

    return KVCache(
        target_page_bytes=target_page_bytes,
        num_slots=group_size,
        groups=tuple(groups),
    )


def _native_fit(spec: KVSpec, block_size: int, tile: int) -> Tuple[int, int]:
    """This spec's tile-legal (entries, bytes) at ``block_size`` tokens, with
    no cross-spec reconciliation. A bounded spec always returns 1 entry."""
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


def _unify_to_page_size(spec: KVSpec, native: Tuple[int, int],
                   target_page_bytes: int):
    """Reconcile ``_native_fit`` against the shared page: (block_size,
    entries, padding). Scales by an exact integer ratio if one exists, else
    keeps native size and pads, never repacked. A bounded spec always pads."""
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


def _pack_to_page_size(spec: KVSpec, target_page_bytes: int, tile: int):
    """Page capacity as (block_size, entries, padding_bytes): fits what the
    page holds, floored to a tile; leftover is padding. Always repacks. A
    bounded spec stays at 1 entry regardless."""
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
    assert block_size % tile == 0
    padding = target_page_bytes - entries * spec.per_entry_bytes
    return block_size, entries, padding


def _group_size(layer_counts):
    """Slots per group. A group with k real layers padded to S slots strands
    (S-k)/S of every page it holds, so:

    - near-equal counts (hi < 1.5 * lo): pad the smaller stream up;
    - otherwise, a usable gcd (>= lo/2): split with zero padding;
    - degenerate gcd: fall back to the smallest count and pad last chunk."""
    lo, hi = min(layer_counts), max(layer_counts)
    if hi < lo * 1.5:
        return hi
    g = reduce(gcd, layer_counts)
    if g == lo or (g > 1 and g >= lo // 2):
        return g
    return lo


def _warn_if_slots_starved(specs, group_size: int) -> None:
    """Flag a spec that fragments because a much smaller one set group_size.

    Mirrors ``_group_size``'s branches but does NOT treat ``g == lo`` as
    clean here, since a small ``lo`` (e.g. 1) makes gcd-with-1 trivially 1.
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
            f"KV plan: stream {s.name!r} ({count} layers) does not share a "
            f"clean multiple with the {group_size}-layer group size "
            f"{setter} set, so it fragments into {chunks} group(s)"
            + (f" ({padding} padded layer(s) on the last one)" if padding
               else f" with no padding, but {chunks}x the group-table and "
                    f"scheduler overhead of a single group"), stacklevel=3)


def _warn_if_page_wastes_bytes(specs, target_page_bytes: int,
                               per_spec: dict) -> None:
    """Flag a spec whose entries don't tile the shared page exactly; the
    remainder is charged as padding on every page its group ever holds."""
    for s in specs:
        _, entries, padding = per_spec[s.name]
        if padding <= 0:
            continue
        pct = 100 * padding / target_page_bytes
        warnings.warn(
            f"KV plan: stream {s.name!r} wastes {padding} B ({pct:.1f}%) of "
            f"every {target_page_bytes} B page -- its {entries} entries of "
            f"{s.per_entry_bytes} B do not tile the page exactly.", stacklevel=3)


class KVEventLog:
    """Record-and-verify instrumentation for the runtime page allocator.

    Wires a ``kv_event_log`` meta tensor into the kernel before compile;
    ``verify()`` replays it after the run and asserts allocator invariants.

    Format: log[0] = event count; event i is 4 ints at [4i+1..4i+4] =
    (type, group_id, request_slot, page_id), type 1=ALLOC 2=FREE 3=ITER
    4=MOVE. MOVE carries no group and reuses the last two fields for the
    request's old/new batch slot.
    """

    def __init__(self, pk, plan: KVCache, capacity: int = 65536,
                 device: str = "cuda"):
        self.num_groups = len(plan.groups)
        self.log = torch.zeros(capacity, dtype=torch.int32, device=device)
        pk.meta_tensors["kv_event_log"] = self.log

    def verify(self):
        return self.replay(self.log, self.num_groups)

    @staticmethod
    def replay(log: torch.Tensor, num_groups: int):
        """Replay the log; assert no double-alloc, no free of an unowned
        page, and nothing live at the end. Returns {"iterations",
        "compactions", "per_group": [{"allocs", "frees"}]}."""
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
