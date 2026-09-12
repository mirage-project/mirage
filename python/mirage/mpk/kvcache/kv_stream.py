"""KV stream declarations: the vocabulary a model builder writes in.

- stream (``KVStream``): one type of cache element, declared by the model
  builder, for which layers.
- ``KVMode`` says how the planner (``kv_planner.py``) will treat a stream:
  PAGED/BOUNDED get a page table; FLAT does not.
- ``KVSpec``/``FlatStream`` are the planner's own derived forms -- a model
  builder never constructs them directly.
"""

from dataclasses import dataclass, fields, replace
from enum import Enum
from functools import reduce
from typing import Optional, Tuple

import torch


class KVMode(Enum):
    """How the planner treats a stream; its NAME never affects planning."""
    PAGED = "paged"       # grows with tokens, gets a group and a page table
    FLAT = "flat"         # read as one [capacity, width] array, no page table
    BOUNDED = "bounded"   # gets a group, but never more than 1 entry/page


@dataclass(frozen=True)
class KVSpec:
    """One KV stream in the planner's vocabulary; ``KVStream`` is the
    builder-facing form it derives from.

    compress_ratio: raw tokens folded into one entry.
    block_size_multiple_of: block-size restriction in raw tokens (the kernel's
        KV tile); None takes default_kv_tile().
    bounded: never more than 1 entry, whatever the shared page size.
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
            f"spec {self.name}: bounded and window_size are exclusive")


def _itemsize(dtype) -> int:
    return torch.empty(0, dtype=dtype).element_size()


@dataclass(frozen=True)
class KVStream:
    """One KV stream: what a page holds, for which layers.
    ``build_kv_cache`` turns it into a ``KVSpec`` (paged) or ``FlatStream``.

    A layer carrying several streams gets each namespaced by its ``name`` in
    ``attach()``; a lone stream gets bare ``k_cache``/``group_id``.
    """
    name: str
    layers: Tuple[int, ...]
    components: Tuple[Tuple[str, Tuple[int, ...], "torch.dtype"], ...]
    window: int = 0
    compress_ratio: int = 1
    block_size_multiple_of: Optional[int] = None
    kind: KVMode = KVMode.PAGED

    @property
    def paged(self) -> bool:
        """Not FLAT: gets a group, a page table, a slot in the shared pool."""
        return self.kind is not KVMode.FLAT

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
        """The unpaged form. Paging knobs must be at their defaults -- an
        unpaged stream would silently ignore them."""
        self._check_components()
        for field_name, value in (("window", self.window),
                                  ("compress_ratio", self.compress_ratio),
                                  ("block_size_multiple_of",
                                   self.block_size_multiple_of)):
            default = KVStream.__dataclass_fields__[field_name].default
            if value != default:
                raise ValueError(
                    f"stream '{self.name}' is kind=KVMode.FLAT but sets "
                    f"{field_name}={value!r}, which describes a page it does "
                    f"not have")
        return FlatStream(name=self.name, layers=tuple(self.layers),
                          components=tuple(tuple(c) for c in self.components),
                          capacity=capacity)

    def _spec(self) -> KVSpec:
        assert self.paged, f"stream {self.name}: not paged, use _flat()"
        self._check_components()
        if self.kind is KVMode.BOUNDED and self.window:
            raise ValueError(
                f"stream '{self.name}' is kind=KVMode.BOUNDED but sets "
                f"window={self.window!r}; the two are exclusive")
        return KVSpec(name=self.name,
                      per_entry_bytes=self.per_entry_bytes,
                      layer_ids=tuple(self.layers),
                      compress_ratio=self.compress_ratio,
                      window_size=self.window or None,
                      block_size_multiple_of=self.block_size_multiple_of,
                      bounded=self.kind is KVMode.BOUNDED)


def _hashable(v):
    """A dict key from a declared field: ``components`` arrives as a list,
    with tuples inside it."""
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    return v


def _merge_identical_streams(streams):
    """Fold streams that lay a page out identically into one stream.

    Keyed on every KVStream field describing the PAGE, so a later-added field
    is included by default. Returns ``(streams, {declared: merged_name})``.
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
    """A stream read as one contiguous ``[capacity, width]`` array.

    Outside the pool -- a constant-stride reader can't take pages from a
    shared free list -- but still owned and budgeted by the plan.
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
    """Unique names: a repeat loses a layout (keyed by name) and collides in
    attach()'s namespacing."""
    seen = set()
    for st in streams:
        if st.name in seen:
            raise ValueError(f"two streams are both named {st.name!r}")
        seen.add(st.name)
