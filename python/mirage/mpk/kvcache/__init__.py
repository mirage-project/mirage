"""Public surface of the KV cache subsystem.

- kv_stream.py: the declaration vocabulary a model builder writes
  (KVMode/KVSpec/KVStream/FlatStream).
- kv_cache.py: the KVCache plan/pool object build_kv_cache() returns.
- kv_planner.py: build_kv_cache(), the fitting/grouping algorithm, KVEventLog.
"""

from .kv_stream import KVMode, KVSpec, KVStream, FlatStream
from .kv_cache import (
    KVCache,
    KVGroupConfig,
    pages_per_request,
    KV_WINDOW_TILE,
    format_bytes,
)
from .kv_planner import (
    build_kv_cache,
    plan_kv_groups,
    KVUnificationError,
    KVEventLog,
    default_kv_tile,
    resolve_kv_budget,
)

__all__ = [
    "KVMode", "KVSpec", "KVStream", "FlatStream",
    "KVCache", "KVGroupConfig", "KVUnificationError", "KVEventLog",
    "build_kv_cache", "plan_kv_groups", "pages_per_request",
    "default_kv_tile", "resolve_kv_budget", "format_bytes", "KV_WINDOW_TILE",
]
