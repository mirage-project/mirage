"""Back-compat shim — ``ParallelConfig`` now lives in :mod:`mirage.mpk.configs.parallel`.

Importing ``from mirage.mpk.parallel import ParallelConfig`` continues to
work (every internal callsite + every external user still uses this
path), but new code should prefer
``from mirage.mpk.configs import ParallelConfig``.
"""

from .configs.parallel import ParallelConfig  # noqa: F401

__all__ = ["ParallelConfig"]
