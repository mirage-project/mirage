"""Re-export of :class:`mirage.mpk.speculative.SpecDecodeConfig`.

Keeps the speculative-decode config alongside the other MPKConfig
sub-configs so users only need to import from ``mirage.mpk.configs``.
The underlying definition (and its concrete subclasses
``LookaheadConfig`` / ``PromptLookupConfig``) still lives in
:mod:`mirage.mpk.speculative` — moving it would break the legacy
demos that import from there.
"""

from ..speculative import (
    LookaheadConfig,
    PromptLookupConfig,
    SpecDecodeConfig,
)

__all__ = ["SpecDecodeConfig", "LookaheadConfig", "PromptLookupConfig"]
