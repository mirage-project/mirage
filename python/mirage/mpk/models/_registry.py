"""Model registry — maps HF ``architectures[0]`` to an MPK ForCausalLM class.

Each migrated model decorates its top-level class with
``@register_model("ArchName")``. The factory
:meth:`PersistentKernel.build_from_config` looks up the architecture
field of the loaded HF config and instantiates the matching class.

This is intentionally tiny — vLLM has a comprehensive registry with
quantization variants, multimodal flags, and remote-code support;
MPK only needs the name → class map until that complexity is justified.
"""

from __future__ import annotations

from typing import Callable, Dict, Type, TypeVar

T = TypeVar("T", bound=type)

MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(arch_name: str) -> Callable[[T], T]:
    """Register a model class under its HF ``architectures[0]`` name.

    Raises ``ValueError`` on duplicate registration so silent overrides
    don't accumulate (a common source of bugs in vLLM's history).
    """
    def deco(cls: T) -> T:
        if arch_name in MODEL_REGISTRY:
            existing = MODEL_REGISTRY[arch_name]
            if existing is cls:
                return cls
            raise ValueError(
                f"register_model: architecture {arch_name!r} is already "
                f"registered to {existing.__module__}.{existing.__qualname__}; "
                f"refusing to silently override with "
                f"{cls.__module__}.{cls.__qualname__}.",
            )
        MODEL_REGISTRY[arch_name] = cls
        return cls
    return deco


def resolve_model_class(arch_name: str) -> Type:
    """Look up the MPK class for an HF architecture name.

    Raises ``ValueError`` with the list of registered names so users
    immediately see what's available (and what's missing).
    """
    if arch_name not in MODEL_REGISTRY:
        raise ValueError(
            f"resolve_model_class: architecture {arch_name!r} not "
            f"registered. Known architectures: "
            f"{sorted(MODEL_REGISTRY) or '(none)'}.",
        )
    return MODEL_REGISTRY[arch_name]
