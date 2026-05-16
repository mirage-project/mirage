"""Compile-time context for MPK layers.

When a model's ``compile()`` is invoked, leaf layers need access to the
``PersistentKernel`` that owns the graph being built (to register tasks,
attach weights, look up arch info, etc.). Threading ``pk`` through every
``compile()`` signature would clutter the API and force composite-module
authors to forward an argument they never use directly.

Instead the active ``PersistentKernel`` is held in a ``ContextVar`` and
read by leaves via :func:`current_pk`. The root of a compile is expected
to enter the scope with ``with pk.compile_scope():`` (see
:meth:`PersistentKernel.compile_scope`); :func:`current_pk` raises a
clear error when called outside such a scope. Nested scopes are
supported via the ``ContextVar`` token mechanism, so unit tests can
build two PKs back-to-back without leaking state.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from .persistent_kernel import PersistentKernel


_CURRENT_PK: ContextVar[Optional["PersistentKernel"]] = ContextVar(
    "mirage_mpk_current_pk", default=None
)


def current_pk() -> "PersistentKernel":
    """Return the PersistentKernel active in the enclosing compile scope.

    Raises:
        RuntimeError: if called outside a ``with pk.compile_scope():``
        block. The error message points the caller at the likely fix.
    """
    pk = _CURRENT_PK.get()
    if pk is None:
        raise RuntimeError(
            "current_pk() called outside a compile scope. Wrap your "
            "model.compile(...) in `with pk.compile_scope():` at the "
            "root of the compile."
        )
    return pk


@contextmanager
def compile_scope(pk: "PersistentKernel") -> "Iterator[PersistentKernel]":
    """Bind ``pk`` as the active PersistentKernel for the duration of the block."""
    token = _CURRENT_PK.set(pk)
    try:
        yield pk
    finally:
        _CURRENT_PK.reset(token)
