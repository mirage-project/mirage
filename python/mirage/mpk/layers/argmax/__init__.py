"""Argmax / token-selection layers."""

from .argmax_partial import ArgmaxPartial
from .argmax_reduce import ArgmaxReduce

__all__ = [
    "ArgmaxPartial",
    "ArgmaxReduce",
]
