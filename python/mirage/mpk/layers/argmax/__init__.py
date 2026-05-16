"""Argmax / token-selection layers."""

from .argmax import Argmax
from .argmax_partial import ArgmaxPartial
from .argmax_reduce import ArgmaxReduce

__all__ = ["Argmax", "ArgmaxPartial", "ArgmaxReduce"]
