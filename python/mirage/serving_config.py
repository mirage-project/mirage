"""Serving ABI shared with CUDA, loaded from the shipped definition file.

Change persistent_kernel/serving_config.def to adjust limits or the layout.
Both source checkouts and wheels use the same definition; no generated Python
offsets or separate buffer sizes need updating.
"""
from pathlib import Path
import re

_package = Path(__file__).resolve().parent
_relative = Path("mirage/persistent_kernel/serving_config.def")
_source = _package.parents[1] / "include" / _relative
_schema = _source if _source.is_file() else _package / "include" / _relative
_values = {}
for _name, _expression in re.findall(r"^SERVING_CONSTANT\((\w+), (.+)\)$",
                                    _schema.read_text(), re.MULTILINE):
    # The trusted package schema contains integer arithmetic, shared with C++.
    _values[_name] = int(eval(_expression, {"__builtins__": {}}, _values))
globals().update(_values)


def sampling_scratch_words(vocab):
    return SCRATCH_VOCAB_ARRAYS * vocab + SCRATCH_WORKSPACE + SCRATCH_STATE_WORDS
