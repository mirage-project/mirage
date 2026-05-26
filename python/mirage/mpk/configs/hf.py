"""HFConfig — wrapper around ``transformers.PretrainedConfig`` with MPK-side debug knobs.

The catalog modeling code reads architecture parameters off the
``transformers`` config object directly (``hidden_size``, ``head_dim``,
``num_hidden_layers``, …). HFConfig keeps that interface working — a
``__getattr__`` fall-through forwards unknown attribute access to the
underlying ``transformers_config`` — while adding two debug levers:

* ``num_hidden_layers_override``: clamp the layer count for fast
  iteration (e.g., compile only 2 layers when debugging a kernel bug).
  Applied to a deepcopy so the override is visible to model
  construction without mutating the user's original config.
* ``trust_remote_code``: forwarded to ``AutoConfig.from_pretrained``;
  required for some checkpoints with custom configuration code.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoConfig, PretrainedConfig


@dataclass
class HFConfig:
    model_path: str
    transformers_config: PretrainedConfig
    num_hidden_layers_override: Optional[int] = None
    trust_remote_code: bool = False

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        num_hidden_layers_override: Optional[int] = None,
        trust_remote_code: bool = False,
    ) -> "HFConfig":
        """Load a transformers config and wrap it.

        ``num_hidden_layers_override`` clamps ``num_hidden_layers`` on a
        deepcopy of the loaded config so the override is visible to
        ``model.__init__(transformers_config)`` without mutating any
        shared state.
        """
        tc = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code,
        )
        if num_hidden_layers_override is not None:
            tc = copy.deepcopy(tc)
            tc.num_hidden_layers = num_hidden_layers_override
        return cls(
            model_path=model_path,
            transformers_config=tc,
            num_hidden_layers_override=num_hidden_layers_override,
            trust_remote_code=trust_remote_code,
        )

    def __getattr__(self, name: str):
        # __getattr__ is only called when the normal lookup fails, so we
        # don't shadow declared fields. Guard against infinite recursion
        # by reading transformers_config from __dict__ rather than via
        # ``self.``; if it isn't yet set (e.g., during early
        # dataclass init or pickle restore), raise AttributeError.
        tc = self.__dict__.get("transformers_config", None)
        if tc is None:
            raise AttributeError(name)
        return getattr(tc, name)
