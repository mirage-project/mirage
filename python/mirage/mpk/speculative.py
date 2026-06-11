from dataclasses import dataclass, field
from typing import Literal, Optional, List

from ..kernel import get_key_paths, KNGraph, TBGraph
from ..core import *


@dataclass
class SpecDecodeConfig:
    method: str

@dataclass
class LookaheadConfig(SpecDecodeConfig):
    method: Literal["lookahead"] = "lookahead"
    spec_length: int = 7

@dataclass
class PromptLookupConfig(SpecDecodeConfig):
    method: Literal["promptlookup"] = "promptlookup"
    ngram_size: int = 3
    spec_length: int = 5

@dataclass
class Eagle3Config(SpecDecodeConfig):
    method: Literal["eagle3"] = "eagle3"
    spec_length: int = 5
    draft_model_path: str = (
        "lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex")
    aux_layer_ids: Optional[List[int]] = None # Default [2, num_layers//2, num_layers-3]
    use_aux_norm: bool = False
    rejection_sample_method: Literal["strict", "probabilistic"] = "strict"


def spec_decode_class(spec_decode: str,
                      ngram_size: int = 3,
                      spec_length: int = 5,
                      draft_model_path: Optional[str] = None,
                      aux_layer_ids: Optional[List[int]] = None,
                      use_aux_norm: bool = False,
                      rejection_sample_method: str = "strict"):
    if spec_decode == "lookahead":
        return LookaheadConfig(spec_length=spec_length)
    elif spec_decode == "promptlookup":
        return PromptLookupConfig(ngram_size=ngram_size, spec_length=spec_length)
    elif spec_decode == "eagle3":
        kwargs = dict(spec_length=spec_length,
                       aux_layer_ids=aux_layer_ids,
                       use_aux_norm=use_aux_norm,
                       rejection_sample_method=rejection_sample_method)
        if draft_model_path is not None:
            kwargs["draft_model_path"] = draft_model_path
        return Eagle3Config(**kwargs)
    elif spec_decode is None:
        return None
    else:
        raise NotImplementedError(f"Spec decode method {spec_decode} not implemented")
