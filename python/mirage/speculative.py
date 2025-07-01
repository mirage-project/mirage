from dataclasses import dataclass
from typing import Literal

from .kernel import get_key_paths, KNGraph, TBGraph
from .core import *


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
    
def spec_decode_class(spec_decode: str,
                      ngram_size: int,
                      spec_length: int):
    if spec_decode == "lookahead":
        return LookaheadConfig(spec_length=spec_length)
    elif spec_decode == "promptlookup":
        return PromptLookupConfig(ngram_size=ngram_size, spec_length=spec_length)
    elif spec_decode is None:
        return None
    else:
        raise NotImplementedError(f"Spec decode method {spec_decode} not implemented")