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
    
def prompt_lookup_ngram_layer(
    kn_graph: KNGraph,
    input: DTensor,
    output: DTensor,
    grid_dim: tuple[int, int, int],
    block_dim: tuple[int, int, int],
):
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(input, (-1, -1, -1), -1, True)
    tb_graph.new_input(output, (-1, -1, -1), -1, True)
    kn_graph.customized([input, output], tb_graph)
    kn_graph.register_task(tb_graph, "prompt_lookup_ngram")
    
def greedy_verify_layer(spec_decode_config: SpecDecodeConfig):
    if spec_decode_config.method == "lookahead":
        return LookaheadConfig(**spec_decode_config)
    elif spec_decode_config.method == "promptlookup":
        return PromptLookupConfig(**spec_decode_config)
    else:
        raise ValueError(f"Invalid spec decode method: {spec_decode_config.method}")