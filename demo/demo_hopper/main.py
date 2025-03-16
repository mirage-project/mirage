import mirage as mi
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.fx as fx

"""
torchao.quantization: 
  pytorch-native QAT (quantization-aware training) and PTQ (post-training quantization)
  support. Currently use model swap, so quantize_ applies to only module but not 
  functionals.
"""
# QAT: quantization-aware training
# from torchao.float8 import convert_to_float8_training

# PTQ: post-training quantization
from torchao.quantization import (
  quantize_,  
  # Device capability 8.9+  
  float8_weight_only,                      # memory-bound model
  float8_dynamic_activation_float8_weight, # compute-bound model
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from linear import MatmulBase, LinearMirageFP8, MugraphInjector
from utils import model_viz


def inject_mugraph(base_model: nn.Module):
  # use torch.fx to trace all 
  # traced = fx.GraphModule(model, Tracer().trace(model))
  traced = fx.symbolic_trace(base_model)
  print(f"== base model fx trace:")
  traced.graph.print_tabular()

  # TODO: replace all <built-in function linear> with mugraph fp8 matmul
  # mirage_model = MugraphInjector(traced).transform()
  # return mirage_model

  return base_model

MNKConfig = {
  "M":  16  ,
  "N":  4096,
  "K":  4096,
}

if __name__ == "__main__":
  M, N, K = MNKConfig["M"], MNKConfig["N"], MNKConfig["K"]
  # input
  _X = torch.randn(M, K, dtype=torch.float32, device='cuda:0')

  # base model: weight in float32
  model = MatmulBase(N, K).cuda()
  out_0 = model(_X)

  # quantize: weight in float8_e4m3fn
  quantize_(model, float8_weight_only()) 
  print("== verify quantize_:")
  model_viz(model)
  
  # correctness check
  out_1 = model(_X)
  # TODO: quantize_ version already deviate by far (failed atol=1e-2)
  assert(torch.allclose(out_0, out_1, atol=1e-1))

  # inject matmul mugraph
  mirage_model = inject_mugraph(model)
  # out_2 = mirage_model(_X)
  # assert(torch.allclose(out_0, out_2, atol=1e-3))