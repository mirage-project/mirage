import torch
import torch.nn as nn
import torch.fx as fx

class MatmulBase(nn.Module):
  """
  Base model class using full-precision (float32) to perform
  one matrix multiplication
  """
  def __init__(self, in_features, out_features):
    super(MatmulBase, self).__init__()
    self.linear = nn.Linear(in_features, out_features, bias=False)
  
  def forward(self, X):
    return self.linear(X)

class LinearMirageFP8(nn.Linear):
  """
  TODO:
  Code injection scheme: 
    override nn.Linear module with this to invoke mirage hopper matmul
  """
  def __init__(self, in_features, out_features, bias=True):
    super().__init__(in_features, out_features, bias)
  
  def forward(self, x):
    M, K = x.shape
    _, N = self.weight

    # TODO: code injection point
    graph = mirage_hopper_matmul(M, N, K)

    output = graph(inputs=[x, self.weight])
    return output[0]

"""
TODO: replace nn.Linear functionality by calling mirage FP8 matmul kernel:
"""
def mirage_hopper_matmul(M, N, K):
  kn_graph = mi.new_kernel_graph()
  X = kn_graph.new_input(dims=(M, K), dtype=mi.float8_e4m3)
  W = kn_graph.new_input(dims=(K, N), dtype=mi.float8_e4m3)

  # launch 64x1x1 blocks, each running a warp group (128 threads)
  tb_graph = mi.new_threadblock_graph(grid_dim=(64,1,1), block_dim=(128,1,1), forloop_range=64, reduction_dimx=64)
  tX = tb_graph.new_input(dtensor=X, input_map=(-1,-1,-1), forloop_dim=1)
  tW = tb_graph.new_input(dtensor=W, input_map=( 1,-1,-1), forloop_dim=0)
  tM = tb_graph.matmul(tX, tW) # see hopper_matmul.cu
  tO = tb_graph.forloop_accum(tM)
  tb_graph.new_output(stensor=tO, output_map=(1, -1, -1))

  O = kn_graph.customized([X, W], tb_graph)

  kn_graph.mark_output(O)
  return kn_graph


class MugraphInjector(fx.Transformer):
  """
  torch.fx transformer that replace all builtin torch.linear with 
  our fp8 matmul mugraph
  """
  def call_module(self, target, args, kwargs):
    orig_module = self.root.get_submodule(target)
    if isinstance(orig_module, nn.Linear):
      new_layer = LinearMirageFP8(orig_module.in_features, orig_module.out_features)
      self.root.add_module(target, new_layer)
      return self.call_module(target, args, kwargs)
    return super().call_module(target, args, kwargs)