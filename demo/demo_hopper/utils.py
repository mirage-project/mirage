
import torch
import torch.nn as nn

def model_viz(model: nn.Module):
  """
  used to check whether _
  """
  for name, module in model.named_modules():
    if isinstance(module, nn.Linear): 
        print(f"\tLayer: {name}")
        print(f"\tLayer type: {type(module)}")
        print(f"\tWeight Tensor Type: {type(module.weight)}")
        print(f"\tWeight Tensor Layout: {module.weight._layout}")  # Check Float8 layout
        print(f"\tWeight Tensor dtype: {module.weight.dtype}")  