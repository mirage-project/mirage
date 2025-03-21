
import torch
import torch.nn as nn

def model_viz(model: nn.Module):
  """
  used to check whether _
  """
  for name, module in model.named_modules():
    if isinstance(module, nn.Linear):  # Only check Linear layers
        print(f"\tLayer: {name}")
        print(f"\tLayer type: {type(module)}")
        print(f"\tWeight Tensor Type: {type(module.weight)}")
        print(f"\tWeight Tensor Layout: {module.weight._layout}")  # Check Float8 layout
        # TODO: for Xinhao: currently this prints float32, please check if this is float8_e4m3fn on H100
        print(f"\tWeight Tensor dtype: {module.weight.dtype}")  