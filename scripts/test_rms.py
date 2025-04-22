import torch
import torch.nn as nn
import torch.onnx
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args, _maybe_get_const
from build_computation_graph import get_computation_graph

class SimpleRMSNormModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, output_dim=10, affine=True, eps=1e-6):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.rms_norm = nn.RMSNorm(hidden_dim, eps=eps, elementwise_affine=affine)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.rms_norm(x)
        x = self.linear2(x)
        return x

model_affine = SimpleRMSNormModel(affine=True, eps=1e-5) # Example different eps
# export_model(model_affine, "scripts/onnx/rms_custom.onnx", opset=opset_version)
# model_no_affine = SimpleRMSNormModel(affine=False, eps=1e-6)
# export_model(model_no_affine, "scripts/onnx/rms_custom_false.onnx", opset=opset_version)

dummy_input = torch.randn(1, 64) # Batch size 1, input_dim 64
operators = get_computation_graph(model_affine, dummy_input, {}, "onnx")
# operators2 = get_computation_graph(model_no_affine, dummy_input, {}, "onnx")