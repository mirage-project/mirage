import torch
from op import Operator
import torch.nn.functional as F
from torch.onnx import _constants as C
import onnxscript
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args, _maybe_get_const


@parse_args('v', 'is', 'v', 'f')
def symbolic_rms_norm(g: torch._C.Graph,
                               input_val: torch._C.Value,
                               normalized_shape: list[int], # From 'is'
                               weight_val: torch._C.Value, # From 'v', could be None if affine=False
                               eps: float):                # From 'f'
    """
    Symbolic function mapping aten::rms_norm to mirage.custom::RMSNorm.

    ONNX Operator Definition:
    -------------------------
    OpType: RMSNorm
    Domain: mirage.custom
    Inputs:
        data (Tensor): Input tensor to normalize.
        weight (Tensor, Optional): Gamma weights for elementwise affine transform.
                                    Absence implies gamma = 1.
    Outputs:
        output (Tensor): Normalized tensor.
    Attributes:
        normalized_shape (List[int]): Dimensions over which to compute RMS. (Maps from PyTorch normalized_shape)
        epsilon (float): Value added to the denominator for stability. (Maps from PyTorch eps)
    """
    # Determine if elementwise affine is enabled by checking if weight_val is present
    # _maybe_get_const helps determine if weight_val represents a None constant
    is_affine = _maybe_get_const(weight_val, 'v') is not None

    op_inputs = [input_val]
    if is_affine:
        op_inputs.append(weight_val)
    
    output_val = g.op("mirage.custom::RMSNorm",
                        *op_inputs,
                        normalized_shape_i=normalized_shape,
                        epsilon_f=eps)
    input_type = input_val.type()
    output_val.setType(input_type)

    return output_val


def register_rms_norm():
    opset_version = 11 
    aten_op_name = "aten::rms_norm"
    custom_domain = "mirage.custom"

    try:
        register_custom_op_symbolic(f"{aten_op_name}", symbolic_rms_norm, opset_version)
        # print(f"Registered custom symbolic function '{symbolic_rms_norm.__name__}' for '{aten_op_name}' (opset {opset_version})")
    except RuntimeError as e:
        if "is already registered" in str(e):
            return
            # print(f"Custom symbolic function for '{aten_op_name}' (opset {opset_version}) already registered.")
        else:
            raise e


def register_custom_operators():
    register_rms_norm()