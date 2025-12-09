"""Build and parse computational graphs from neural network models.

This module provides functionality to convert PyTorch models into computational graphs
by exporting to ONNX format and parsing the resulting graph structure. It supports:

- Parsing ONNX models into operator graphs with shape information
- Expanding high-level operations (Softmax, ReduceMean, Sigmoid, Neg) into primitives
- Inserting broadcast operations for shape compatibility
- Building computation graphs for model partitioning and optimization

Main functions:
    - get_computation_graph: Convert a model to a computational graph
    - parse_onnx_model: Parse ONNX representation into operator graph
    - print_computational_graph: Visualize graph structure
"""

import torch
from torch import nn
import pickle
import os
import onnx
from onnx import shape_inference
from partitioning.op import Operator
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SimpleClassifierMix(nn.Module):
    """Simple neural network classifier with mixed supported and unsupported operations."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=2):
        super(SimpleClassifierMix, self).__init__()
        
        # Supported
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.linear2 = nn.Linear(hidden_size, num_classes)  
        
        # Unsupported
        self.dropout = nn.Dropout(0.2)  
        
    def forward(self, x):
        """Forward pass through the classifier."""
        x = self.linear1(x)  
        x = F.relu(x)  
        x = self.dropout(x)  
        x = self.linear2(x)          
        # Unsupported activation
        x = torch.tanh(x) 
        
        return x
    
class SplitModel(nn.Module):
    """Neural network that splits input and processes through parallel paths."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SplitModel, self).__init__()
        self.fc1 = nn.Linear(input_dim // 2, hidden_dim)
        self.fc2 = nn.Linear(input_dim // 2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        """Forward pass that splits input, processes in parallel, and concatenates results."""
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        x = torch.cat((x1, x2), dim=1)

        x1 = torch.relu(self.fc1(x1))
        x2 = torch.relu(self.fc2(x2))
        
        x = torch.cat((x1, x2), dim=1)
        x = self.fc_out(x)
        return x

def print_computational_graph(root_node, indent=0, visited=None):
    """Print a hierarchical representation of the computational graph."""
    if visited is None:
        visited = set()
    
    if id(root_node) in visited:
        return
    visited.add(id(root_node))
    
    prefix = "  " * indent
    print(f"{prefix} ||| Operation : {root_node.name} |||")
    
    print(root_node.__dict__)
    for attribute_name, attribute_value in root_node.__dict__.items():
        print(f"Attribute: {attribute_name}, Type: {type(attribute_value)}")


    if root_node.input_tensor_shapes:
        print(f"{prefix}  Input shapes: {root_node.input_tensor_shapes}")
    
    if root_node.output_tensor_shapes:
        print(f"{prefix}  Output shapes: {root_node.output_tensor_shapes}")
    
    if root_node.input_ops:
        print(f"{prefix}  Input operations:")
        for op in root_node.input_ops:
            print(f"{prefix}    - {op.name}")
    
    if root_node.output_ops:
        print(f"{prefix}  Output operations:")
        for op in root_node.output_ops:
            print(f"{prefix}    - {op.name}")
            print_computational_graph(op, indent + 2, visited)

    
    if root_node.input_ops:
        print(f"{prefix}  Subgraph:")
        # for op in root_node.input_ops:
        #     print_computational_graph(op, indent + 2, visited)

def parse_onnx_model(model, unique_operators):
    """Parse ONNX model and construct operator graph with shape information."""
    shape_value_dict = {}
    for initializer in model.graph.initializer:
        shape_value_dict[initializer.name] = tuple(initializer.dims)
    
    for value in model.graph.value_info:
        shape_value_dict[value.name] = tuple([d_i.dim_value for d_i in value.type.tensor_type.shape.dim])

    for input_info in model.graph.input:
        shape_value_dict[input_info.name] = tuple([d_i.dim_value for d_i in input_info.type.tensor_type.shape.dim])
    
    for output_info in model.graph.output:
        shape_value_dict[output_info.name] = tuple([d_i.dim_value for d_i in output_info.type.tensor_type.shape.dim])

    tensor_producers = {}
    tensor_consumers = {}
    tensor_id = {}
    # store the tensor producers and consumers in a dict
    for node in model.graph.node:
        node_name = node.name if node.name else f"{node.op_type}_{id(node)}"

        for output_name in node.output:
            tensor_producers[output_name] = node_name
            if output_name not in tensor_id:
                tensor_id[output_name] = len(tensor_id)+1
        for input_name in node.input:
            if input_name not in tensor_consumers:
                tensor_consumers[input_name] = []
            tensor_consumers[input_name].append(node_name)
            if input_name not in tensor_id:
                tensor_id[input_name] = len(tensor_id)+1
    
    operators = {}

    # store the operators in a dict
    for node in model.graph.node:
        op_type = node.op_type
        if op_type not in unique_operators:
            unique_operators[op_type] = 1
        else:
            unique_operators[op_type] += 1
        node_name = node.name or f"{op_type}_{id(node)}"
        kwargs = {}

        # TODO: fix specific operator attributes
        # if node.op_type == "Clip":
        #     # Using .item() here pulls out the value from the rank-0 tensor
        #     if node.min != None:
        #         kwargs["min"] = node.min.item()
        #     else: # Defaults are to do nothing but could be changed
        #         kwargs["min"] = float('-inf')
        #     if node.max!= None:
        #         kwargs["max"] = node.max.item()
        #     else:
        #         kwargs["max"] = float('inf')
        # if node.op_type == "ReduceSum":
        #     # TODO add ReduceMean here as well?
        #     if node.axis != None:
        #         axis = node.axis[0]
        #         kwargs["axis"] = axis
        #     else:
        #         kwargs["axis"] = 0
        if node.op_type == "Gather":
            kwargs["axis"] = node.attribute[0].i
        elif node.op_type == "Cast" or node.op_type == "CastLike":
            # data type to cast to https://github.com/dmlc/tensorboard/blob/master/tensorboard/src/onnx.proto?utm_source=chatgpt.com
            if node.attribute:
                kwargs["to"] = node.attribute[0].i
        elif node.op_type == "Constant":
            attr = node.attribute[0]
            if attr.type == onnx.AttributeProto.TENSOR:
                value = onnx.numpy_helper.to_array(attr.t)
            elif attr.type == onnx.AttributeProto.FLOAT:
                value = attr.f
            elif attr.type == onnx.AttributeProto.INT:
                value = attr.i
            elif attr.type == onnx.AttributeProto.FLOATS:
                value = list(attr.floats)
            elif attr.type == onnx.AttributeProto.INTS:
                value = list(attr.ints)
            else:
                raise TypeError(f"Unsupported constant attribute type: {attr.type}")
            # convert scalar or array to torch.Tensor
            value = torch.as_tensor(value, dtype=torch.float16, device="cuda")
            kwargs["t"] = value
        elif node.op_type == "Transpose":
            kwargs["perm"] = list(node.attribute[0].ints)

        input_tensor_shapes = []
        for i, input_name in enumerate(node.input):
            if input_name in shape_value_dict and input_name in tensor_id:
                shape = shape_value_dict[input_name]

                # modifications for specific operators
                if i == 1:
                    if op_type == "ReduceMean":
                        # the second tensor of ReduceMean is a 1x1 tensor with the reduction dimension
                        # TODO for now, we assume the last dimension is the reduction dimension always
                        continue
                    elif op_type == "MatMul":
                        if 0 in shape:
                            shape = tuple()
                        elif (len(input_tensor_shapes[0][0]) - len(shape)) == 1:
                            # batch dimension must be the same
                            shape = (input_tensor_shapes[0][0][0],) + shape
                
                input_tensor_shapes.append((shape, tensor_id[input_name]))
        output_tensor_shapes = []
        for output_name in node.output:
            if output_name in shape_value_dict and output_name in tensor_id:
                shape = shape_value_dict[output_name]

                output_tensor_shapes.append((shape, tensor_id[output_name]))

        operator = Operator(name=node_name, fn=op_type, input_ops=[], output_ops=[], input_tensor_shapes=input_tensor_shapes, output_tensor_shapes=output_tensor_shapes, kwargs=kwargs)
        operators[node_name] = operator
        
    # Connect the input ops
    for node in model.graph.node:
        node_name = node.name or f"{node.op_type}_{id(node)}"

        for input_name in node.input:
            if input_name in tensor_producers:
                tensor_producer_name = tensor_producers[input_name]
                if tensor_producer_name in operators:
                    operators[node_name].input_ops.append(operators[tensor_producer_name])

            else:
                dummy_const_operator = Operator(name=input_name, fn="Constant")
                operators[node.name].input_ops.append(dummy_const_operator)
    
    # Filling in output ops
    for node in model.graph.node:
        node_name = node.name or f"{node.op_type}_{id(node)}"
        for output_name in node.output:
            if output_name in tensor_consumers:
                for consumer_name in tensor_consumers[output_name]: # consider all the consumers of a tensor
                    operators[node_name].output_ops.append(operators[consumer_name])

            else:
                dummy_const_operator = Operator(name=output_name, fn="Constant")
                operators[node.name].output_ops.append(dummy_const_operator)
    
    # pass to expand softmax to constituent components
    def expand_softmax_pass(op):
        """Expand Softmax operation into Exp, ReduceSum, and Div operations."""
        if op.fn != "Softmax":
            return

        # exp op
        op_id = op.name.split('_')[-1]
        exp_op = Operator(name=f"node_Exp_{op_id}_softmax", 
                          fn="Exp",
                          input_ops=[op.input_ops[0]],
                          input_tensor_shapes=[op.input_tensor_shapes[0]])
        
        # input ops
        for inp in op.input_ops:
            inp.output_ops.remove(op)
            inp.output_ops.append(exp_op)
        exp_out_id = len(tensor_id) + 1
        tensor_id[f"exp_out_{exp_out_id}"] = exp_out_id
        exp_op.output_tensor_shapes = [(op.input_tensor_shapes[0][0], exp_out_id)]

        # reduction op
        input_shape = op.input_tensor_shapes[0][0]
        reduce_dim = len(input_shape) - 1 if len(input_shape) > 0 else 0
        
        red_op = Operator(name=f"node_ReduceSum_{op_id}_softmax",
                          fn="ReduceSum",
                          input_ops=[exp_op],
                          input_tensor_shapes=[exp_op.output_tensor_shapes[0]],
                          kwargs={'dim': reduce_dim})
        red_out_id = len(tensor_id) + 1
        tensor_id[f"red_out_{red_out_id}"] = red_out_id
        red_op.output_tensor_shapes = [(exp_op.output_tensor_shapes[0][0], red_out_id)]
        exp_op.output_ops = [red_op]

        # division op
        div_op = Operator(name=f"node_Div_{op_id}_softmax",
                          fn="Div",
                          input_ops=[exp_op, red_op],
                          output_ops=op.output_ops,
                          input_tensor_shapes=[exp_op.output_tensor_shapes[0], red_op.output_tensor_shapes[0]],
                          output_tensor_shapes=op.output_tensor_shapes)
        exp_op.output_ops = [div_op]

        # output ops
        for out in op.output_ops:
            if op in out.input_ops:
                out.input_ops.remove(op)
            out.input_ops.append(div_op)
        
        # add to operators
        operators[exp_op.name] = exp_op
        operators[red_op.name] = red_op
        operators[div_op.name] = div_op

        # delete from operators
        del operators[op.name]

    # pass to expand reduce mean to constituent components
    def expand_reduce_mean_pass(op):
        """Expand ReduceMean operation into ReduceSum and Div operations."""
        if op.fn != "ReduceMean":   
            return

        # sum op
        op_id = op.name.split('_')[-1]
        input_shape = op.input_tensor_shapes[0][0]
        reduce_dim = len(input_shape) - 1 if len(input_shape) > 0 else 0
        
        sum_op = Operator(name=f"node_ReduceSum_{op_id}_reducemean", 
                          fn="ReduceSum",
                          input_ops=[op.input_ops[0]],
                          input_tensor_shapes=[op.input_tensor_shapes[0]],
                          kwargs={'dim': reduce_dim}) # pass axis parameter

        # input ops
        for inp in op.input_ops:
            if op in inp.output_ops:
                inp.output_ops.remove(op)
            inp.output_ops.append(sum_op)
        sum_out_id = len(tensor_id) + 1
        tensor_id[f"sum_out_{sum_out_id}"] = sum_out_id
        sum_op.output_tensor_shapes = [(op.output_tensor_shapes[0][0], sum_out_id)]

        # division op
        div_op = Operator(name=f"node_Div_{op_id}_reducemean",
                          fn="Div",
                          input_ops=[sum_op],
                          input_tensor_shapes=[sum_op.output_tensor_shapes[0]],
                          output_ops=op.output_ops,
                          output_tensor_shapes=op.output_tensor_shapes,
                          additional_params={"arg1": np.prod([dim for dim in sum_op.input_tensor_shapes[0][0]]) / np.prod([dim for dim in sum_op.output_tensor_shapes[0][0]])}) # pass the constant divisor
        sum_op.output_ops = [div_op]

        # output ops
        for out in op.output_ops:
            if op in out.input_ops:
                if op in out.input_ops:
                    out.input_ops.remove(op)
            out.input_ops.append(div_op)
        
        # add to operators
        operators[sum_op.name] = sum_op
        operators[div_op.name] = div_op

        # delete from operators
        del operators[op.name]
    
    def expand_sigmoid_pass(op):
        """Expand Sigmoid operation into Exp, Add, and Div operations."""
        if op.fn != "Sigmoid":
            return
        
        # exp op
        op_id = op.name.split('_')[-1]
        exp_op = Operator(name=f"node_Exp_{op_id}_sigmoid", 
                          fn="Exp",
                          input_ops=[op.input_ops[0]],
                          input_tensor_shapes=[op.input_tensor_shapes[0]])
        
        # input ops
        for inp in op.input_ops:
            inp.output_ops.remove(op)
            inp.output_ops.append(exp_op)
        exp_out_id = len(tensor_id) + 1
        tensor_id[f"exp_out_{exp_out_id}"] = exp_out_id
        exp_op.output_tensor_shapes = [(op.input_tensor_shapes[0][0], exp_out_id)]

        # addition op
        add_op = Operator(name=f"node_Add_{op_id}_sigmoid",
                          fn="Add",
                          input_ops=[exp_op],
                          input_tensor_shapes=[exp_op.output_tensor_shapes[0]],
                          additional_params={"arg1": 1.0})
        add_out_id = len(tensor_id) + 1
        tensor_id[f"add_out_{add_out_id}"] = add_out_id
        add_op.output_tensor_shapes = [(exp_op.output_tensor_shapes[0][0], add_out_id)]
        
        # division op
        div_op = Operator(name=f"node_Div_{op_id}_sigmoid",
                          fn="Div",
                          input_ops=[exp_op, add_op],
                          output_ops=op.output_ops,
                          input_tensor_shapes=[exp_op.output_tensor_shapes[0], add_op.output_tensor_shapes[0]],
                          output_tensor_shapes=op.output_tensor_shapes)
        exp_op.output_ops = [add_op, div_op]  # exp_op feeds both add_op and div_op
        add_op.output_ops = [div_op]

        # output ops
        for out in op.output_ops:
            if op in out.input_ops:
                out.input_ops.remove(op)
            out.input_ops.append(div_op)
        
        # add to operators
        operators[exp_op.name] = exp_op
        operators[add_op.name] = add_op
        operators[div_op.name] = div_op

        # delete from operators
        del operators[op.name]
    
    def expand_neg_pass(op):
        """Expand Neg operation into Mul operation with -1."""
        if op.fn != "Neg":
            return
        
        # multiplication op
        op_id = op.name.split('_')[-1]
        mul_op = Operator(name=f"node_Mul_{op_id}_neg", 
                          fn="Mul",
                          input_ops=[op.input_ops[0]],
                          input_tensor_shapes=[op.input_tensor_shapes[0]],
                          additional_params={"arg1": -1.0})
        
        # input ops
        for inp in op.input_ops:
            inp.output_ops.remove(op)
            inp.output_ops.append(mul_op)
        mul_out_id = len(tensor_id) + 1
        tensor_id[f"mul_out_{mul_out_id}"] = mul_out_id
        mul_op.output_tensor_shapes = [(op.input_tensor_shapes[0][0], mul_out_id)]
        mul_op.output_ops = op.output_ops

        # output ops
        for out in op.output_ops:
            if op in out.input_ops:
                out.input_ops.remove(op)
            out.input_ops.append(mul_op)
        
        # add to operators
        operators[mul_op.name] = mul_op

        # delete from operators
        del operators[op.name]
    
    def expand_reciprocal_pass(op):
        """Add reciprocal parameter (1.0) to Reciprocal operations."""
        if op.fn != "Reciprocal":
            return
        op.additional_params = {"arg0": 1.0}
    
    def insert_broadcast_pass(op):
        """Insert Expand operations to broadcast tensors with mismatched shapes."""
        if op.fn in ["MatMul", "Gemm", "ReduceSum", "Expand", "Gather", "Transpose", "Unsqueeze", "Reshape"]:
            return
        for i, (shape, _) in enumerate(op.input_tensor_shapes):
            if shape != op.output_tensor_shapes[0][0]:
                # need to insert a broadcast
                op_id = op.name.split('_')[-1]
                broadcast_op = Operator(name=f"node_Expand_{op_id}_{i}",
                                        fn="Expand",
                                        input_ops=[op.input_ops[i]],
                                        input_tensor_shapes=[op.input_tensor_shapes[i]],
                                        output_tensor_shapes=[op.output_tensor_shapes[0]])
                
                # input ops
                inp = op.input_ops[i]
                if op in broadcast_op.input_ops:
                    inp.output_ops.remove(op)
                inp.output_ops.append(broadcast_op)
                broadcast_out_id = len(tensor_id) + 1
                tensor_id[f"broadcast_out_{broadcast_out_id}"] = broadcast_out_id
                broadcast_op.output_tensor_shapes = [(op.output_tensor_shapes[0][0], broadcast_out_id)]

                # connect to current op
                op.input_ops[i] = broadcast_op
                op.input_tensor_shapes[i] = broadcast_op.output_tensor_shapes[0]
                broadcast_op.output_ops = [op]

                # add to operators
                operators[broadcast_op.name] = broadcast_op
    
    for op in list(operators.values()):
        expand_softmax_pass(op)
        expand_reduce_mean_pass(op)
        expand_sigmoid_pass(op)
        expand_neg_pass(op)
        expand_reciprocal_pass(op)
        insert_broadcast_pass(op)

    tensor_id_to_name = {v: k for k, v in tensor_id.items()}
    return operators, tensor_id_to_name

def test_cfg():
    """Test computation graph building with sample models."""
    model = SimpleClassifierMix()
    model.to(device)
    batch_size = 2
    input_size = 784  
    dummy_input = torch.randn(batch_size, input_size, device=device)

    torch.onnx.export(
        model,    
        dummy_input,              
        "scripts/onnx/my_model.onnx", 
        input_names=["input"], 
        dynamo=True                  
    )

    model = SplitModel(input_dim=8, hidden_dim=16, output_dim=4)
    x = torch.randn(5, 8)  # Batch of 5 samples with 8 features each
    output = model(x)
    print(output.shape)  # Expected output shape: (5, 4)

    input_dim = 8
    hidden_dim = 16
    output_dim = 4
    model = SplitModel(input_dim, hidden_dim, output_dim)

    model.eval()

    dummy_input = torch.randn(1, input_dim)  # Batch size 1, feature size 8

    """Trying using a smaller model"""
    model = onnx.load('scripts/onnx/my_model.onnx')
    # model = onnx.load('scripts/onnx/split_model.onnx')

    parse_onnx_model(model)

def get_computation_graph(model, dummy_input, unique_operators, method):
    """Build computation graph from model using specified method (currently supports 'onnx')."""
    match method:
        case "onnx":
            # Generate the ONNX file
            # custom_onnx_operators.register_custom_operators() # Register any custom operators we have defined. eg: RMSNorm, etc.
            
            onnx_path = "scripts/onnx/integrate_test.onnx"
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                dynamo=True
            )
            
            shape_inference.infer_shapes_path(model_path="scripts/onnx/integrate_test.onnx",
                                              output_path="scripts/onnx/inferred_model.onnx") # for shape inference of inputs and outputs
            inferred_model = onnx.load("scripts/onnx/inferred_model.onnx")
            operators, tensor_id_to_name = parse_onnx_model(inferred_model, unique_operators)

            return operators, tensor_id_to_name
        case _:
            print("Unsupported method for build_graph")
            return None
