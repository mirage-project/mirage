import torch
from itertools import combinations as comb
import time
import mirage as mi
from op import Operator
from build_computation_graph import get_computation_graph

# ids_to_nodes = {}

# def build_computational_graph(op_node, unique_operators):
#     unique_operators.add(op_node.name)
#     if op_node.input_ops != []:
#         return
    
#     local_scope = {}
#     exec(
# f"""
# def hook_fn(inputs, outputs):
#     if {id(op_node.fn)} in ids_to_nodes:
#         op_node = ids_to_nodes[{id(op_node.fn)}]
#         op_node.input_tensor_shapes = [(input.shape, id(input)) if input != None else None for input in inputs]
#         op_node.output_tensor_shapes = [(output.shape, id(output)) if output != None else None for output in outputs]
# """,
#     globals(),
#     local_scope
#     )

#     op_node.fn.register_hook(local_scope["hook_fn"])

#     if id(op_node.fn) not in ids_to_nodes:
#         ids_to_nodes[id(op_node.fn)] = op_node

#     for next_fn in op_node.fn.next_functions:
#         if id(next_fn[0]) in ids_to_nodes:
#             ids_to_nodes[id(next_fn[0])].output_ops.append(op_node)
#         else:
#             if next_fn[0] == None:
#                 continue
#             ids_to_nodes[id(next_fn[0])] = Operator(name=next_fn[0].name(), fn=next_fn[0], input_ops=[], output_ops=[op_node])
#         op_node.input_ops.append(ids_to_nodes[id(next_fn[0])])
        
#         build_computational_graph(ids_to_nodes[id(next_fn[0])], unique_operators)

def copy_subgraph(subgraph):
    new_subgraph = {}
    for from_op, to_ops in subgraph.items():
        new_subgraph[from_op] = to_ops.copy()
    return new_subgraph

def get_partitions(op_node, min_num_ops, max_num_ops, visited_start_nodes, all_subgraphs, UNSUPPORTED_OPS, IGNORE_OPS):
    if id(op_node.name) not in visited_start_nodes:
        visited_start_nodes.add(id(op_node.name))
        if op_node.fn not in UNSUPPORTED_OPS.union(IGNORE_OPS):
            # handle non-matching shapes
            op_needs_broadcast = False
            input_dims = len(op_node.input_tensor_shapes[0][0])
            for s in op_node.input_tensor_shapes:
                if len(s[0]) != input_dims:
                    op_needs_broadcast = True
                    break
            if not op_needs_broadcast:
                get_partitions_helper(op_node, {op_node: []}, min_num_ops, max_num_ops, set(), all_subgraphs, UNSUPPORTED_OPS, IGNORE_OPS)
            
        for output_node in op_node.output_ops:
            if id(output_node.name) not in visited_start_nodes:
                get_partitions(output_node, min_num_ops, max_num_ops, visited_start_nodes, all_subgraphs, UNSUPPORTED_OPS, IGNORE_OPS)

def get_partitions_helper(op_node, curr_subgraph, min_num_ops, max_num_ops, visited, all_subgraphs, UNSUPPORTED_OPS, IGNORE_OPS):
    if id(op_node.name) in visited:
        return
    if len(curr_subgraph) > max_num_ops:
        return
    
    # assume op_node already in curr_subgraph
    visited.add(id(op_node.name))
    if len(curr_subgraph) >= min_num_ops:
        all_subgraphs.append(copy_subgraph(curr_subgraph))

    valid_output_ops = []
    for output_op in op_node.output_ops:
        # handle non-matching shapes
        output_op_needs_broadcast = False
        input_dims = len(output_op.input_tensor_shapes[0][0])
        for s in output_op.input_tensor_shapes:
            if len(s[0]) != input_dims:
                output_op_needs_broadcast = True
                break
        if output_op_needs_broadcast:
            continue

        # handle IGNORE_OPS
        ignore_op_is_last_op = False
        while output_op.fn in IGNORE_OPS:
            if len(output_op.output_ops) == 0:
                ignore_op_is_last_op = True
                break
            assert len(output_op.output_ops) == 1
            output_op = output_op.output_ops[0]
        if ignore_op_is_last_op:
            continue
        
        # handle UNSUPPORTED_OPS
        if output_op.fn not in UNSUPPORTED_OPS:
            valid_output_ops.append(output_op)
    
    for choose_k in range(1, len(valid_output_ops) + 1):
        curr_subgraph_copy = copy_subgraph(curr_subgraph)
        for comb_outputs in comb(valid_output_ops, choose_k):
            visited_copy = visited.copy()
            for output_node in comb_outputs:
                if output_node not in curr_subgraph_copy:
                    curr_subgraph_copy[output_node] = []
                curr_subgraph_copy[op_node].append(output_node)
                get_partitions_helper(output_node, copy_subgraph(curr_subgraph_copy), min_num_ops, max_num_ops, visited_copy, all_subgraphs, UNSUPPORTED_OPS, IGNORE_OPS)

def partition_graph(model, 
                    dummy_input, 
                    min_num_ops=2, 
                    max_num_ops=4, 
                    UNSUPPORTED_OPS=set(), # these are operators not supported by Mirage
                    IGNORE_OPS=set()): # these are operators that performs no operations on the tensors
    unique_operators = {}
    operators = get_computation_graph(model, dummy_input, unique_operators, "onnx")

    all_subgraphs = []
    visited_start_nodes = set()
    for _, op_node in operators.items():
        get_partitions(op_node, min_num_ops, max_num_ops, visited_start_nodes, all_subgraphs, UNSUPPORTED_OPS, IGNORE_OPS)

    return all_subgraphs, unique_operators

# TODO: add support for reduction, clamp, rms_norm. These rely on additional
# inputs that the Operator class doesn't currently support
def function_map(graph, func, inputs, kwargs={}):
    match func.fn:
        case "MatMul": return graph.matmul(*inputs)
        case "ReduceSum": return graph.reduction(*inputs)
        case "Exp": return graph.exp(*inputs)
        case "Gelu": return graph.gelu(*inputs)
        case "Relu": return graph.relu(*inputs)
        case "Clip": return graph.clamp(*inputs, **kwargs)
        case "Add": return graph.add(*inputs)
        case "Mul": return graph.mul(*inputs)
        case "Div": return graph.div(*inputs)
        # case "Sqrt": return graph.sqrt(*inputs)
        case "Reciprocal": return graph.div(*inputs)
        case "Softmax": # In case of softmax, inputs must be of form (mat, axis)
            exp = graph.exp(inputs[0])
            summed = graph.reduction(exp, inputs[1])
            return graph.div(exp, summed)
        case "Sigmoid":
            matrix = inputs[0]
            ones = inputs[1]
            neg_ones = inputs[2]
            neg_mat = graph.mul(neg_ones, matrix)
            neg_exp = graph.exp(neg_mat)
            summed = graph.add(neg_exp, ones)
            return graph.div(ones, summed)
        case "Square":
            matrix = inputs[0]
            return graph.mul(matrix, matrix)
        case "Neg": 
            return graph.mul(*inputs)
        case "RMSNormalization": return graph.rms_norm(*inputs, **kwargs)
        case _: 
            raise NotImplementedError

# Take in an adjacency list formatted subgraph and generate a mirage kernel graph
def to_kernel_graph(subgraph):
    graph = mi.new_kernel_graph()
    dims = []
    # stores output tensors of operations + their reference counts based on ID
    intermediates = {}
    for op, _ in subgraph.items():
        inputs = []
        for (shape, tensor_id) in op.input_tensor_shapes:
            if tensor_id not in intermediates:
                dims.append((shape, "V"))
                inputs.append(graph.new_input(dims=shape, dtype=mi.float16))
            else:
                inputs.append(intermediates[tensor_id][0])
                intermediates[tensor_id][1] += 1
        if (op.fn == "Sigmoid"):
            shape = op.output_tensor_shapes[0][0]
            dims.append((shape, "C", 1.0))
            inputs.append(graph.new_input(dims=shape, dtype=mi.float16))
            dims.append((shape, "C", -1.0))
            inputs.append(graph.new_input(dims=shape, dtype=mi.float16))
        elif (op.fn == "Neg"):
            shape = op.output_tensor_shapes[0][0]
            dims.append((shape, "C", -1.0))
            inputs.append(graph.new_input(dims=shape, dtype=mi.float16))
        elif (op.fn == "Reciprocal"):
            shape = op.output_tensor_shapes[0][0]
            dims.append((shape, "C", 1.0))
            inputs.insert(0, graph.new_input(dims=shape, dtype=mi.float16))
        inputs += op.additional_params
        kwargs = op.kwargs
        res = function_map(graph, op, inputs, kwargs)
        if type(res) == list:
            for i, tensor in enumerate(res):
                intermediates[op.output_tensor_shapes[i][1]] = [tensor, 0]
        else:
            intermediates[op.output_tensor_shapes[0][1]] = [res, 0]
    for tensor, count in intermediates.items():
        if count == 0: graph.mark_output(tensor)
    return graph, dims
        
def generate_all_kernels(model, dummy_inputs, min_num_ops=2, max_num_ops=4, UNSUPPORTED_OPS=set(), IGNORE_OPS=set()):
    subgraphs, _ = partition_graph(model, dummy_inputs, min_num_ops, max_num_ops, UNSUPPORTED_OPS, IGNORE_OPS)
    kernel_input_dims = []
    all_kernels = []
    for subgraph in subgraphs:
        kernel_graph, dims = to_kernel_graph(subgraph)
        all_kernels.append(kernel_graph.superoptimize())
        kernel_input_dims.append(dims)
    return all_kernels, kernel_input_dims

def time_kernels(kernels, device, iterations=1):
    times = []
    for kernel, dims in kernels:
        total_time = 0
        for _ in range(iterations):
            inputs = []
            for dim in dims:
                if (dim[1] == "V"):
                    inputs.append(torch.randn(dim[0], requires_grad=True).to(device))
                elif (dim[1] == "C"):
                    inputs.append(torch.full(dim[0], dim[2]).to(device))
            start = time.time()
            _ = kernel(*inputs)
            total_time += time.time() - start
        times.append(total_time / iterations)
    return times



"""
Example usage:

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

from partition_graph import partition_graph

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Define dummy tensors and run it through the model to get the computational graph
dummy_input_ids = torch.randint(0, 30522, (2, 128)).to(device)
dummy_attention_mask = torch.ones((2, 128), requires_grad=True).to(device)
dummy_labels = torch.tensor([0, 1], dtype=torch.long).to(device)

dummy_outputs = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask, labels=dummy_labels)
dummy_loss = dummy_outputs.loss

subgraphs, unique_operators = partition_graph(dummy_loss)
all_kernels, kernel_input_dims = generate_all_kernels(subgraphs)
times = time_kernels(all_kernels, device)
"""

"""
Format of subgraphs:

The format of the subgraphs are in an adjacency list format, the keys are Operator objects each
containing the name, grad_fn, the list of input/output operators that the operator recieves
tensors from/sends tensors to, as well as the shapes of the input and output tensors to this operator.
The values are lists of Operator objects.

The entry op1: [op2, op3] means that op1 outputs its tensors to op2 and op3, i.e.

op1 --> op2
|
v
op3

"""
