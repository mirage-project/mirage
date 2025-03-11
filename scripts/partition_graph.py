import torch
from itertools import combinations as comb
import time
import mirage as mi

ids_to_nodes = {}

class Operator:
    def __init__(self, name=None, fn=None, input_ops=[], output_ops=[], input_tensor_shapes=[], output_tensor_shapes=[]):
        self.name = name
        self.fn = fn
        self.input_ops = input_ops
        self.output_ops = output_ops
        self.input_tensor_shapes = input_tensor_shapes
        self.output_tensor_shapes = output_tensor_shapes

def build_computational_graph(op_node, unique_operators):
    unique_operators.add(op_node.name)
    if op_node.input_ops != []:
        return
    
    local_scope = {}
    exec(
f"""
def hook_fn(inputs, outputs):
    if {id(op_node.fn)} in ids_to_nodes:
        op_node = ids_to_nodes[{id(op_node.fn)}]
        op_node.input_tensor_shapes = [(input.shape, id(input)) if input != None else None for input in inputs]
        op_node.output_tensor_shapes = [(output.shape, id(output)) if output != None else None for output in outputs]
""",
    globals(),
    local_scope
    )

    op_node.fn.register_hook(local_scope["hook_fn"])

    if id(op_node.fn) not in ids_to_nodes:
        ids_to_nodes[id(op_node.fn)] = op_node

    for next_fn in op_node.fn.next_functions:
        if id(next_fn[0]) in ids_to_nodes:
            ids_to_nodes[id(next_fn[0])].output_ops.append(op_node)
        else:
            if next_fn[0] == None:
                continue
            ids_to_nodes[id(next_fn[0])] = Operator(name=next_fn[0].name(), fn=next_fn[0], input_ops=[], output_ops=[op_node])
        op_node.input_ops.append(ids_to_nodes[id(next_fn[0])])
        
        build_computational_graph(ids_to_nodes[id(next_fn[0])], unique_operators)

def get_partitions(op_node, min_num_ops, max_num_ops, visited_start_nodes, all_subgraphs, UNSUPPORTED_OPS):
    visited_start_nodes.add(id(op_node.fn))
    
    if op_node.name not in UNSUPPORTED_OPS:
        get_partitions_helper(op_node, {op_node: None}, min_num_ops, max_num_ops, set(), all_subgraphs, UNSUPPORTED_OPS)
        
    for input_node in op_node.input_ops:
        if id(input_node.fn) not in visited_start_nodes:
            get_partitions(input_node, min_num_ops, max_num_ops, visited_start_nodes, all_subgraphs, UNSUPPORTED_OPS)

def get_partitions_helper(op_node, curr_subgraph, min_num_ops, max_num_ops, visited, all_subgraphs, UNSUPPORTED_OPS):
    if id(op_node.fn) in visited:
        return
    if len(curr_subgraph) > max_num_ops:
        return
    
    # assume op_node already in curr_subgraph
    visited.add(id(op_node.fn))
    if len(curr_subgraph) >= min_num_ops:
        all_subgraphs.append(curr_subgraph.copy())

    valid_input_ops = []
    for input_op in op_node.input_ops:
        if input_op.name not in UNSUPPORTED_OPS:
            valid_input_ops.append(input_op)
    
    for choose_k in range(1, len(valid_input_ops)):
        curr_subgraph_copy = curr_subgraph.copy()
        for comb_inputs in comb(valid_input_ops, choose_k):
            visited_copy = visited.copy()
            for input_node in comb_inputs:
                if input_node not in curr_subgraph_copy:
                    curr_subgraph_copy[input_node] = [op_node]
                else:
                    curr_subgraph_copy[input_node].append(op_node)

                get_partitions_helper(input_node, curr_subgraph_copy.copy(), min_num_ops, max_num_ops, visited_copy, all_subgraphs, UNSUPPORTED_OPS)


def partition_graph(dummy_loss, min_num_ops=2, max_num_ops=4, UNSUPPORTED_OPS=set(["torch::autograd::AccumulateGrad", 
                                                              "NllLossBackward0", 
                                                              "EmbeddingBackward0"])):
    loss_node = Operator(name=dummy_loss.grad_fn.name(), fn=dummy_loss.grad_fn)
    
    unique_operators = set()
    build_computational_graph(loss_node, unique_operators)

    dummy_loss.backward()

    all_subgraphs = []
    visited_start_nodes = set()
    get_partitions(loss_node, min_num_ops, max_num_ops, visited_start_nodes, all_subgraphs, UNSUPPORTED_OPS)

    return all_subgraphs, unique_operators

# TODO: add support for reduction, clamp, rms_norm. These rely on additional
# inputs that the Operator class doesn't currently support
def function_map(graph, func, inputs):
    match func.name:
        case "matmul": return graph.matmul(*inputs)
        #case "reduction": return graph.reduction(*inputs)
        case "exp": return graph.exp(*inputs)
        case "silu": return graph.silu(*inputs)
        case "gelu": return graph.gelu(*inputs)
        case "relu": return graph.relu(*inputs)
        #case "clamp": return graph.clamp(*inputs)
        case "add": return graph.add(*inputs)
        case "mul": return graph.mul(*inputs)
        case "div": return graph.div(*inputs)
        #case "rms_norm": return graph.rms_norm(*inputs)
        case _: raise NotImplementedError


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
                dims.append(shape)
                inputs.append(graph.new_input(dims=shape, dtype=mi.float16))
            else:
                inputs.append(intermediates[tensor_id][0])
                intermediates[tensor_id][1] += 1
        res = function_map(graph, op.fn, inputs)
        if type(res) == list:
            for i, tensor in enumerate(res):
                intermediates[op.output_tensor_shapes[i][1]] = (tensor, 0)
        else:
            intermediates[op.output_tensor_shapes[0][1]] = (res, 0)
    for tensor, count in intermediates.items():
        if count == 0: graph.mark_output(tensor)
    return graph, dims
        
def generate_all_kernels(dummy_loss, max_num_ops=3, UNSUPPORTED_OPS=set(["torch::autograd::AccumulateGrad", 
                                                              "NllLossBackward0", 
                                                              "EmbeddingBackward0"])):
    subgraphs, unique_operators = partition_graph(dummy_loss, max_num_ops, UNSUPPORTED_OPS)
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
                inputs.append(torch.randn(dim, requires_grad=True).to(device))
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