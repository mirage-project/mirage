import torch
from itertools import combinations as comb

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
        op_node.input_tensor_shapes = [input.shape if input != None else None for input in inputs]
        op_node.output_tensor_shapes = [output.shape if output != None else None for output in outputs]
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

def get_partitions(op_node, max_num_ops, visited_start_nodes, all_subgraphs, UNSUPPORTED_OPS):
    visited_start_nodes.add(id(op_node.fn))
    
    if op_node.name not in UNSUPPORTED_OPS:
        get_partitions_helper(op_node, {op_node: None}, max_num_ops, set(), all_subgraphs, UNSUPPORTED_OPS)
        
    for input_node in op_node.input_ops:
        if id(input_node.fn) not in visited_start_nodes:
            get_partitions(input_node, max_num_ops, visited_start_nodes, all_subgraphs, UNSUPPORTED_OPS)

def get_partitions_helper(op_node, curr_subgraph, max_num_ops, visited, all_subgraphs, UNSUPPORTED_OPS):
    if id(op_node.fn) in visited:
        return
    if len(curr_subgraph) > max_num_ops:
        return
    if op_node.name in UNSUPPORTED_OPS:
        return
    
    # assume op_node already in curr_subgraph
    visited.add(id(op_node.fn))
    all_subgraphs.append(curr_subgraph.copy())
    
    for choose_k in range(1, len(op_node.input_ops)):
        curr_subgraph_copy = curr_subgraph.copy()
        for comb_inputs in comb(op_node.input_ops, choose_k):
            visited_copy = visited.copy()
            for input_node in comb_inputs:
                if input_node not in curr_subgraph_copy:
                    curr_subgraph_copy[input_node] = [op_node]
                else:
                    curr_subgraph_copy[input_node].append(op_node)

                get_partitions_helper(input_node, curr_subgraph_copy.copy(), max_num_ops, visited_copy, all_subgraphs, UNSUPPORTED_OPS)


def partition_graph(dummy_loss, max_num_ops=3, UNSUPPORTED_OPS=set(["torch::autograd::AccumulateGrad", 
                                                              "NllLossBackward0", 
                                                              "EmbeddingBackward0"])):
    loss_node = Operator(name=dummy_loss.grad_fn.name(), fn=dummy_loss.grad_fn)
    
    unique_operators = set()
    build_computational_graph(loss_node, unique_operators)

    dummy_loss.backward()

    all_subgraphs = []
    visited_start_nodes = set()
    get_partitions(loss_node, max_num_ops, visited_start_nodes, all_subgraphs, UNSUPPORTED_OPS)

    return all_subgraphs, unique_operators

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