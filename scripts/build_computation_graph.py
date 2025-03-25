import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch import nn
import pickle
import onnx
from onnx import shape_inference
from partition_graph import Operator
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=2):
        super(SimpleClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SimpleClassifierMix(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=2):
        super(SimpleClassifierMix, self).__init__()
        
        # Supported
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.linear2 = nn.Linear(hidden_size, num_classes)  
        
        # Unsupported
        self.dropout = nn.Dropout(0.2)  
        
    def forward(self, x):
        x = self.linear1(x)  
        x = F.relu(x)  
        x = self.dropout(x)  
        x = self.linear2(x)          
        # Unsupported activation
        x = torch.tanh(x) 
        
        return x
    

# model = SimpleClassifier()
# model.to(device)
# batch_size = 2
# input_size = 784  
# dummy_input = torch.randn(batch_size, input_size, device=device)

# torch.onnx.export(
#     model,    
#     dummy_input,              
#     "scripts/onnx/my_model.onnx", 
#     input_names=["input"], 
#     dynamo=True                  
# )



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

class SplitModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SplitModel, self).__init__()
        self.fc1 = nn.Linear(input_dim // 2, hidden_dim)
        self.fc2 = nn.Linear(input_dim // 2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        x = torch.cat((x1, x2), dim=1)

        # x1 = torch.relu(self.fc1(x1))
        # x2 = torch.relu(self.fc2(x2))
        
        # x = torch.cat((x1, x2), dim=1)
        # x = self.fc_out(x)
        return x

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

# onnx_path = "scripts/onnx/split_model.onnx"
# torch.onnx.export(
#     model, 
#     dummy_input, 
#     onnx_path,
#     input_names=["input"],
#     output_names=["output"],
#     dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # Allows dynamic batch size
#     dynamo = True
# )


"""
Prints a representzation of the computational graph
"""
def print_computational_graph(root_node, indent=0, visited=None):
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



"""
Parse ONNX representation of model and build operator graph
"""
def parse_onnx_model(model):
    inferred_model = shape_inference.infer_shapes(model) # for shape inference of inputs and outputs
    shape_value_dict = {}

    for initializer in inferred_model.graph.initializer:
        shape_value_dict[initializer.name] = initializer.dims

    for value in inferred_model.graph.value_info:
        shape_value_dict[value.name] = [d_i.dim_value for d_i in value.type.tensor_type.shape.dim]

    for input_info in inferred_model.graph.input:
        shape_value_dict[input_info.name] = [d_i.dim_value for d_i in input_info.type.tensor_type.shape.dim]
    
    for output_info in inferred_model.graph.output:
        shape_value_dict[output_info.name] = [d_i.dim_value for d_i in output_info.type.tensor_type.shape.dim]

    tensor_producer = {}
    tensor_consumer = {}

    # store the tensor producers in a dict
    for node in model.graph.node:
        for output in node.output:
            tensor_producer[output] = node.name if node.name else f"{node.op_type}_{id(node)}"

        for input in node.input:
            tensor_consumer[input] = node.name if node.name else f"{node.op_type}_{id(node)}"

    operators = {}
    # store the operators in a dict
    for node in model.graph.node:
        op_type = node.op_type
        node_name = node.name or f"{node.op_type}_{id(node)}"
        input_tensor_shapes = [(shape_value_dict[input_name], id(node)) for input_name in node.input]
        output_tensor_shapes = [(shape_value_dict[output_name], id(node)) for output_name in node.output]
        print(node_name)
        print("inputs : ", [input_name for input_name in node.input])

        operator = Operator(name=node_name, fn=op_type, input_ops=[], output_ops=[], input_tensor_shapes=input_tensor_shapes, output_tensor_shapes=output_tensor_shapes)
        # operator = {"name":node_name, "fn":op_type, "input_ops":[], "output_ops":[], "input_tensor_shapes":input_tensor_shapes, "output_tensor_shapes":output_tensor_shapes}
        operators[node_name] = operator
    # print("Tensor producers: ", tensor_producer)
    # print("Tensor consumers: ", tensor_consumer)

    # print(operators)

    # Filling in input ops
    for node in model.graph.node:
        for input_name in node.input:
            if input_name in tensor_producer:
                operators[node.name].input_ops.append(operators[tensor_producer[input_name]])
            else:
                dummy_const_operator = Operator(name=input_name, fn="Constant")
                operators[node.name].input_ops.append(dummy_const_operator)
    
    # print("After adding inputs")
    # print(operators)
    
    # Filling in output ops
    for node in model.graph.node:
        for output_name in node.output:
            if output_name in tensor_consumer:
                operators[node.name].output_ops.append(operators[tensor_consumer[output_name]])
            else:
                dummy_const_operator = Operator(name=output_name, fn="Constant")
                operators[node.name].input_ops.append(dummy_const_operator)
    # print("After adding outputs")
    # print(operators)

    # print([node.name for node in model.graph.node])
    root_node = operators[model.graph.node[0].name]
    # print_computational_graph(operators['node_Transpose_0'])
    print_computational_graph(root_node)
    return operators


"""Trying using a smaller model"""
model = onnx.load('scripts/onnx/my_model.onnx')
# model = onnx.load('scripts/onnx/split_model.onnx')

parse_onnx_model(model)


"""Trying using BERT"""

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# bert_model.to(device)
# dummy_input_ids = torch.randint(0, 30522, (2, 128)).to(device)
# dummy_attention_mask = torch.ones((2, 128), requires_grad=True).to(device)
# dummy_labels = torch.tensor([0, 1], dtype=torch.long).to(device)

# dummy_outputs = bert_model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask, labels=dummy_labels)
# dummy_loss = dummy_outputs.loss


# torch.onnx.export(
#     bert_model,    
#     (dummy_input_ids, dummy_attention_mask),              
#     "scripts/onnx/bert_model.onnx", 
#     input_names=["input_ids", "attention_mask"],
#     output_names=["logits"],
#     dynamo=True                  
# )

# bert_model = onnx.load('scripts/onnx/bert_model.onnx')

# parse_onnx_model(bert_model)


