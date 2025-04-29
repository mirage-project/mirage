import torch
from torch import nn
import pickle
import os
import onnx
from onnx import shape_inference
from op import Operator
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    


class SplitModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SplitModel, self).__init__()
        self.fc1 = nn.Linear(input_dim // 2, hidden_dim)
        self.fc2 = nn.Linear(input_dim // 2, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        x = torch.cat((x1, x2), dim=1)

        x1 = torch.relu(self.fc1(x1))
        x2 = torch.relu(self.fc2(x2))
        
        x = torch.cat((x1, x2), dim=1)
        x = self.fc_out(x)
        return x



"""
Prints a representation of the computational graph
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
def parse_onnx_model(model, unique_operators):
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
        additional_params = []
        kwargs = {}
        if node.op_type == "Clip":
            # Using .item() here pulls out the value from the rank-0 tensor
            if node.min != None:
                additional_params.append(node.min.item()) 
            else: # Defaults are to do nothing but could be changed
                additional_params.append(float('-inf'))
            if node.max!= None:
                additional_params.append(node.max.item()) 
            else:
                additional_params.append(float('inf'))
        if node.op_type == "ReduceSum":
            # TODO add ReduceMean here as well?
            if node.axis != None:
                axis = node.axis[0]
                additional_params.append(axis)
            else:
                additional_params.append(0)

        input_tensor_shapes = []
        for i, input_name in enumerate(node.input):
            if input_name in shape_value_dict and input_name in tensor_id:
                shape = shape_value_dict[input_name]

                # TODO we should leave modifying the graph structure to partition_graph.py
                # since build_computational graph should faithfully construct the model
                if i == 0:
                    if op_type == "Softmax":
                        shape = shape[:3]
                if i == 1:
                    if op_type in ["Div", "Add", "Pow"]:
                        shape = input_tensor_shapes[0][0]
                    if op_type == "ReduceMean":
                        # the second tensor of ReduceMean is a 1x1 tensor with the reduction dimension
                        # TODO for now, we assume the last dimension is the reduction dimension always
                        continue
                    if op_type == "MatMul":
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
                if i == 0:
                    if op_type == "Softmax":
                        shape = shape[:3]
                output_tensor_shapes.append((shape, tensor_id[output_name]))
        
        operator = Operator(name=node_name, fn=op_type, input_ops=[], output_ops=[], input_tensor_shapes=input_tensor_shapes, output_tensor_shapes=output_tensor_shapes, additional_params=additional_params, kwargs=kwargs)
        # operator = {"name":node_name, "fn":op_type, "input_ops":[], "output_ops":[], "input_tensor_shapes":input_tensor_shapes, "output_tensor_shapes":output_tensor_shapes}
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
    
    # print("After adding inputs")
    # print(operators)
    
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
    # print("After adding outputs")
    # print(operators)

    # print([node.name for node in model.graph.node])
    # root_node = operators[model.graph.node[0].name]
    # print_computational_graph(operators['node_Transpose_0'])
    # print_computational_graph(root_node)
    return operators

def test_cfg():
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


def get_computation_graph(model, dummy_input, unique_operators, method):
    match method:
        case "onnx":
            # Generate the ONNX file
            onnx_path = "scripts/onnx/integrate_test.onnx"
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

            # dynamic_axes = {name: {0: "batch_size"} for name in model.get_input_names() + model.get_output_names()}

            # torch.onnx.export(
            #     model,
            #     dummy_input,
            #     onnx_path,
            #     dynamo=True
            # )
            
            # shape_inference.infer_shapes_path(model_path="scripts/onnx/integrate_test.onnx",
            #                                   output_path="scripts/onnx/inferred_model.onnx") # for shape inference of inputs and outputs
            inferred_model = onnx.load("scripts/onnx/inferred_model.onnx")
            operators = parse_onnx_model(inferred_model, unique_operators)

            # for k, v in operators.items():
            #     print(k, " input ops: ", [(inp.name, inp.fn) for inp in v.input_ops])
            #     print(k, " output ops: ", [(out.name, out.fn) for out in v.output_ops])

            return operators
        case _:
            print("Unsupported method for build_graph")
            return None
    
