import os
import sys
import torch
import onnx

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, "python"))  
sys.path.append(root_dir) 

from scripts.build_computation_graph import SimpleClassifierMix, parse_onnx_model, SplitModel
from scripts.partition_graph import Operator
from scripts.operators_to_mirage_kernel_graph import GraphSplitter, process_operator_graph

def load_model_and_create_graph():
    """
    Loads the SimpleClassifierMix model, converts it to ONNX, and extracts the operator graph.
    """
    model = SimpleClassifierMix(input_size=784, hidden_size=128, num_classes=2)
    model.eval() 
    batch_size = 2
    input_size = 784
    dummy_input = torch.randn(batch_size, input_size)
    
    
    # Generate the ONNX file
    onnx_path = "scripts/onnx/integrate_test.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=True
    )
    

    onnx_model = onnx.load(onnx_path)
    operators = parse_onnx_model(onnx_model)
    
    # Convert to the format expected by process_operator_graph
    operators_graph = {op: True for op in operators.values()}
    
    return operators_graph

def print_graph_info(subgraphs, dependencies):
    """Print detailed information about subgraphs and their dependencies"""
    print("\nDetailed Subgraph Information:")
    
    for sg_id, (sg, sg_type) in enumerate(subgraphs):
        op_type = "Supported" if sg_type == "mirage" else "Unsupported"
        
        print(f"Subgraph {sg_id}: {op_type}, {len(sg)} operators")
        print(f"  Operators: {[op.name for op in sg.keys()]}")
        
        if sg_id in dependencies:
            print(f"  Depends on subgraphs: {dependencies[sg_id]}")
        else:
            print("  No dependencies (entry point)")
    
def main():
    try:
        print("Loading model and creating operator graph...")
        operators_graph = load_model_and_create_graph()
    
        print("\nRunning process_operator_graph:")
        subgraphs, dependencies = process_operator_graph(operators_graph)
        
        print_graph_info(subgraphs, dependencies)
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 