import os
import sys
import torch
import onnx

from build_computation_graph import get_computation_graph

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, "python"))  
sys.path.append(root_dir) 

from scripts.build_computation_graph import SimpleClassifierMix, parse_onnx_model, SplitModel
from scripts.partition_graph import Operator
from scripts.graph_splitter import process_operator_graph

def load_model_and_create_graph():
    """
    Loads the SimpleClassifierMix model, converts it to ONNX, and extracts the operator graph.
    """
    model = SimpleClassifierMix(input_size=784, hidden_size=128, num_classes=2)
    model.eval() 
    batch_size = 2
    input_size = 784
    dummy_input = torch.randn(batch_size, input_size)
    
    unique_operators = set()
    root_node, operators = get_computation_graph(model, dummy_input, unique_operators, "onnx")
    
    return operators

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
        operators = load_model_and_create_graph()
    
        print("\nRunning process_operator_graph:")
        subgraphs, dependencies = process_operator_graph(operators)
        
        print_graph_info(subgraphs, dependencies)
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 