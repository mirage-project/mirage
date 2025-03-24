import os
import sys

# Add paths to modules
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, "python"))  # for mirage module
sys.path.append(root_dir)  # for scripts module

from scripts.partition_graph import Operator
from scripts.operators_to_mirage_kernel_graph import GraphSplitter, process_operator_graph

def create_simple_test_graph():
    """
    Creates a test graph with the following structure:
    
    (A) -> B -> D -> E
     |             ^
     v             |
     C -------------
    
    Only A is unsupported.
    """
    # Create tensor shapes and IDs for simulation
    shape_1 = ((6, 6), 2001)
    shape_2 = ((3, 3), 2002)
    shape_3 = ((3, 3), 2003)
    shape_4 = ((3, 3), 2004)
    shape_5 = ((3, 3), 2005)
    shape_6 = ((3, 3), 2006)
    
    # Create operators with names indicating their support status
    node_A = Operator(
        name="A_custom_op",
        fn="custom_op",  # Unsupported operation
        input_tensor_shapes=[shape_1],  # External input
        output_tensor_shapes=[shape_2, shape_3]
    )
    
    node_B = Operator(
        name="B_relu",
        fn="relu",
        input_tensor_shapes=[shape_2],
        output_tensor_shapes=[shape_4]
    )
    
    node_C = Operator(
        name="C_exp",
        fn="exp",
        input_tensor_shapes=[shape_3],
        output_tensor_shapes=[shape_5]
    )
    
    node_D = Operator(
        name="D_relu",
        fn="relu",
        input_tensor_shapes=[shape_4],
        output_tensor_shapes=[shape_6]
    )
    
    node_E = Operator(
        name="E_add",
        fn="add",
        input_tensor_shapes=[shape_5, shape_6],
        output_tensor_shapes=[((8, 9), 2007)]
    )
    
    # Set up connections
    node_A.input_ops = []
    node_A.output_ops = [node_B, node_C]
    
    node_B.input_ops = [node_A]
    node_B.output_ops = [node_D]
    
    node_C.input_ops = [node_A]
    node_C.output_ops = [node_E]
    
    node_D.input_ops = [node_B]
    node_D.output_ops = [node_E]
    
    node_E.input_ops = [node_C, node_D]
    node_E.output_ops = []
    
    # Create the graph dictionary
    operators_graph = {
        node_A: True,  # A is unsupported
        node_B: True,
        node_C: True,
        node_D: True,
        node_E: True
    }
    
    # Print graph structure for verification
    print("Graph structure:")
    for node, supported in operators_graph.items():
        print(f"Node: {node.name} (Supported: {supported})")
        print(f"  Inputs: {[inp.name for inp in node.input_ops]}")
        print(f"  Outputs: {[out.name for out in node.output_ops]}")
    
    return operators_graph

def print_detailed_info(subgraphs, dependencies):
    """
    Print detailed information about subgraphs and dependencies
    """
    print("\nDetailed Subgraph Information:")
    for sg_id, (sg, sg_type) in enumerate(subgraphs):
        op_type = "Supported" if sg_type == "mirage" else "Unsupported"
        print(f"Subgraph {sg_id}: {op_type} operations, {len(sg)} operators")
        
        # If you need to print operator names:
        print(f"  Operators: {[op.name for op in sg.keys()]}")
        
        # Print dependencies
        if sg_id in dependencies:
            print(f"  Depends on subgraphs: {dependencies[sg_id]}")
        else:
            print("  No dependencies")
            
        print()

def main():
    print("Creating test graph with only node A unsupported...")
    operators_graph = create_simple_test_graph()
    
    print("\nRunning graph splitting algorithm...")
    subgraphs, dependencies = process_operator_graph(operators_graph)
    
    # Print detailed information about the results
    print_detailed_info(subgraphs, dependencies)

if __name__ == "__main__":
    main()