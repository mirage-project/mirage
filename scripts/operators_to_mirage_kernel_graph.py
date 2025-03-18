import os
import sys

# Add the path to mirage module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from partition_graph import to_kernel_graph, generate_all_kernels

# import mirage as mi
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Any, Optional, NamedTuple
from dataclasses import dataclass, field

@dataclass
class SubgraphData:
    """Represents a subgraph with detailed I/O information"""
    id: int
    type: str  # "mirage" or "pytorch"
    operators: Dict[Any, bool]
    input_edges: Dict[Any, List[Tuple[int, Any]]] = field(default_factory=dict)  # {op: [(source_sg_id, source_op), ...]}
    output_edges: Dict[Any, List[Tuple[int, Any]]] = field(default_factory=dict)  # {op: [(target_sg_id, target_op), ...]}

class GraphSplitter:
    """Splits a mixed operator graph into Mirage-supported and unsupported subgraphs."""
    
    def __init__(self):
        # Mirage supported operations
        self.mirage_supported_ops = {
            "matmul", "reduction", "exp", "silu", "gelu", "relu", 
            "clamp", "add", "mul", "div", "rms_norm"
        }
    
    def is_supported_op(self, op) -> bool:
        """Check if an operation is supported by Mirage."""
        if not hasattr(op, 'fn'):
            return False
        
        op_type = op.fn
        if isinstance(op_type, str):
            op_type_lower = op_type.lower()
            return any(supported_op in op_type_lower for supported_op in self.mirage_supported_ops)
        
        return False
    
    def split_graph(self, operators_graph: Dict) -> Tuple[List[Tuple[Dict, str]], Dict[int, Set[int]], List[Dict]]:
        """
        Split operator graph into subgraphs by type and connectivity.
        
        The algorithm works as follows:
        1. Topologically sort the operators
        2. Group operators by type (supported/unsupported)
        3. Find connected components within each type
        4. Build I/O relationships between subgraphs
        5. Resolve cycles in the dependency graph if needed
        """
        # Topologically sort the operators
        sorted_ops = self._topological_sort(operators_graph)
        
        # Group by operation type
        mirage_ops = [op for op in sorted_ops if self.is_supported_op(op)]
        pytorch_ops = [op for op in sorted_ops if not self.is_supported_op(op)]
        
        # Find connected components for each type
        subgraphs = []  # [(subgraph_dict, type)]
        op_to_subgraph = {}  # Maps operator -> subgraph_id
        
        # Process both operation types
        for ops_group, sg_type in [(mirage_ops, "mirage"), (pytorch_ops, "pytorch")]:
            components = self._find_connected_components(ops_group)
            
            for component in components:
                sg_id = len(subgraphs)
                subgraphs.append((component, sg_type))
                
                # Map each operator to its subgraph
                for op in component:
                    op_to_subgraph[op] = sg_id
        
        # Track subgraph I/O relationships
        subgraph_io = self._build_io_relationships(subgraphs, op_to_subgraph)
        
        # Build dependencies between subgraphs
        subgraph_deps = self._build_dependencies(subgraph_io)
        
        # Fix cycles if needed
        if not self._is_acyclic(subgraph_deps):
            subgraphs, subgraph_io, subgraph_deps = self._break_cycles(
                subgraphs, subgraph_io, subgraph_deps, sorted_ops
            )
        
        # Print debug info
        self._print_subgraph_info(subgraphs, subgraph_io)
        
        return subgraphs, subgraph_deps, subgraph_io
    
    def _topological_sort(self, operators_graph: Dict) -> List[Any]:
        """
        Topologically sort the operators based on their dependencies.
        This ensures we process operators in an order that respects data flow.
        """
        in_degree = {op: 0 for op in operators_graph}
        
        # Calculate in-degrees
        for op in operators_graph:
            for output_op in op.output_ops:
                if output_op in operators_graph:
                    in_degree[output_op] += 1
        
        # Start with nodes having no incoming edges
        queue = deque([op for op, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for output_op in current.output_ops:
                if output_op in in_degree:
                    in_degree[output_op] -= 1
                    if in_degree[output_op] == 0:
                        queue.append(output_op)
        
        if len(result) != len(operators_graph):
            raise ValueError("Operator graph contains cycles")
            
        return result
    
    def _find_connected_components(self, operators):
        """
        Find connected components within a set of operators.
        Each component represents operators that can be executed as a single subgraph.
        """
        visited = set()
        components = []
        
        for op in operators:
            if op in visited:
                continue
                
            # Start a new component
            component = {}
            queue = [op]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                    
                visited.add(current)
                component[current] = True
                
                # Check connections to same-type operators
                for connected in list(current.input_ops) + list(current.output_ops):
                    if connected in operators and connected not in visited:
                        queue.append(connected)
            
            if component:
                components.append(component)
        
        return components
    
    def _build_io_relationships(self, subgraphs, op_to_subgraph):
        """
        Build the I/O relationships between subgraphs.
        This identifies how data flows between different subgraphs.
        """
        subgraph_io = [{"inputs": {}, "outputs": {}} for _ in subgraphs]
        
        for sg_id, (sg, _) in enumerate(subgraphs):
            for op in sg:
                # Find inputs from other subgraphs
                for input_op in op.input_ops:
                    if input_op in op_to_subgraph:
                        input_sg_id = op_to_subgraph[input_op]
                        if input_sg_id != sg_id:
                            # Record cross-subgraph connections
                            if op not in subgraph_io[sg_id]["inputs"]:
                                subgraph_io[sg_id]["inputs"][op] = []
                            subgraph_io[sg_id]["inputs"][op].append((input_sg_id, input_op))
                            
                            if input_op not in subgraph_io[input_sg_id]["outputs"]:
                                subgraph_io[input_sg_id]["outputs"][input_op] = []
                            subgraph_io[input_sg_id]["outputs"][input_op].append((sg_id, op))
        
        return subgraph_io
    
    def _build_dependencies(self, subgraph_io):
        """Build dependencies between subgraphs based on I/O relationships."""
        subgraph_deps = {}
        
        for sg_id, sg_io in enumerate(subgraph_io):
            deps = set()
            for op, inputs in sg_io["inputs"].items():
                for input_sg_id, _ in inputs:
                    deps.add(input_sg_id)
            if deps:
                subgraph_deps[sg_id] = deps
        
        return subgraph_deps
    
    def _is_acyclic(self, dependencies):
        """Check if the dependency graph is acyclic."""
        try:
            self._topological_sort_subgraphs(dependencies)
            return True
        except ValueError:
            return False
    
    def _topological_sort_subgraphs(self, dependencies):
        """Topologically sort the subgraphs based on dependencies."""
        all_nodes = set()
        for node, deps in dependencies.items():
            all_nodes.add(node)
            all_nodes.update(deps)
        
        in_degree = {node: 0 for node in all_nodes}
        for node, deps in dependencies.items():
            for dep in deps:
                in_degree[dep] += 1
        
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            dependent_nodes = [n for n, deps in dependencies.items() if node in deps]
            for dep_node in dependent_nodes:
                in_degree[dep_node] -= 1
                if in_degree[dep_node] == 0:
                    queue.append(dep_node)
        
        if len(result) != len(all_nodes):
            raise ValueError("Dependency graph contains cycles")
        
        return result
    
    def _find_cycle(self, dependencies):
        """Find a cycle in the dependency graph."""
        visited = set()
        recursion_stack = set()
        
        cycle_nodes = []
        
        def dfs_check_cycle(node, path):
            nonlocal cycle_nodes
            
            if node in recursion_stack:
                cycle_start_idx = path.index(node)
                cycle_nodes = path[cycle_start_idx:] + [node]
                return True
                
            if node in visited:
                return False
                
            visited.add(node)
            recursion_stack.add(node)
            path.append(node)
            
            if node in dependencies:
                for dep in dependencies[node]:
                    if dfs_check_cycle(dep, path):
                        return True
            
            path.pop()
            recursion_stack.remove(node)
            return False
        
        for node in dependencies:
            if node not in visited:
                if dfs_check_cycle(node, []):
                    return cycle_nodes
        
        return []
    
    def _break_cycles(self, subgraphs, subgraph_io, subgraph_deps, sorted_ops):
        """
        Break cycles in the subgraph dependency graph.
        
        Strategy:
        1. Find a cycle in the dependency graph
        2. Split one of the subgraphs in the cycle into two parts
        3. Rebuild I/O relationships and dependencies
        4. Repeat until no cycles remain
        """
        topo_indices = {op: idx for idx, op in enumerate(sorted_ops)}
        
        # Try to break cycles a limited number of times
        attempts = 0
        max_attempts = 10
        
        while not self._is_acyclic(subgraph_deps) and attempts < max_attempts:
            attempts += 1
            
            cycle = self._find_cycle(subgraph_deps)
            if not cycle:
                break
            
            # Split the first subgraph in the cycle
            subgraph_to_split = cycle[0]
            sg_dict, sg_type = subgraphs[subgraph_to_split]
            
            # Split based on topological order
            ops_in_sg = list(sg_dict.keys())
            ops_in_sg.sort(key=lambda op: topo_indices[op])
            
            mid_point = len(ops_in_sg) // 2
            first_half = {op: True for op in ops_in_sg[:mid_point]}
            second_half = {op: True for op in ops_in_sg[mid_point:]}
            
            # Replace with two new subgraphs
            subgraphs[subgraph_to_split] = None
            subgraphs.append((first_half, sg_type))
            subgraphs.append((second_half, sg_type))
            
            # Create new I/O entries
            subgraph_io.append({"inputs": {}, "outputs": {}})
            subgraph_io.append({"inputs": {}, "outputs": {}})
            
            # Rebuild operator to subgraph mapping
            op_to_subgraph = {}
            for sg_id, sg_tuple in enumerate(subgraphs):
                if sg_tuple is not None:
                    sg, _ = sg_tuple
                    for op in sg:
                        op_to_subgraph[op] = sg_id
            
            # Rebuild I/O relationships
            subgraph_io = self._rebuild_io_relationships(subgraphs, op_to_subgraph)
            
            # Rebuild dependencies
            subgraph_deps = self._build_dependencies(subgraph_io)
        
        # Compact the IDs to remove None entries
        return self._compact_subgraphs(subgraphs, subgraph_io, subgraph_deps)
    
    def _rebuild_io_relationships(self, subgraphs, op_to_subgraph):
        """Rebuild I/O relationships after breaking a cycle."""
        subgraph_io = [{"inputs": {}, "outputs": {}} for _ in subgraphs]
        
        for sg_id, sg_tuple in enumerate(subgraphs):
            if sg_tuple is None:
                continue
                
            sg, _ = sg_tuple
            for op in sg:
                # Find inputs from other subgraphs
                for input_op in op.input_ops:
                    if input_op in op_to_subgraph:
                        input_sg_id = op_to_subgraph[input_op]
                        if input_sg_id != sg_id:
                            # Record cross-subgraph input
                            if op not in subgraph_io[sg_id]["inputs"]:
                                subgraph_io[sg_id]["inputs"][op] = []
                            subgraph_io[sg_id]["inputs"][op].append((input_sg_id, input_op))
                            
                            # Record as output in source subgraph
                            if input_op not in subgraph_io[input_sg_id]["outputs"]:
                                subgraph_io[input_sg_id]["outputs"][input_op] = []
                            subgraph_io[input_sg_id]["outputs"][input_op].append((sg_id, op))
        
        return subgraph_io
    
    def _compact_subgraphs(self, subgraphs, subgraph_io, subgraph_deps):
        """Remove None entries and compact IDs."""
        compact_subgraphs = []
        compact_io = []
        id_mapping = {}
        
        for old_id, sg_tuple in enumerate(subgraphs):
            if sg_tuple is not None:
                new_id = len(compact_subgraphs)
                id_mapping[old_id] = new_id
                compact_subgraphs.append(sg_tuple)
                compact_io.append(subgraph_io[old_id])
        
        # Update dependencies with new IDs
        compact_deps = {}
        for old_id, deps in subgraph_deps.items():
            if old_id in id_mapping:
                new_id = id_mapping[old_id]
                compact_deps[new_id] = {id_mapping[d] for d in deps if d in id_mapping}
        
        # Update I/O references
        for sg_id, sg_io in enumerate(compact_io):
            # Update inputs
            updated_inputs = {}
            for op, inputs in sg_io["inputs"].items():
                updated_inputs[op] = [(id_mapping[src_id], src_op) for src_id, src_op in inputs if src_id in id_mapping]
            sg_io["inputs"] = updated_inputs
            
            # Update outputs
            updated_outputs = {}
            for op, outputs in sg_io["outputs"].items():
                updated_outputs[op] = [(id_mapping[tgt_id], tgt_op) for tgt_id, tgt_op in outputs if tgt_id in id_mapping]
            sg_io["outputs"] = updated_outputs
        
        return compact_subgraphs, compact_io, compact_deps
    
    def _print_subgraph_info(self, subgraphs, subgraph_io):
        """Print detailed subgraph information."""
        for sg_id, (sg, sg_type) in enumerate(subgraphs):
            print(f"\nSubgraph {sg_id} ({sg_type}):")
            print(f"  Operators: {[op.name for op in sg]}")
            
            print("  Inputs:")
            for op, inputs in subgraph_io[sg_id]["inputs"].items():
                for src_sg_id, src_op in inputs:
                    print(f"    {op.name} takes input from Subgraph {src_sg_id}'s {src_op.name}")
            
            print("  Outputs:")
            for op, outputs in subgraph_io[sg_id]["outputs"].items():
                for tgt_sg_id, tgt_op in outputs:
                    print(f"    {op.name} sends output to Subgraph {tgt_sg_id}'s {tgt_op.name}")
    
    def create_mirage_graph(self, subgraphs, subgraph_io):
        """Create a Mirage execution graph from the subgraphs."""
        mirage_graph = {}
        
        for sg_id, (sg, sg_type) in enumerate(subgraphs):
            if sg_type == "mirage":
                kernel_id = f"mirage_kernel_{sg_id}"
                
                # Collect inputs and outputs
                inputs = {}
                for op, input_edges in subgraph_io[sg_id]["inputs"].items():
                    for src_sg_id, src_op in input_edges:
                        input_key = f"input_from_sg_{src_sg_id}_{src_op.name}"
                        inputs[input_key] = {
                            "source_subgraph": src_sg_id,
                            "source_operator": src_op.name
                        }
                
                outputs = {}
                for op, output_edges in subgraph_io[sg_id]["outputs"].items():
                    for tgt_sg_id, tgt_op in output_edges:
                        output_key = f"output_to_sg_{tgt_sg_id}_{tgt_op.name}"
                        outputs[output_key] = {
                            "target_subgraph": tgt_sg_id,
                            "target_operator": tgt_op.name
                        }
                
                mirage_graph[kernel_id] = {
                    "type": "mirage",
                    "operators": [op.name for op in sg],
                    "inputs": inputs,
                    "outputs": outputs
                }
        
        return mirage_graph

    def convert_to_adjacency_list(self, subgraphs):
        """
        Converts the subgraphs into adjacency list format as described in partition_graph.py.
        
        Returns:
            List of subgraph dictionaries, where each subgraph is:
            {
                operator1: {
                    "outputs": [output_op1, output_op2, ...],
                    "input_tensor_shapes": [...],
                    "output_tensor_shapes": [...]
                },
                operator2: {...},
                ...
            }
        """
        op_to_subgraph = {}
        for sg_id, (sg, _) in enumerate(subgraphs):
            for op in sg:
                op_to_subgraph[op] = sg_id
            
        adjacency_list_subgraphs = []
        
        for sg_dict, sg_type in subgraphs:
            adj_list = {}
            
            for op in sg_dict:
                adj_list[op] = {
                    "outputs": op.output_ops,
                    "input_tensor_shapes": op.input_tensor_shapes,
                    "output_tensor_shapes": op.output_tensor_shapes
                }
            
            adjacency_list_subgraphs.append(adj_list)
        
        return adjacency_list_subgraphs

    def to_kernel_graph(self, sg):
        """
        Generate a Mirage kernel graph from a subgraph.

        Args:
            sg: dict, the subgraph to convert to a Mirage kernel graph
            
        Returns:
            tuple: (graph, dims)
        """
        import mirage as mi
        
        graph = mi.new_kernel_graph()
        dims = []
        # Store the output tensors and reference counts (based on ID)
        intermediates = {}
        
        # Process each operator in the subgraph
        for op in sg:
            inputs = []
            
            # Process input tensors
            for shape_info in op.input_tensor_shapes:
                if shape_info is None:
                    inputs.append(None)
                    continue
                    
                shape, tensor_id = shape_info
                
                if tensor_id not in intermediates:
                    # External input or intermediate result首次使用
                    if shape is not None:
                        dims.append(shape)
                        inputs.append(graph.new_input(dims=shape, dtype=mi.float16))
                    else:
                        inputs.append(None)
                else:
                    # Tensor produced by previous operations in the subgraph
                    tensor, ref_count = intermediates[tensor_id]
                    inputs.append(tensor)
                    intermediates[tensor_id] = (tensor, ref_count + 1)
            
            # Apply operation, determine operation type based on fn field
            fn_name = op.fn.lower() if hasattr(op, 'fn') else op.name.lower()
            
            try:
                if "matmul" in fn_name:
                    res = graph.matmul(*inputs)
                elif "exp" in fn_name:
                    res = graph.exp(*inputs)
                elif "silu" in fn_name:
                    res = graph.silu(*inputs)
                elif "gelu" in fn_name:
                    res = graph.gelu(*inputs)
                elif "relu" in fn_name:
                    res = graph.relu(*inputs)
                elif "add" in fn_name:
                    res = graph.add(*inputs)
                elif "mul" in fn_name:
                    res = graph.mul(*inputs)
                elif "div" in fn_name:
                    res = graph.div(*inputs)
                else:
                    print(f"Unsupported operation: {fn_name}")
                    continue
            except Exception as e:
                print(f"Error executing {fn_name}: {e}")
                continue
            
            # Store output tensors
            if isinstance(res, list):
                for i, tensor in enumerate(res):
                    if i < len(op.output_tensor_shapes) and op.output_tensor_shapes[i]:
                        tensor_id = op.output_tensor_shapes[i][1]
                        intermediates[tensor_id] = (tensor, 0)
            else:
                if op.output_tensor_shapes and op.output_tensor_shapes[0]:
                    tensor_id = op.output_tensor_shapes[0][1]
                    intermediates[tensor_id] = (res, 0)
        
        # Mark unused tensors as outputs
        for tensor_id, (tensor, ref_count) in intermediates.items():
            if ref_count == 0:
                graph.mark_output(tensor)
            
        return graph, dims

def process_operator_graph(operators_graph: Dict) -> Tuple[List[Tuple[Dict, str]], Dict[int, Set[int]]]:
    """
    Process an operator graph and split it into Mirage and PyTorch subgraphs.
    
    Args:
        operators_graph: Dict mapping operators to boolean values
        
    Returns:
        Tuple[List, Dict]: List of subgraphs and their dependencies
    """
    splitter = GraphSplitter()
    
    print("Splitting operator graph into subgraphs...")
    subgraphs, subgraph_deps, subgraph_io = splitter.split_graph(operators_graph)
    
    adjacency_list_subgraphs = splitter.convert_to_adjacency_list(subgraphs)
    
    print(f"Total subgraphs: {len(subgraphs)}")
    
    for sg_id, (sg, sg_type) in enumerate(subgraphs):
        op_type = "Supported" if sg_type == "mirage" else "Unsupported"
        print(f"Subgraph {sg_id}: {op_type} operations, {len(sg)} operators")
        
        tensor_inputs = subgraph_io[sg_id].get("tensor_inputs", {})
        if tensor_inputs:
            total_tensors = sum(len(indices) for indices in tensor_inputs.values())
            print(f"  - Has {total_tensors} tensor inputs")
    
    print("\nAdjacency List Representation (with Tensor Shapes):")
    for sg_id, adj_list in enumerate(adjacency_list_subgraphs):
        sg_type = subgraphs[sg_id][1]
        op_type = "Supported" if sg_type == "mirage" else "Unsupported"
        print(f"Subgraph {sg_id} ({op_type}):")
        
        for op, op_info in adj_list.items():
            outputs = op_info["outputs"]
            input_shapes = op_info["input_tensor_shapes"]
            output_shapes = op_info["output_tensor_shapes"]
            
            print(f"  {op.name} (fn={op.fn}):")
            
            print("    Input Tensor Shapes:")
            for i, shape_info in enumerate(input_shapes):
                if shape_info:
                    shape, tensor_id = shape_info
                    is_external = i >= len(op.input_ops) or op.input_ops[i] is None
                    source = "External Input" if is_external else f"From {op.input_ops[i].name}"
                    print(f"      [{i}] {source}: shape={shape}, id={tensor_id}")
                else:
                    print(f"      [{i}] None")
            
            print("    Output Tensor Shapes:")
            for i, shape_info in enumerate(output_shapes):
                if shape_info:
                    shape, tensor_id = shape_info
                    print(f"      [{i}] shape={shape}, id={tensor_id}")
                else:
                    print(f"      [{i}] None")
            
            print("    Output Connections:")
            for i, out_op in enumerate(outputs):
                target_sg = "?"
                for other_sg_id, (other_sg, _) in enumerate(subgraphs):
                    if out_op in other_sg:
                        target_sg = other_sg_id
                        break
                
                if target_sg == sg_id:
                    print(f"      [{i}] -> {out_op.name}")
                else:
                    print(f"      [{i}] -> {out_op.name} (→ Subgraph {target_sg})")
            
            print()


    kernel_graphs = []
    input_dims = []

    try:
        print("\nGenerating Mirage kernel graphs...")
        for i, (sg, sg_type) in enumerate(subgraphs):
            if sg_type == "mirage":
                print(f"Processing subgraph {i}...")
                try:
                    kernel_graph, dims = splitter.to_kernel_graph(sg) 
                    kernel_graphs.append(kernel_graph)
                    input_dims.append(dims)
                    print(f"Successfully processed subgraph {i}")
                except Exception as e:
                    print(f"Error processing subgraph {i}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Skipping subgraph {i}: not supported by Mirage")
        
        print(f"Generated {len(kernel_graphs)} Mirage kernel graphs")
        
    except Exception as e:
        print(f"Error in kernel graph generation: {e}")
        import traceback
        traceback.print_exc()
    
    
    
    return subgraphs, subgraph_deps