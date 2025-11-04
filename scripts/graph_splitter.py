import os
import sys
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
    
    def __init__(self, unsupported_ops=None):
        # Operations that are NOT supported by Mirage (from user input)
        self.mirage_unsupported_ops = set()
        if unsupported_ops:
            # Convert all names to lowercase for case-insensitive matching
            self.mirage_unsupported_ops = {op.lower() for op in unsupported_ops}
    
    def is_supported_op(self, op) -> bool:
        """Check if an operation is supported by Mirage."""
        if not hasattr(op, 'fn'):
            return False
        
        op_type = op.fn
        if isinstance(op_type, str):
            op_type_lower = op_type.lower()
            # If operation is in the unsupported list, it's not supported
            if any(unsupported_op in op_type_lower for unsupported_op in self.mirage_unsupported_ops):
                return False
        else:
            return False
        
        # Check for 4D tensors and broadcast operations (not supported)
        from partition_graph import contains_4D_tensors
        if contains_4D_tensors(op):
            return False
        
        if hasattr(op, 'input_tensor_shapes') and op.input_tensor_shapes:
            input_dims = len(op.input_tensor_shapes[0][0])
            for s in op.input_tensor_shapes:
                if len(s[0]) != input_dims:
                    return False
        
        return True
    
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
        # self._print_subgraph_info(subgraphs, subgraph_io)
        
        return subgraphs, subgraph_deps, subgraph_io, sorted_ops
    
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
            in_degree[node] = len(deps)
        
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
        2. Select the largest splittable subgraph in the cycle (with >1 operator)
        3. Split the selected subgraph into two parts
        4. Rebuild I/O relationships and dependencies
        5. Repeat until no cycles remain
        """
        topo_indices = {op: idx for idx, op in enumerate(sorted_ops)}
        
        # Try to break cycles a limited number of times
        attempts = 0
        max_attempts = 100
        
        while not self._is_acyclic(subgraph_deps) and attempts < max_attempts:
            attempts += 1
            
            cycle = self._find_cycle(subgraph_deps)
            if not cycle:
                break
            
            # Identify the back edge: cycle[-2] -> cycle[0]
            back_edge_src = cycle[-2] if len(cycle) > 1 else cycle[0]
            back_edge_tgt = cycle[0]
            
            # Strategy: Precisely split the target of the back edge
            # Find all ops in target that depend on source
            if subgraphs[back_edge_tgt] is None or len(subgraphs[back_edge_tgt][0]) <= 1:
                # Target can't be split, try source instead
                back_edge_src, back_edge_tgt = back_edge_tgt, back_edge_src
            
            if subgraphs[back_edge_tgt] is None or len(subgraphs[back_edge_tgt][0]) <= 1:
                # Neither can be split, try largest in cycle
                subgraph_to_split = None
                max_size = 0
                for sg_id in cycle:
                    if subgraphs[sg_id] is not None:
                        sg_dict, _ = subgraphs[sg_id]
                        if len(sg_dict) > 1 and len(sg_dict) > max_size:
                            max_size = len(sg_dict)
                            subgraph_to_split = sg_id
                
                if subgraph_to_split is None:
                    print(f"Warning: Cannot break cycle {cycle} - all subgraphs have only one operator")
                    break
                
                sg_dict, sg_type = subgraphs[subgraph_to_split]
                ops_in_sg = list(sg_dict.keys())
                ops_in_sg.sort(key=lambda op: topo_indices[op])
                split_point = len(ops_in_sg) // 2
                first_half = {op: True for op in ops_in_sg[:split_point]}
                second_half = {op: True for op in ops_in_sg[split_point:]}
            else:
                # Precise split: separate ops that depend on back_edge_src
                subgraph_to_split = back_edge_tgt
                sg_dict, sg_type = subgraphs[back_edge_tgt]
                
                # Find ops that depend on back_edge_src via subgraph_io
                ops_with_src_input = set()
                if back_edge_tgt < len(subgraph_io):
                    for op, inputs in subgraph_io[back_edge_tgt]["inputs"].items():
                        for input_sg_id, _ in inputs:
                            if input_sg_id == back_edge_src:
                                ops_with_src_input.add(op)
                                break
                
                # Split: ops that depend on src vs ops that don't
                ops_to_move = [op for op in sg_dict.keys() if op in ops_with_src_input]
                ops_to_keep = [op for op in sg_dict.keys() if op not in ops_with_src_input]
                
                # If all ops depend on src or none depend on src, fall back to middle split
                if not ops_to_move or not ops_to_keep:
                    ops_in_sg = list(sg_dict.keys())
                    ops_in_sg.sort(key=lambda op: topo_indices[op])
                    split_point = len(ops_in_sg) // 2
                    first_half = {op: True for op in ops_in_sg[:split_point]}
                    second_half = {op: True for op in ops_in_sg[split_point:]}
                else:
                    first_half = {op: True for op in ops_to_keep}
                    second_half = {op: True for op in ops_to_move}
            
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
        
        if not self._is_acyclic(subgraph_deps):
            final_cycle = self._find_cycle(subgraph_deps)
            print(f"Warning: Cycles still exist after {attempts} split attempts")
        
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
    
    def _print_subgraph_stats(self, subgraphs):
        """Print subgraph statistics."""
        from collections import Counter
        
        mirage_subgraphs = [(sg_id, sg) for sg_id, (sg, sg_type) in enumerate(subgraphs) if sg_type == "mirage"]
        pytorch_subgraphs = [(sg_id, sg) for sg_id, (sg, sg_type) in enumerate(subgraphs) if sg_type == "pytorch"]
        
        # Count by size
        mirage_sizes = Counter(len(sg) for _, sg in mirage_subgraphs)
        pytorch_sizes = Counter(len(sg) for _, sg in pytorch_subgraphs)
        
        print(f"Total subgraphs: {len(subgraphs)} ({len(mirage_subgraphs)} mirage, {len(pytorch_subgraphs)} pytorch)")
        
        if mirage_sizes:
            print(f"Mirage subgraph sizes: {dict(sorted(mirage_sizes.items()))}")
            total_mirage_ops = sum(len(sg) for _, sg in mirage_subgraphs)
            avg_mirage = total_mirage_ops / len(mirage_subgraphs) if mirage_subgraphs else 0
            print(f"  Total mirage ops: {total_mirage_ops}, Avg: {avg_mirage:.1f} ops/subgraph")
        
        if pytorch_sizes:
            print(f"PyTorch subgraph sizes: {dict(sorted(pytorch_sizes.items()))}")
            total_pytorch_ops = sum(len(sg) for _, sg in pytorch_subgraphs)
            avg_pytorch = total_pytorch_ops / len(pytorch_subgraphs) if pytorch_subgraphs else 0
            print(f"  Total pytorch ops: {total_pytorch_ops}, Avg: {avg_pytorch:.1f} ops/subgraph")
    
    def _print_subgraph_info(self, subgraphs, subgraph_io):
        """Print detailed subgraph information."""
        for sg_id, (sg, sg_type) in enumerate(subgraphs):
            print(f"\nSubgraph {sg_id} ({sg_type}):")
            print(f"  Operators: {[op.name for op in sg]}")
            
            # print("  Inputs:")
            # for op, inputs in subgraph_io[sg_id]["inputs"].items():
            #     for src_sg_id, src_op in inputs:
            #         print(f"    {op.name} takes input from Subgraph {src_sg_id}'s {src_op.name}")
            
            # print("  Outputs:")
            # for op, outputs in subgraph_io[sg_id]["outputs"].items():
            #     for tgt_sg_id, tgt_op in outputs:
            #         print(f"    {op.name} sends output to Subgraph {tgt_sg_id}'s {tgt_op.name}")
    
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

    

def process_operator_graph(operators: Dict, IGNORE_OPS: Set[str] = None, UNSUPPORTED_OPS: Set[str] = None) -> Tuple[List[Tuple[Dict, str]], Dict[int, Set[int]]]:
    """
    Process an operator graph and split it into Mirage and PyTorch subgraphs.
    
    Args:
        operators: Operators
        IGNORE_OPS: Set of operator names to ignore/remove
        UNSUPPORTED_OPS: Set of operator names that are not supported by Mirage
        
    Returns:
        Tuple[List, Dict]: List of subgraphs and their dependencies
    """
    operators_graph = {op: True for op in operators.values()}
    # Preprocess the graph to handle special operators
    operators_graph = preprocess_special_operators(operators_graph, IGNORE_OPS)
    
    splitter = GraphSplitter(UNSUPPORTED_OPS)
    
    print("Splitting operator graph into subgraphs...")
    subgraphs, subgraph_deps, subgraph_io, sorted_ops = splitter.split_graph(operators_graph)
    
    adjacency_list_subgraphs = splitter.convert_to_adjacency_list(subgraphs)
    
    mirage_subgraphs_count = sum(1 for _, sg_type in subgraphs if sg_type == "mirage")
    pytorch_subgraphs_count = sum(1 for _, sg_type in subgraphs if sg_type == "pytorch")
    
    print(f"Subgraph Statistics:")
    print(f"  - Mirage subgraphs: {mirage_subgraphs_count}")
    print(f"  - PyTorch subgraphs: {pytorch_subgraphs_count}")
    print(f"  - Total subgraphs: {len(subgraphs)}")
    
    # print("\nAdjacency List Representation (with Tensor Shapes and Connections):")
    # for sg_id, adj_list in enumerate(adjacency_list_subgraphs):
    #     sg_type = subgraphs[sg_id][1]
    #     op_type = "Supported" if sg_type == "mirage" else "Unsupported"
    #     print(f"Subgraph {sg_id} ({op_type}):")
        
    #     for op, op_info in adj_list.items():
    #         outputs = op_info["outputs"]
    #         input_shapes = op_info["input_tensor_shapes"]
    #         output_shapes = op_info["output_tensor_shapes"]
            
    #         print(f"  {op.name} (fn={op.fn}):")
            
    #         # Print input operations
    #         print("    Input Operations:")
    #         for i, in_op in enumerate(op.input_ops):
    #             print(f"      [{i}] {in_op.name}")
            
    #         print("    Input Tensor Shapes:")
    #         for i, shape_info in enumerate(input_shapes):
    #             if shape_info:
    #                 shape, tensor_id = shape_info
    #                 is_external = i >= len(op.input_ops) or op.input_ops[i] is None
    #                 source = "External Input" if is_external else f"From {op.input_ops[i].name}"
    #                 print(f"      [{i}] {source}: shape={shape}, id={tensor_id}")
    #             else:
    #                 print(f"      [{i}] None")
            
    #         print("    Output Tensor Shapes:")
    #         for i, shape_info in enumerate(output_shapes):
    #             if shape_info:
    #                 shape, tensor_id = shape_info
    #                 print(f"      [{i}] shape={shape}, id={tensor_id}")
    #             else:
    #                 print(f"      [{i}] None")
            
    #         print("    Output Operations:")
    #         for i, out_op in enumerate(outputs):
    #             target_sg = "?"
    #             for other_sg_id, (other_sg, _) in enumerate(subgraphs):
    #                 if out_op in other_sg:
    #                     target_sg = other_sg_id
    #                     break
                
    #             if target_sg == sg_id:
    #                 print(f"      [{i}] -> {out_op.name}")
    #             else:
    #                 print(f"      [{i}] -> {out_op.name} (→ Subgraph {target_sg})")
            
    #         print()

    
    return subgraphs, subgraph_deps, sorted_ops

def preprocess_special_operators(operators_graph: Dict, IGNORE_OPS: Set[str] = None) -> Dict:
    """
    Preprocess the graph to handle special operators that need custom treatment.
    
    Currently handles:
    - Ignored operators: removed and connections bypassed
    
    Args:
        operators_graph: Dict mapping operators to boolean values
        IGNORE_OPS: Set of operator names to ignore/remove
        
    Returns:
        Dict: Processed operator graph
    """
    # If no IGNORE_OPS provided, use default set
    if IGNORE_OPS is None:
        IGNORE_OPS = {"Identity", "Cast", "Constant", "Dropout"}
    
    # Apply preprocessing to remove ignored operators
    processed_graph = remove_ignored_operators(operators_graph, IGNORE_OPS)
    
    return processed_graph

def remove_ignored_operators(operators_graph: Dict, IGNORE_OPS: Set[str]) -> Dict:
    """
    Remove ignored operators from the graph and fix connections.
    
    Args:
        operators_graph: Dict mapping operators to boolean values
        IGNORE_OPS: Set of operator names to ignore/remove
        
    Returns:
        Dict: Cleaned operator graph without ignored operators
    """
    # Convert all ignore ops to lowercase for case-insensitive matching
    ignore_ops_lower = {op.lower() for op in IGNORE_OPS}
    
    # Find operators to ignore
    ignored_ops = [
        op for op in operators_graph 
        if hasattr(op, 'fn') and 
        (isinstance(op.fn, str) and op.fn.lower() in ignore_ops_lower or
         hasattr(op.fn, 'lower') and op.fn.lower() in ignore_ops_lower)
    ]
    
    if not ignored_ops:
        return operators_graph
    
    print(f"Removing {len(ignored_ops)} ignored operators")
    
    # CRITICAL PART: Find all connections that need bypass 
    # For each output operator that takes input from an ignored operator,
    # we need to find the original source operator and its output tensor
    
    # First, create a mapping for each output operator's inputs that come from ignored ops
    replacements = {}  # {(out_op, input_idx): (source_op, tensor_id)}
    
    for ig_op in ignored_ops:
        # Only handle operators with valid inputs/outputs
        if not hasattr(ig_op, 'output_ops') or not hasattr(ig_op, 'input_ops'):
            continue
            
        # Find the source of this ignored operator's input tensors
        source_tensors = []
        if hasattr(ig_op, 'input_tensor_shapes') and ig_op.input_ops:
            for i, shape in enumerate(ig_op.input_tensor_shapes):
                if i < len(ig_op.input_ops) and shape:
                    source_tensors.append((ig_op.input_ops[i], shape[1]))
        
        # If no valid source found, skip this ignored op
        if not source_tensors:
            continue
            
        # For all output operators, track which inputs need replacement
        for out_op in ig_op.output_ops:
            for j, in_op in enumerate(out_op.input_ops):
                if in_op == ig_op and j < len(out_op.input_tensor_shapes) and source_tensors:
                    # Map this input to its original source (use first source as default)
                    replacements[(out_op, j)] = source_tensors[0]
    
    # Now apply all replacements
    for (out_op, input_idx), (source_op, tensor_id) in replacements.items():
        # Replace the input operator reference
        out_op.input_ops[input_idx] = source_op
        
        # Find the corresponding tensor shape from the source
        if hasattr(source_op, 'output_tensor_shapes'):
            for shape in source_op.output_tensor_shapes:
                if shape and shape[1] == tensor_id:
                    out_op.input_tensor_shapes[input_idx] = shape
                    break
    
    # Finally remove ignored ops from the graph
    result = {op: True for op in operators_graph if op not in ignored_ops}
    
    # Fix direct connections
    for op in result:
        op.output_ops = [next_op for next_op in op.output_ops if next_op in result]
        op.input_ops = [prev_op for prev_op in op.input_ops if prev_op in result]
    
    # Rebuild connections broken by ignored ops
    for (out_op, input_idx), (source_op, tensor_id) in replacements.items():
        if source_op in result and out_op in result and out_op not in source_op.output_ops:
            source_op.output_ops.append(out_op)
        
    return result