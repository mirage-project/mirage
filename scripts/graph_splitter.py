import os
import sys
from collections import defaultdict, deque, Counter
from typing import Dict, List, Set, Tuple, Any, Optional, NamedTuple
from dataclasses import dataclass, field

@dataclass
class SubgraphData:
    """Represents a subgraph with detailed I/O information"""
    id: int
    type: str  # "mirage" or "pytorch"
    operators: Dict[Any, bool]
    input_edges: Dict[Any, List[Tuple[int, Any]]] = field(default_factory=dict)
    output_edges: Dict[Any, List[Tuple[int, Any]]] = field(default_factory=dict)

class GraphSplitter:
    """
    Four-stage graph splitting and optimization algorithm for mixed operator graphs.
    
    Stage 1: Conservative splitting for correctness
      - Topological ordering with conservative merging
      - Only merge when: 1 same-type upstream + 0 cross-type inputs
      - Guarantees acyclic and connected subgraphs
      - May produce smaller fragment subgraphs (intentional for correctness)
    
    Stage 2: First merge optimization
      - Merge same-type subgraphs with cycle safety checks
      
    Stage 3: Demote small Mirage subgraphs
      - Convert small Mirage subgraphs to PyTorch
      - Reduces overhead and creates more merge opportunities
      
    Stage 4: Second merge optimization
      - Merge again (now including demoted subgraphs)
      - Produces larger, more efficient PyTorch regions
    
    Architecture benefits:
    - Separation of concerns: correctness vs optimization
    - Iterative refinement: merge -> demote -> merge
    - Easy to debug and extend
    """
    
    def __init__(self, unsupported_ops=None):
        self.mirage_unsupported_ops = set()
        if unsupported_ops:
            self.mirage_unsupported_ops = {op.lower() for op in unsupported_ops}
    
    def is_supported_op(self, op) -> bool:
        """Check if an operation is supported by Mirage."""
        if not hasattr(op, 'fn'):
            return False
        
        op_type = op.fn
        if isinstance(op_type, str):
            op_type_lower = op_type.lower()
            if any(unsupported_op in op_type_lower for unsupported_op in self.mirage_unsupported_ops):
                return False
        else:
            return False
        
        # Check for 4D tensors
        from partition_graph import contains_4D_tensors
        if contains_4D_tensors(op):
            return False
        
        # Check for broadcast operations
        if hasattr(op, 'input_tensor_shapes') and op.input_tensor_shapes:
            valid_shapes = [s[0] for s in op.input_tensor_shapes if s and s[0]]
            if not valid_shapes:
                return True
                
            input_dims = len(valid_shapes[0])
            for shape in valid_shapes[1:]:
                if len(shape) != input_dims:
                    return False
        
        return True
    
    def split_graph(self, operators_graph: Dict) -> Tuple[List[Tuple[Dict, str]], Dict[int, Set[int]], List[Dict], List[Any]]:
        """
        Stage 1: Conservative graph splitting algorithm.
        
        Strategy: Only merge when there is a SINGLE upstream subgraph of the same type
        - len(same_type_input_sgs) == 1: merge into unique upstream (safe)
        - len(same_type_input_sgs) == 0: create new subgraph (start point)
        - len(same_type_input_sgs) > 1:  create new subgraph (merge point, conservative)
        
        Why conservative?
        - Creates new subgraphs at merge points to avoid complex dependencies
        - 100% guarantees acyclic and connected properties
        - Trade-off: may produce fragment subgraphs (handled by Stage 2)
        
        Returns:
            subgraphs: List of (subgraph_dict, type)
            subgraph_deps: Dict mapping subgraph_id to its dependency subgraph_ids
            subgraph_io: List of I/O relationship dicts
            sorted_ops: Topologically sorted operators
        """
        sorted_ops = self._topological_sort(operators_graph)
        
        subgraphs = []
        op_to_subgraph_id = {}
        subgraph_deps = defaultdict(set)
        
        for op in sorted_ops:
            op_type = "mirage" if self.is_supported_op(op) else "pytorch"
            
            same_type_input_sgs = set()
            diff_type_input_sgs = set()
            
            for input_op in op.input_ops:
                if input_op in op_to_subgraph_id:
                    input_sg_id = op_to_subgraph_id[input_op]
                    input_sg_type = subgraphs[input_sg_id][1]
                    
                    if input_sg_type == op_type:
                        same_type_input_sgs.add(input_sg_id)
                    else:
                        diff_type_input_sgs.add(input_sg_id)

            # Decision: merge or create new
            if len(same_type_input_sgs) == 1 and len(diff_type_input_sgs) == 0:
                # Safe merge: only one upstream of same type
                target_sg_id = list(same_type_input_sgs)[0]
                subgraphs[target_sg_id][0][op] = True
                op_to_subgraph_id[op] = target_sg_id
                
                for dep_id in diff_type_input_sgs:
                    subgraph_deps[target_sg_id].add(dep_id)
            else:
                # Create new subgraph: either start point or merge point
                new_sg_id = len(subgraphs)
                subgraphs.append(({op: True}, op_type))
                op_to_subgraph_id[op] = new_sg_id
                
                # Add all dependencies (same-type and cross-type)
                all_input_sgs = same_type_input_sgs.union(diff_type_input_sgs)
                for dep_id in all_input_sgs:
                    subgraph_deps[new_sg_id].add(dep_id)

        subgraph_io = self._build_io_relationships(subgraphs, op_to_subgraph_id)
        
        # Verify acyclicity
        deps_dict = dict(subgraph_deps)
        is_acyclic, cycle = self._verify_acyclic(deps_dict)
        if not is_acyclic:
            print("\n" + "="*60)
            print("ERROR: Stage 1 produced cycle!")
            print("="*60)
            print(f"Cycle: {' -> '.join(map(str, cycle))}")
            raise ValueError(f"Stage 1 algorithm bug: produced cycle {cycle}")
        
        print("\n[Stage 1] Conservative splitting complete: acyclic and connected")
        print(f"  Total subgraphs: {len(subgraphs)}")
        mirage_count = sum(1 for _, (_, sg_type) in enumerate(subgraphs) if sg_type == "mirage")
        print(f"  Mirage subgraphs: {mirage_count}, PyTorch subgraphs: {len(subgraphs) - mirage_count}")
        
        return subgraphs, deps_dict, subgraph_io, sorted_ops
    
    def _topological_sort(self, operators_graph: Dict) -> List[Any]:
        """Topologically sort operators based on data flow dependencies."""
        in_degree = {op: 0 for op in operators_graph}
        adj = defaultdict(list)
        
        for op in operators_graph:
            if not hasattr(op, 'output_ops'):
                continue
            for output_op in op.output_ops:
                if output_op in operators_graph:
                    adj[op].append(output_op)
                    in_degree[output_op] += 1
        
        queue = deque([op for op, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for output_op in adj[current]:
                in_degree[output_op] -= 1
                if in_degree[output_op] == 0:
                    queue.append(output_op)
        
        if len(result) != len(operators_graph):
            cycle_nodes = [op.name if hasattr(op, 'name') else str(op) 
                          for op, count in in_degree.items() if count > 0]
            print(f"ERROR: Operator graph contains cycles. Nodes in cycle: {cycle_nodes[:10]}...")
            raise ValueError("Operator graph contains cycles")
            
        return result
    
    def demote_small_mirage_subgraphs(self, subgraphs: List[Tuple[Dict, str]], 
                                      max_size_to_demote: int = 1) -> List[Tuple[Dict, str]]:
        """
        Stage 3: Demote small Mirage subgraphs to PyTorch.
        
        Rationale: Very small Mirage subgraphs (especially size-1) may not be worth
        the overhead of Mirage execution and subgraph switching. Demoting them to
        PyTorch allows them to merge with surrounding PyTorch subgraphs in Stage 4.
        
        Args:
            subgraphs: Subgraphs from Stage 2
            max_size_to_demote: Demote Mirage subgraphs with size <= this value
            
        Returns:
            Modified subgraphs with small Mirage subgraphs demoted to PyTorch
        """
        demoted_count = 0
        
        for sg_id, (sg_dict, sg_type) in enumerate(subgraphs):
            if sg_type == "mirage" and len(sg_dict) <= max_size_to_demote:
                # Demote to PyTorch
                subgraphs[sg_id] = (sg_dict, "pytorch")
                demoted_count += 1
        
        if demoted_count > 0:
            print(f"  Demoted {demoted_count} small Mirage subgraphs (size <= {max_size_to_demote}) to PyTorch")
            mirage_count = sum(1 for _, (_, sg_type) in enumerate(subgraphs) if sg_type == "mirage")
            print(f"  Remaining Mirage subgraphs: {mirage_count}")
        else:
            print(f"  No small Mirage subgraphs to demote")
        
        return subgraphs
    
    def merge_subgraphs(self, subgraphs: List[Tuple[Dict, str]], 
                       subgraph_deps: Dict[int, Set[int]], 
                       subgraph_io: List[Dict],
                       max_iterations: int = 10) -> Tuple[List[Tuple[Dict, str]], Dict[int, Set[int]], List[Dict]]:
        """
        Merge optimization - merge fragment subgraphs.
        
        Strategy:
        1. Attempt to merge subgraphs into upstream subgraphs of the same type
        2. Perform cycle check before each merge (safety brake)
        3. Iterate until no more merges possible
        
        Args:
            subgraphs: Input subgraphs
            subgraph_deps: Subgraph dependencies
            subgraph_io: Subgraph I/O relationships
            max_iterations: Maximum number of iterations
            
        Returns:
            Optimized (subgraphs, subgraph_deps, subgraph_io)
        """
        merged_count = 0
        iteration = 0
        
        while iteration < max_iterations:
            did_merge = False
            iteration += 1
            
            for sg_id in range(len(subgraphs)):
                if subgraphs[sg_id] is None:
                    continue
                
                sg_dict, sg_type = subgraphs[sg_id]
                
                if sg_id not in subgraph_deps:
                    continue
                
                # Find merge target: upstream subgraph of same type
                candidate = None
                for dep_id in subgraph_deps[sg_id]:
                    if subgraphs[dep_id] is not None:
                        dep_sg_dict, dep_sg_type = subgraphs[dep_id]
                        if dep_sg_type == sg_type:
                            candidate = dep_id
                            break
                
                if candidate is None:
                    continue
                    
                # Safety brake: check if merge would create cycle
                other_deps = subgraph_deps[sg_id] - {candidate}
                is_safe = True
                
                for other_dep_id in other_deps:
                    if self._has_path(subgraph_deps, other_dep_id, candidate):
                        is_safe = False
                        break
                
                if is_safe:
                    self._merge_into(subgraphs, sg_id, candidate, subgraph_deps, subgraph_io)
                    merged_count += 1
                    did_merge = True
                    break
            
            if not did_merge:
                break
        
        # Compact subgraphs
        subgraphs, subgraph_deps, subgraph_io = self._compact_merged_subgraphs(
            subgraphs, subgraph_deps, subgraph_io
        )
        
        print(f"  Merged {merged_count} subgraphs")
        print(f"  Result: {len(subgraphs)} subgraphs")
        
        return subgraphs, subgraph_deps, subgraph_io
    
    def _merge_into(self, subgraphs, source_id, target_id, subgraph_deps, subgraph_io):
        """Merge source subgraph into target subgraph."""
        source_dict, source_type = subgraphs[source_id]
        target_dict, target_type = subgraphs[target_id]
        
        # Merge operators
        for op in source_dict:
            target_dict[op] = True
        
        # Update dependencies
        if source_id in subgraph_deps:
            if target_id not in subgraph_deps:
                subgraph_deps[target_id] = set()
            
            for dep_id in subgraph_deps[source_id]:
                if dep_id != target_id:
                    subgraph_deps[target_id].add(dep_id)
            del subgraph_deps[source_id]
        
        # Update other subgraphs that depend on source
        for sg_id, deps in subgraph_deps.items():
            if source_id in deps:
                deps.remove(source_id)
                if sg_id != target_id:
                    deps.add(target_id)
        
        subgraphs[source_id] = None
    
    def _compact_merged_subgraphs(self, subgraphs, subgraph_deps, subgraph_io):
        """Remove None subgraphs and renumber IDs."""
        compact_subgraphs = []
        id_mapping = {}
        
        for old_id, sg_tuple in enumerate(subgraphs):
            if sg_tuple is not None:
                new_id = len(compact_subgraphs)
                id_mapping[old_id] = new_id
                compact_subgraphs.append(sg_tuple)
        
        # Update dependencies
        compact_deps = {}
        for old_id, deps in subgraph_deps.items():
            if old_id in id_mapping:
                new_id = id_mapping[old_id]
                compact_deps[new_id] = {id_mapping[d] for d in deps if d in id_mapping}
        
        # Update I/O
        compact_io = [subgraph_io[old_id] if old_id < len(subgraph_io) else {"inputs": {}, "outputs": {}} 
                      for old_id in range(len(subgraphs)) if old_id in id_mapping]
        
        return compact_subgraphs, compact_deps, compact_io
    
    def _has_path(self, dependencies: Dict[int, Set[int]], start: int, target: int) -> bool:
        """Check if there exists a path from start to target in the dependency graph."""
        if start == target:
            return True
        
        visited = set()
        queue = [start]
        
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            
            if node in dependencies:
                for dep in dependencies[node]:
                    if dep == target:
                        return True
                    if dep not in visited:
                        queue.append(dep)
        
        return False
    
    def _verify_acyclic(self, dependencies: Dict[int, Set[int]]) -> Tuple[bool, List[int]]:
        """
        Verify that the dependency graph is acyclic.
        
        Returns:
            (is_acyclic, cycle): (True, []) if acyclic, (False, [cycle_path]) otherwise
        """
        all_nodes = set(dependencies.keys())
        for deps in dependencies.values():
            all_nodes.update(deps)
        
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in all_nodes}
        parent = {node: None for node in all_nodes}
        cycle_path = []
        
        def dfs(node):
            nonlocal cycle_path
            color[node] = GRAY
            
            if node in dependencies:
                for dep in dependencies[node]:
                    if color[dep] == GRAY:
                        cycle_path = []
                        curr = node
                        while curr != dep:
                            cycle_path.append(curr)
                            curr = parent[curr]
                            if curr is None:
                                cycle_path = [dep, node]
                                break
                        cycle_path.append(dep)
                        cycle_path.reverse()
                        cycle_path.append(node)
                        return True
                    elif color[dep] == WHITE:
                        parent[dep] = node
                        if dfs(dep):
                            return True
            
            color[node] = BLACK
            return False
        
        for node in all_nodes:
            if color[node] == WHITE:
                if dfs(node):
                    return False, cycle_path
        
        return True, []
    
    def _build_io_relationships(self, subgraphs, op_to_subgraph):
        """Build I/O relationships between subgraphs."""
        subgraph_io = [{"inputs": {}, "outputs": {}} for _ in subgraphs]
        
        for sg_id, (sg, _) in enumerate(subgraphs):
            for op in sg:
                for input_op in op.input_ops:
                    if input_op in op_to_subgraph:
                        input_sg_id = op_to_subgraph[input_op]
                        if input_sg_id != sg_id:
                            if op not in subgraph_io[sg_id]["inputs"]:
                                subgraph_io[sg_id]["inputs"][op] = []
                            subgraph_io[sg_id]["inputs"][op].append((input_sg_id, input_op))
                            
                            if input_op not in subgraph_io[input_sg_id]["outputs"]:
                                subgraph_io[input_sg_id]["outputs"][input_op] = []
                            subgraph_io[input_sg_id]["outputs"][input_op].append((sg_id, op))
        
        return subgraph_io
    
    def _print_subgraph_stats(self, subgraphs):
        """Print subgraph statistics."""
        mirage_subgraphs = [(sg_id, sg) for sg_id, (sg, sg_type) in enumerate(subgraphs) if sg_type == "mirage"]
        pytorch_subgraphs = [(sg_id, sg) for sg_id, (sg, sg_type) in enumerate(subgraphs) if sg_type == "pytorch"]
        
        mirage_sizes = Counter(len(sg) for _, sg in mirage_subgraphs)
        pytorch_sizes = Counter(len(sg) for _, sg in pytorch_subgraphs)
        
        print(f"\nTotal subgraphs: {len(subgraphs)} ({len(mirage_subgraphs)} mirage, {len(pytorch_subgraphs)} pytorch)")
        
        if mirage_sizes:
            print(f"Mirage subgraph size distribution: {dict(sorted(mirage_sizes.items()))}")
            total_mirage_ops = sum(len(sg) for _, sg in mirage_subgraphs)
            avg_mirage = total_mirage_ops / len(mirage_subgraphs) if mirage_subgraphs else 0
            print(f"  Total Mirage ops: {total_mirage_ops}, Avg: {avg_mirage:.1f} ops/subgraph")
        
        if pytorch_sizes:
            print(f"PyTorch subgraph size distribution: {dict(sorted(pytorch_sizes.items()))}")
            total_pytorch_ops = sum(len(sg) for _, sg in pytorch_subgraphs)
            avg_pytorch = total_pytorch_ops / len(pytorch_subgraphs) if pytorch_subgraphs else 0
            print(f"  Total PyTorch ops: {total_pytorch_ops}, Avg: {avg_pytorch:.1f} ops/subgraph")
    
    def _print_subgraph_info(self, subgraphs, subgraph_io):
        """Print detailed subgraph information."""
        for sg_id, (sg, sg_type) in enumerate(subgraphs):
            print(f"\nSubgraph {sg_id} ({sg_type}):")
            print(f"  Operators: {[op.name if hasattr(op, 'name') else str(op) for op in sg]}")
            
            print("  Inputs (from other subgraphs):")
            for op, inputs in subgraph_io[sg_id]["inputs"].items():
                for src_sg_id, src_op in inputs:
                    op_name = op.name if hasattr(op, 'name') else str(op)
                    src_op_name = src_op.name if hasattr(src_op, 'name') else str(src_op)
                    print(f"    {op_name} <- Subgraph {src_sg_id} ({src_op_name})")
            
            print("  Outputs (to other subgraphs):")
            for op, outputs in subgraph_io[sg_id]["outputs"].items():
                for tgt_sg_id, tgt_op in outputs:
                    op_name = op.name if hasattr(op, 'name') else str(op)
                    tgt_op_name = tgt_op.name if hasattr(tgt_op, 'name') else str(tgt_op)
                    print(f"    {op_name} -> Subgraph {tgt_sg_id} ({tgt_op_name})")
    
    def create_mirage_graph(self, subgraphs, subgraph_io):
        """Create a Mirage execution graph from the subgraphs."""
        mirage_graph = {}
        
        for sg_id, (sg, sg_type) in enumerate(subgraphs):
            if sg_type == "mirage":
                kernel_id = f"mirage_kernel_{sg_id}"
                
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
        """Convert subgraphs into adjacency list format."""
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

    

def process_operator_graph(operators: Dict, 
                          IGNORE_OPS: Set[str] = None, 
                          UNSUPPORTED_OPS: Set[str] = None,
                          demotion_size_threshold: int = 1,
                          merge_iterations_round1: int = 100,
                          merge_iterations_round2: int = 100) -> Tuple[List[Tuple[Dict, str]], Dict[int, Set[int]], List[Any]]:
    """
    Multi-stage subgraph splitting and optimization algorithm.
    
    Pipeline:
      Stage 1: Conservative splitting (guaranteed correctness)
      Stage 2: Merge same-type subgraphs (first optimization)
      Stage 3: Demote small Mirage subgraphs to PyTorch
      Stage 4: Merge again (second optimization, now with demoted subgraphs)
    
    Args:
        operators: Operator dictionary
        IGNORE_OPS: Set of operator names to ignore/remove
        UNSUPPORTED_OPS: Set of operator names not supported by Mirage
        demotion_size_threshold: Demote Mirage subgraphs with size <= this value (default 2)
        merge_iterations_round1: Max iterations for Stage 2 (default 100)
        merge_iterations_round2: Max iterations for Stage 4 (default 100)
        
    Returns:
        Tuple[List, Dict, List]: Subgraphs, dependencies, sorted operators
    """
    operators_graph = {op: True for op in operators.values()}
    operators_graph = preprocess_special_operators(operators_graph, IGNORE_OPS)
    
    splitter = GraphSplitter(UNSUPPORTED_OPS)
    
    print("\n" + "="*70)
    print("Four-Stage Subgraph Splitting Algorithm")
    print("="*70)
    
    # Stage 1: Conservative splitting
    subgraphs, subgraph_deps, subgraph_io, sorted_ops = splitter.split_graph(operators_graph)
    
    # Stage 2: First merge pass
    print(f"\n[Stage 2] Merge same-type subgraphs...")
    subgraphs, subgraph_deps, subgraph_io = splitter.merge_subgraphs(
        subgraphs, subgraph_deps, subgraph_io, 
        max_iterations=merge_iterations_round1
    )
    
    # Stage 3: Demotion of small Mirage subgraphs
    print(f"\n[Stage 3] Demote small Mirage subgraphs (size <= {demotion_size_threshold})...")
    subgraphs = splitter.demote_small_mirage_subgraphs(
        subgraphs, 
        max_size_to_demote=demotion_size_threshold
    )
    
    # Rebuild I/O relationships after demotion (types changed)
    op_to_subgraph = {}
    for sg_id, (sg_dict, _) in enumerate(subgraphs):
        for op in sg_dict:
            op_to_subgraph[op] = sg_id
    subgraph_io = splitter._build_io_relationships(subgraphs, op_to_subgraph)
    
    # Stage 4: Second merge pass (with demoted subgraphs)
    print(f"\n[Stage 4] Merge again (including demoted subgraphs)...")
    subgraphs, subgraph_deps, subgraph_io = splitter.merge_subgraphs(
        subgraphs, subgraph_deps, subgraph_io, 
        max_iterations=merge_iterations_round2
    )
    
    # Final statistics
    print("\n" + "="*70)
    print("Final Subgraph Statistics")
    print("="*70)
    splitter._print_subgraph_stats(subgraphs)
    
    # Convert to adjacency list format
    adjacency_list_subgraphs = splitter.convert_to_adjacency_list(subgraphs)
    
    # # Print detailed adjacency list
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
    if IGNORE_OPS is None:
        IGNORE_OPS = {"Identity", "Cast", "Constant", "Dropout"}
    
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
    ignore_ops_lower = {op.lower() for op in IGNORE_OPS}
    
    ignored_ops = [
        op for op in operators_graph 
        if hasattr(op, 'fn') and 
        (isinstance(op.fn, str) and op.fn.lower() in ignore_ops_lower or
         hasattr(op.fn, 'lower') and op.fn.lower() in ignore_ops_lower)
    ]
    
    if not ignored_ops:
        return operators_graph
    
    print(f"Removing {len(ignored_ops)} ignored operators")
    
    replacements = {}
    
    for ig_op in ignored_ops:
        if not hasattr(ig_op, 'output_ops') or not hasattr(ig_op, 'input_ops'):
            continue
            
        source_tensors = []
        if hasattr(ig_op, 'input_tensor_shapes') and ig_op.input_ops:
            for i, shape in enumerate(ig_op.input_tensor_shapes):
                if i < len(ig_op.input_ops) and shape:
                    source_tensors.append((ig_op.input_ops[i], shape[1]))
        
        if not source_tensors:
            continue
            
        # Find which input of downstream ops should be replaced by matching tensor IDs
        for out_op in ig_op.output_ops:
            ignored_output_tid = ig_op.output_tensor_shapes[0][1] if ig_op.output_tensor_shapes else None
            
            if ignored_output_tid is not None:
                for j, (shape, tid) in enumerate(out_op.input_tensor_shapes):
                    if tid == ignored_output_tid:
                        replacements[(out_op, j)] = source_tensors[0]
                        break
    
    # Apply replacements: update both input_ops and input_tensor_shapes
    for (out_op, input_idx), (source_op, tensor_id) in replacements.items():
        out_op.input_ops[input_idx] = source_op
        
        # Update input_tensor_shapes with the correct tensor ID from source
        if hasattr(source_op, 'output_tensor_shapes') and source_op.output_tensor_shapes:
            for shape in source_op.output_tensor_shapes:
                if shape and shape[1] == tensor_id:
                    out_op.input_tensor_shapes[input_idx] = shape
                    break
        else:
            # For parameter/constant nodes without output_tensor_shapes, keep shape but update tensor ID
            if input_idx < len(out_op.input_tensor_shapes):
                old_shape = out_op.input_tensor_shapes[input_idx]
                out_op.input_tensor_shapes[input_idx] = (old_shape[0], tensor_id)
    
    result = {op: True for op in operators_graph if op not in ignored_ops}
    
    # Clean up input_ops/output_ops AND synchronize input_tensor_shapes to keep them aligned
    for op in result:
        op.output_ops = [next_op for next_op in op.output_ops if next_op in result]
        
        # Filter input_ops and simultaneously filter input_tensor_shapes to keep them in sync
        new_input_ops = []
        new_input_tensor_shapes = []
        for i, prev_op in enumerate(op.input_ops):
            # Keep this input if it's in the result graph OR it's a parameter/constant node
            if prev_op in result or prev_op not in operators_graph:
                new_input_ops.append(prev_op)
                if i < len(op.input_tensor_shapes):
                    new_input_tensor_shapes.append(op.input_tensor_shapes[i])
        
        op.input_ops = new_input_ops
        op.input_tensor_shapes = new_input_tensor_shapes
    
    # Update output_ops of source operators
    for (out_op, input_idx), (source_op, tensor_id) in replacements.items():
        if source_op in result and out_op in result and out_op not in source_op.output_ops:
            source_op.output_ops.append(out_op)
        
    return result
