import random
from collections import deque, defaultdict
from typing import Dict, List, Set, Optional, Any, Tuple
import json
from partitioning.build_computation_graph import get_computation_graph

def _shallow_copy_adjacency(partition: Dict[Any, List[Any]]) -> Dict[Any, List[Any]]:
    """Create a shallow copy of adjacency list."""
    return {node: list(connections) for node, connections in partition.items()}


def _expand_partition_adjacency(partition: Dict[Any, List[Any]], 
                               all_operators: Dict[str, Any],
                               UNSUPPORTED_OPS: set,
                               IGNORE_OPS: set,
                               expansion_size: int = 2) -> Dict[Any, List[Any]]:
    """Expand a partition in adjacency list format by adding connected neighbors."""
    
    expanded = _shallow_copy_adjacency(partition)
    partition_nodes = set(partition.keys())
    
    # Find boundary candidates
    boundary_candidates = []
    
    for op_node in partition_nodes:
        # Check input neighbors
        for input_op in op_node.input_ops:
            if (hasattr(input_op, 'name') and 
                input_op.name in all_operators and
                hasattr(input_op, 'fn') and
                input_op.fn not in UNSUPPORTED_OPS and 
                input_op.fn not in IGNORE_OPS and
                input_op not in partition_nodes):
                boundary_candidates.append(input_op)
        
        # Check output neighbors  
        for output_op in op_node.output_ops:
            if (hasattr(output_op, 'name') and 
                output_op.name in all_operators and
                hasattr(output_op, 'fn') and
                output_op.fn not in UNSUPPORTED_OPS and
                output_op.fn not in IGNORE_OPS and
                output_op not in partition_nodes):
                boundary_candidates.append(output_op)
    
    # Randomly select nodes to add
    candidates = list(set(boundary_candidates))
    random.shuffle(candidates)
    
    added = 0
    for candidate in candidates:
        if added >= expansion_size:
            break
        
        expanded[candidate] = []
        
        # Add edges to existing nodes in the partition
        for existing_node in partition_nodes:
            # Check if candidate should connect to existing_node
            if candidate in existing_node.output_ops:
                expanded[existing_node].append(candidate)
            if candidate in existing_node.input_ops:
                expanded[candidate].append(existing_node)
        
        partition_nodes.add(candidate)
        added += 1
    
    return expanded


def _contract_partition_adjacency(partition: Dict[Any, List[Any]], 
                                 contraction_size: int = 1) -> Dict[Any, List[Any]]:
    """Contract a partition by removing leaf nodes."""
    
    if len(partition) <= contraction_size + 1:
        return partition
    
    contracted = _shallow_copy_adjacency(partition)
    partition_nodes = set(partition.keys())
    
    # Find leaf nodes (nodes with minimal connections within the partition)
    node_connections = {}
    for node in partition_nodes:
        connections = len(partition[node])  # outgoing connections
        # Count incoming connections
        for other_node, connections_list in partition.items():
            if node in connections_list:
                connections += 1
        node_connections[node] = connections
    
    # Sort by connection count and remove nodes with fewest connections
    sorted_nodes = sorted(node_connections.items(), key=lambda x: x[1])
    
    removed = 0
    for node, _ in sorted_nodes:
        if removed >= contraction_size:
            break
        
        # Remove the node and all references to it
        if node in contracted:
            del contracted[node]
            
            # Remove references to this node from other nodes
            for other_node, connections_list in contracted.items():
                if node in connections_list:
                    connections_list.remove(node)
            
            removed += 1
    
    return contracted


def _compute_partition_hash(partition):
    """
    Compute a hash for a partition based on structure and operation types.
    This allows us to identify duplicate subgraphs.
    """
    # Create a canonical representation
    nodes = []
    edges = []
    
    # Sort nodes by a stable property (name,fn)
    sorted_nodes = sorted(partition.keys(), key=lambda n: (getattr(n, 'name', ''), getattr(n, 'fn', '')))
    
    # Create node mapping
    node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    
    # Collect node features
    for node in sorted_nodes:
        node_repr = (
            getattr(node, 'fn', ''),
            tuple(getattr(node, 'input_tensor_shapes', [])),
            tuple(getattr(node, 'output_tensor_shapes', [])),
        )
        nodes.append(node_repr)
    
    # Collect edges (sorted for consistency)
    for from_node, to_nodes in partition.items():
        from_idx = node_to_idx[from_node]
        for to_node in to_nodes:
            if to_node in node_to_idx:
                to_idx = node_to_idx[to_node]
                edges.append((from_idx, to_idx))
    
    edges.sort()
    
    # Create hash from canonical representation
    canonical = (tuple(nodes), tuple(edges))
    return hash(canonical)


def prune_duplicate_subgraphs(subgraphs):
    """
    Remove duplicate subgraphs from the dataset.
    
    Args:
        subgraphs: List of subgraphs in adjacency list format
        
    Returns:
        List of unique subgraphs and count of duplicates removed
    """
    seen_hashes = set()
    unique_subgraphs = []
    
    for subgraph in subgraphs:
        subgraph_hash = _compute_partition_hash(subgraph)
        
        if subgraph_hash not in seen_hashes:
            seen_hashes.add(subgraph_hash)
            unique_subgraphs.append(subgraph)
    
    num_duplicates = len(subgraphs) - len(unique_subgraphs)
    
    return unique_subgraphs, num_duplicates

def get_total_partition_ops(partition: Dict[Any, List[Any]]) -> int:
    """
    total = operators + external_inputs + external_outputs
    """
    partition_nodes = set(partition.keys())
    
    # Count operators
    num_operators = len(partition_nodes)
    
    # Count external inputs (input_ops not in partition)
    external_inputs = set()
    for op_node in partition_nodes:
        for input_op in op_node.input_ops:
            if input_op not in partition_nodes:
                # This is an external input - will become a kernel input op
                external_inputs.add(id(input_op)) 
    
    # Count external outputs (output_ops not in partition)
    external_outputs = set()
    for op_node in partition_nodes:
        for output_op in op_node.output_ops:
            if output_op not in partition_nodes:
                # This is an external output - will become a kernel output op
                external_outputs.add(id(output_op))
    
    total_kernel_ops = num_operators + len(external_inputs) + len(external_outputs)
    
    return total_kernel_ops

def augment_partitions(valid_partitions,
                      all_operators,
                      augmentation_factor=5,
                      perturbation_strategies=None,
                      prune_duplicates=True,
                      min_size=2,
                      max_size=9,
                      UNSUPPORTED_OPS=set(),
                      IGNORE_OPS=set()):
    """
    Augment dataset by creating variations of valid partitions.
    
    Args:
        valid_partitions: List of valid partitions from get_partitions (adjacency list format)
        all_operators: Dictionary of operators from get_computation_graph
        augmentation_factor: Number of variations to create per valid partition
        perturbation_strategies: List of strategies ['expand', 'contract']
        prune_duplicates: Whether to remove duplicate subgraphs
        min_size: Minimum partition size
        max_size: Maximum partition size
        UNSUPPORTED_OPS: Set of unsupported operations
        IGNORE_OPS: Set of operations to ignore
        
    Returns:
        List of augmented subgraphs in adjacency list format
    """
    
    if perturbation_strategies is None:
        perturbation_strategies = ['expand', 'contract']
    
    augmented_subgraphs = []
    
    # Include original valid partitions
    augmented_subgraphs.extend(valid_partitions)
    
    # STEP 1: Create structural variations (expand/contract only)
    for partition in valid_partitions:
        for _ in range(augmentation_factor):
            strategy = random.choice(perturbation_strategies)
            augmented = None
            current_size = get_total_partition_ops(partition)

            if strategy == 'expand':
                max_expansion = max_size - current_size
                if max_expansion > 0:
                    expansion_size = random.randint(1, min(3, max_expansion))
                    augmented = _expand_partition_adjacency(partition, all_operators, 
                                                           UNSUPPORTED_OPS, IGNORE_OPS, 
                                                           expansion_size)
                else:
                    continue # too large to expand

            elif strategy == 'contract':
                if current_size > min_size:
                    max_contraction = current_size - min_size
                    contraction_size = random.randint(1, min(2, max_contraction))
                    augmented = _contract_partition_adjacency(partition, contraction_size)
                else:
                    continue  # Skip if too small to contract
            
            else:
                continue
            
            if augmented and min_size <= get_total_partition_ops(augmented) <= max_size:
                augmented_subgraphs.append(augmented)
    
    print(f"Generated {len(augmented_subgraphs)} structural variations (before deduplication)")
    
    if prune_duplicates:
        original_count = len(augmented_subgraphs)
        augmented_subgraphs, num_duplicates = prune_duplicate_subgraphs(augmented_subgraphs)
        print(f"Removed {num_duplicates} duplicate subgraphs ({num_duplicates/original_count*100:.1f}%)")
    
    structures_before_param_perturb = len(augmented_subgraphs)
    num_to_perturb = int(len(augmented_subgraphs))
    
    # param_variations_per_structure = 4

    # if num_to_perturb > 0:
    #     structures_to_perturb = random.sample(augmented_subgraphs, num_to_perturb)
        
    #     param_perturbed_subgraphs = []
    #     for structure in structures_to_perturb:

    #         # Create multiple parameter variations of this structure
    #         for _ in range(param_variations_per_structure):
    #             num_ops_to_perturb = random.randint(1, min(3, len(structure)))
    #             perturbed = _perturb_operator_params(structure, num_ops_to_perturb)
    #             param_perturbed_subgraphs.append(perturbed)
        
    #     # Add the parameter-perturbed versions
    #     augmented_subgraphs.extend(param_perturbed_subgraphs)
        
    #     print(f"Created {len(param_perturbed_subgraphs)} parameter variations from {num_to_perturb} structures")
    
    # # Print final statistics
    # sizes = [len(sg) for sg in augmented_subgraphs]
    # if sizes:
    #     print(f"Final dataset size: {len(augmented_subgraphs)} subgraphs")
    #     print(f"Size distribution: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")

    return augmented_subgraphs

def serialize_subgraphs_to_json(subgraphs: List[Dict[Any, List[Any]]], filename: str = None) -> str:
    """
    Serialize adjacency list subgraphs to JSON format.
    """
    
    serialized_data = []
    
    for i, subgraph in enumerate(subgraphs):
        subgraph_data = {
            'subgraph_id': i,
            'nodes': {},
            'edges': []
        }
        
        # Create mapping from operator objects to IDs for JSON serialization
        node_to_id = {}
        id_counter = 0
        
        # First pass: serialize all nodes
        for op_node in subgraph.keys():
            node_id = f"node_{id_counter}"
            node_to_id[op_node] = node_id
            id_counter += 1
            
            subgraph_data['nodes'][node_id] = {
                'name': getattr(op_node, 'name', str(op_node)),
                'fn': getattr(op_node, 'fn', str(op_node)),
                'input_tensor_shapes': getattr(op_node, 'input_tensor_shapes', []),
                'output_tensor_shapes': getattr(op_node, 'output_tensor_shapes', []),
                'additional_params': getattr(op_node, 'additional_params', [])
            }
        
        # Second pass: serialize edges
        for from_node, to_nodes in subgraph.items():
            from_id = node_to_id[from_node]
            for to_node in to_nodes:
                if to_node in node_to_id:  # Only include edges within the subgraph
                    to_id = node_to_id[to_node]
                    subgraph_data['edges'].append([from_id, to_id])
        
        serialized_data.append(subgraph_data)
    
    # Convert to JSON
    json_str = json.dumps(serialized_data, indent=2)
    
    if filename:
        with open(filename, 'w') as f:
            f.write(json_str)
        print(f"Saved {len(subgraphs)} subgraphs to {filename}")
    
    return json_str

def _perturb_operator_params(partition: Dict[Any, List[Any]], 
                            num_to_perturb: int = 1) -> Dict[Any, List[Any]]:
    """
    Create a variation by modifying operator parameters.
    
    Args:
        partition: Partition in adjacency list format
        num_to_perturb: Number of operators to modify
        
    Returns:
        Modified partition with perturbed parameters
    """
    import copy
    
    # Deep copy to avoid modifying original
    perturbed = {}
    for node, connections in partition.items():
        # Create a shallow copy of the node to modify its params
        new_node = copy.copy(node)
        perturbed[new_node] = list(connections)
    
    # Get perturbable operators
    perturbable_ops = []
    for node in perturbed.keys():
        op_type = getattr(node, 'fn', '').lower()
        
        # Check if this operator type has perturbable parameters
        if any(op in op_type for op in ['split', 'chunk', 'conv', 'pool', 'pad', 
                                         'slice', 'reshape', 'transpose', 'permute']):
            perturbable_ops.append(node)
    
    if not perturbable_ops:
        return perturbed
    
    # Randomly select operators to perturb
    num_to_perturb = min(num_to_perturb, len(perturbable_ops))
    nodes_to_perturb = random.sample(perturbable_ops, num_to_perturb)
    
    for node in nodes_to_perturb:
        _apply_parameter_perturbation(node)
    
    return perturbed

def _apply_parameter_perturbation(node):
    """
    Apply parameter perturbation to a single operator node.
    Modifies the node's kwargs in place.
    
    Only perturbs operations that are supported by Mirage and have
    parameters that affect kernel generation.
    """
    op_type = getattr(node, 'fn', '')
    
    if not hasattr(node, 'kwargs'):
        node.kwargs = {}
    elif node.kwargs is None:
        node.kwargs = {}
    
    # Helper function to safely get input shape
    def get_input_shape(node):
        """Safely extract input shape from node."""
        try:
            shapes = getattr(node, 'input_tensor_shapes', [])
            if not shapes:
                return []
            
            # Get first shape - handle tuple format (shape, tensor_id)
            first_shape = shapes[0]
            if isinstance(first_shape, tuple):
                first_shape = first_shape[0]  # Extract shape from (shape, id) tuple
            
            # Convert to list of integers
            if isinstance(first_shape, (list, tuple)):
                return [int(s) for s in first_shape if isinstance(s, (int, float))]
            return []
        except:
            return []
    
    # ReduceSum - perturb reduction dimensions
    if op_type == 'ReduceSum':
        input_shape = get_input_shape(node)
        if len(input_shape) > 0:
            # Randomly select dimension(s) to reduce
            num_dims = len(input_shape)
            # Choose between reducing 1 dim or multiple dims
            if random.random() < 0.5 and num_dims > 1:
                # Reduce single dimension
                node.kwargs['dim'] = random.randint(0, num_dims - 1)
            else:
                # Reduce multiple dimensions
                num_to_reduce = random.randint(1, min(2, num_dims))
                dims_to_reduce = random.sample(range(num_dims), num_to_reduce)
                node.kwargs['dim'] = dims_to_reduce if len(dims_to_reduce) > 1 else dims_to_reduce[0]
            
            # Randomly set keepdim
            node.kwargs['keepdim'] = random.choice([True, False])
    
    # Clip/Clamp - perturb min/max values
    elif op_type == 'Clip':
        # Generate reasonable clamp ranges
        min_val = random.choice([-10.0, -5.0, -1.0, 0.0])
        max_val = random.choice([1.0, 5.0, 10.0, 100.0])
        
        # Ensure min < max
        if min_val >= max_val:
            min_val, max_val = max_val - 1.0, min_val + 1.0
        
        node.kwargs['min'] = min_val
        node.kwargs['max'] = max_val
    
    # Pow - perturb exponent value
    elif op_type == 'Pow':
        # Randomly select exponent
        node.kwargs['exponent'] = random.choice([0.5, 1.0, 2.0, 3.0, -1.0, -0.5])
    
    # RMSNormalization - perturb epsilon
    elif op_type == 'RMSNormalization':
        node.kwargs['epsilon'] = random.choice([1e-5, 1e-6, 1e-7, 1e-8])