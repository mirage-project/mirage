import random
from collections import deque, defaultdict
from typing import Dict, List, Set, Optional, Any, Tuple
import json
from build_computation_graph import get_computation_graph
from partition_graph import get_partitions, to_kernel_graph, time_kernels
from visualize_partitions import visualize_partition, compare_augmentations

def _shallow_copy_adjacency(partition: Dict[Any, List[Any]]) -> Dict[Any, List[Any]]:
    """Create a shallow copy of adjacency list."""
    return {node: list(connections) for node, connections in partition.items()}

def _expand_partition_adjacency(partition: Dict[Any, List[Any]], 
                               all_operators: Dict[str, Any],
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
                input_op not in partition_nodes):
                boundary_candidates.append(input_op)
        
        # Check output neighbors  
        for output_op in op_node.output_ops:
            if (hasattr(output_op, 'name') and 
                output_op.name in all_operators and 
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


def augment_partitions(valid_partitions,
                      all_operators,
                      augmentation_factor=5,
                      perturbation_strategies=None,
                      prune_duplicates=True,
                      max_size = 4):
    """
    Augment dataset by creating variations of valid partitions.
    
    Args:
        valid_partitions: List of valid partitions from get_partitions (adjacency list format)
        all_operators: Dictionary of operators from get_computation_graph
        augmentation_factor: Number of variations to create per valid partition
        perturbation_strategies: List of strategies ['expand', 'contract', 'perturb']
        prune_duplicates: Whether to remove duplicate subgraphs
        
    Returns:
        List of augmented subgraphs in adjacency list format
    """
    
    if perturbation_strategies is None:
        perturbation_strategies = ['expand', 'contract', 'perturb']
    
    augmented_subgraphs = []
    
    # Include original valid partitions
    augmented_subgraphs.extend(valid_partitions)
    
    for partition in valid_partitions:
        for _ in range(augmentation_factor):
            strategy = random.choice(perturbation_strategies)
            
            if strategy == 'expand':
                current_size = len(partition)
                max_expansion = max_size - current_size
                if max_expansion > 0:
                    expansion_size = random.randint(1, min(3, max_expansion))
                    augmented = _expand_partition_adjacency(partition, all_operators, expansion_size)
                else:
                    continue # too large to expand

            elif strategy == 'contract':
                if len(partition) > 3:  # Only contract if large enough
                    contraction_size = random.randint(1, min(2, len(partition) - 2))
                    augmented = _contract_partition_adjacency(partition, contraction_size)
                else:
                    continue  # Skip if too small to contract
                        
            else:
                continue
            
            if augmented and len(augmented) >= 2:  # Ensure minimum size
                augmented_subgraphs.append(augmented)
    
    # Prune duplicates
    if prune_duplicates:
        original_count = len(augmented_subgraphs)
        augmented_subgraphs, num_duplicates = prune_duplicate_subgraphs(augmented_subgraphs)
        print(f"Removed {num_duplicates} duplicate subgraphs ({num_duplicates/original_count*100:.1f}%)")
    
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
    Modifies the node's additional_params in place.
    """
    op_type = getattr(node, 'fn', '').lower()
    
    if not hasattr(node, 'additional_params') or not node.additional_params:
        node.additional_params = {}
    
    # Split/Chunk operations
    if 'split' in op_type or 'chunk' in op_type:
        # Perturb split sizes or number of chunks
        if 'split_size' in str(node.additional_params):
            # Example: change split sizes
            node.additional_params['split_size'] = random.choice([2, 3, 4, 5, 8])
        elif 'chunks' in str(node.additional_params):
            node.additional_params['chunks'] = random.choice([2, 3, 4, 5])
        # Perturb dimension
        if 'dim' in str(node.additional_params):
            # Keep dimension valid based on tensor rank
            input_shape = getattr(node, 'input_tensor_shapes', [[]])[0]
            if len(input_shape) > 0:
                max_dim = len(input_shape) - 1
                node.additional_params['dim'] = random.randint(0, max_dim)
    
    # Convolution operations
    elif 'conv' in op_type:
        # Perturb kernel size, stride, padding
        if 'kernel_size' in str(node.additional_params):
            node.additional_params['kernel_size'] = random.choice([1, 3, 5, 7])
        if 'stride' in str(node.additional_params):
            node.additional_params['stride'] = random.choice([1, 2])
        if 'padding' in str(node.additional_params):
            node.additional_params['padding'] = random.choice([0, 1, 2, 3])
        if 'dilation' in str(node.additional_params):
            node.additional_params['dilation'] = random.choice([1, 2])
    
    # Pooling operations
    elif 'pool' in op_type:
        if 'kernel_size' in str(node.additional_params):
            node.additional_params['kernel_size'] = random.choice([2, 3, 4, 5])
        if 'stride' in str(node.additional_params):
            node.additional_params['stride'] = random.choice([1, 2, 3])
        if 'padding' in str(node.additional_params):
            node.additional_params['padding'] = random.choice([0, 1, 2])
    
    # Padding operations
    elif 'pad' in op_type:
        # Perturb padding values
        node.additional_params['padding'] = [random.randint(0, 3) for _ in range(4)]
    
    # Transpose/Permute operations
    elif 'transpose' in op_type or 'permute' in op_type:
        input_shape = getattr(node, 'input_tensor_shapes', [[]])[0]
        if len(input_shape) > 0:
            # Generate valid permutation
            dims = list(range(len(input_shape)))
            random.shuffle(dims)
            node.additional_params['dims'] = dims
    
    # Slice operations
    elif 'slice' in op_type:
        input_shape = getattr(node, 'input_tensor_shapes', [[]])[0]
        if len(input_shape) > 0:
            # Perturb slice parameters
            dim = random.randint(0, len(input_shape) - 1)
            max_size = input_shape[dim] if input_shape[dim] > 0 else 10
            start = random.randint(0, max(0, max_size - 2))
            end = random.randint(start + 1, max_size)
            node.additional_params['dim'] = dim
            node.additional_params['start'] = start
            node.additional_params['end'] = end
    
    # Reshape operations
    elif 'reshape' in op_type or 'view' in op_type:
        input_shape = getattr(node, 'input_tensor_shapes', [[]])[0]
        if len(input_shape) > 0 and all(s > 0 for s in input_shape):
            # Calculate total size
            total_size = 1
            for s in input_shape:
                total_size *= s
            
            # Generate valid reshape dimensions
            possible_shapes = _generate_valid_reshape_sizes(total_size)
            if possible_shapes:
                node.additional_params['shape'] = random.choice(possible_shapes)


def _generate_valid_reshape_sizes(total_size: int, max_dims: int = 4) -> List[List[int]]:
    """Generate valid reshape dimensions for a given total size."""
    shapes = []
    
    # Find divisors
    divisors = []
    for i in range(1, min(total_size + 1, 100)):
        if total_size % i == 0:
            divisors.append(i)
    
    # Generate 2D shapes
    for d1 in divisors:
        d2 = total_size // d1
        shapes.append([d1, d2])
    
    # Generate 3D shapes (sample a few)
    if len(divisors) > 2:
        for _ in range(min(5, len(divisors))):
            d1 = random.choice(divisors)
            remaining = total_size // d1
            sub_divisors = [d for d in divisors if remaining % d == 0]
            if sub_divisors:
                d2 = random.choice(sub_divisors)
                d3 = remaining // d2
                shapes.append([d1, d2, d3])
    
    return shapes[:10]  # Return up to 10 variations
def partition_graph_with_sampling(model, 
                                 dummy_input, 
                                 min_num_ops=2, 
                                 max_num_ops=4, 
                                 augmentation_factor=5,
                                 UNSUPPORTED_OPS=set(), 
                                 COMPOSITE_OPS=dict(), 
                                 IGNORE_OPS=set()):
    """
    Generate augmented dataset using only valid partitions as seeds.
    
    Args:
        model: The model to analyze
        dummy_input: Dummy input for the model
        min_num_ops: Minimum number of operations in a partition
        max_num_ops: Maximum number of operations in a partition
        augmentation_factor: Number of variations to create per valid partition
        UNSUPPORTED_OPS: Set of unsupported operations
        COMPOSITE_OPS: Dictionary of composite operations
        IGNORE_OPS: Set of operations to ignore
        
    Returns:
        Tuple of (augmented_subgraphs, unique_operators)
    """
    
    # Get the computation graph
    unique_operators = {}
    operators = get_computation_graph(model, dummy_input, unique_operators, "onnx")

    # Get valid partitions (already in adjacency list format)
    valid_partitions = []
    for _, op_node in operators.items():
        get_partitions(op_node, min_num_ops, max_num_ops, valid_partitions, 
                      UNSUPPORTED_OPS, COMPOSITE_OPS, IGNORE_OPS)
    
    print(f"Found {len(valid_partitions)} valid partitions")
    # plt = visualize_partition(valid_partitions[0], "Example Valid Partition", "printed_partition0")


    # Augment the valid partitions
    augmented_subgraphs = augment_partitions(
        valid_partitions=valid_partitions,
        all_operators=operators,
        augmentation_factor=augmentation_factor,
        perturbation_strategies=['expand', 'contract'],
        max_size = max_num_ops
    )
    for i, subgraph in enumerate(augmented_subgraphs):
        print(f"Subgraph {i}: {len(subgraph)} operators")
    print(f"Generated {len(augmented_subgraphs)} total subgraphs (including originals)")
    print(f"Augmentation ratio: {len(augmented_subgraphs) / len(valid_partitions):.1f}x")
    compare_augmentations(valid_partitions[0], augmented_subgraphs[:3])

    print("Generating kernels from augmented subgraphs")
    kernel_input_dims = []
    all_kernels = []
    hashes = set()
    performance = {}
    dataset_name = "augmented_dataset_timed"
    for i, subgraph in enumerate(augmented_subgraphs[:5]):
        kernel_graph, dims = to_kernel_graph(subgraph)
        
        # check for duplicate subgraphs
        graph_hash = kernel_graph.get_owner_independent_hash()
        if graph_hash in hashes:
            continue
        hashes.add(graph_hash)
        
        # save original mugraph
        kernel_graph.to_json(f"original_{graph_hash}.json")
        
        try:
            print(f"Superoptimizing {graph_hash}")
            optimized_graph, best_perf = kernel_graph.superoptimize()
        except Exception as e:
            print(f"Subgraph {graph_hash} superoptimize failed with error: {e}")
            continue

        performance[graph_hash] = best_perf
        optimized_graph.to_json(f"optimized_{graph_hash}.json")
        all_kernels.append(optimized_graph)
        kernel_input_dims.append(dims)
        print(f"Done {i}/{len(augmented_subgraphs)}")
    if dataset_name is not None:
        json.dump(performance, open(f"{dataset_name}_performance.json", "w"))
    else:
        json.dump(performance, open("performance.json", "w"))
    return all_kernels, kernel_input_dims

