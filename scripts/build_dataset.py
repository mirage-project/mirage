import random
from collections import deque, defaultdict
from typing import Dict, List, Set, Optional, Any, Tuple
import json
from build_computation_graph import get_computation_graph

def _shallow_copy_adjacency(partition: Dict[Any, List[Any]]) -> Dict[Any, List[Any]]:
    """Create a shallow copy of adjacency list."""
    return {node: list(connections) for node, connections in partition.items()}

def _expand_partition_adjacency(partition: Dict[Any, List[Any]], 
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
            if (hasattr(input_op, 'fn') and 
                input_op.fn not in UNSUPPORTED_OPS and 
                input_op.fn not in IGNORE_OPS and
                input_op not in partition_nodes):
                boundary_candidates.append(input_op)
        
        # Check output neighbors  
        for output_op in op_node.output_ops:
            if (hasattr(output_op, 'fn') and 
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


def _perturb_partition_adjacency(partition: Dict[Any, List[Any]],
                                UNSUPPORTED_OPS: set,
                                IGNORE_OPS: set,
                                perturbation_ratio: float = 0.3) -> Dict[Any, List[Any]]:
    """Create a variation by removing some nodes and adding new connected ones."""
    
    partition_size = len(partition)
    num_to_change = max(1, int(partition_size * perturbation_ratio))
    
    # First contract, then expand
    contracted = _contract_partition_adjacency(partition, num_to_change)
    perturbed = _expand_partition_adjacency(contracted, UNSUPPORTED_OPS, IGNORE_OPS, num_to_change)
    
    return perturbed


def augment_partitions(valid_partitions: List[Dict[Any, List[Any]]],
                      UNSUPPORTED_OPS: set,
                      IGNORE_OPS: set,
                      augmentation_factor: int = 5,
                      perturbation_strategies: List[str] = None) -> List[Dict[Any, List[Any]]]:
    """
    Augment dataset by creating variations of valid partitions.
    
    Args:
        valid_partitions: List of valid partitions from get_partitions (adjacency list format)
        all_operators: Dictionary of operators from get_computation_graph
        augmentation_factor: Number of variations to create per valid partition
        perturbation_strategies: List of strategies ['expand', 'contract', 'perturb']
        
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
                expansion_size = random.randint(1, 3)
                augmented = _expand_partition_adjacency(partition, UNSUPPORTED_OPS, IGNORE_OPS, expansion_size)
            
            elif strategy == 'contract':
                if len(partition) > 3:  # Only contract if large enough
                    contraction_size = random.randint(1, min(2, len(partition) - 2))
                    augmented = _contract_partition_adjacency(partition, contraction_size)
                else:
                    continue  # Skip if too small to contract
            
            elif strategy == 'perturb':
                perturbation_ratio = random.uniform(0.2, 0.5)
                augmented = _perturb_partition_adjacency(partition, UNSUPPORTED_OPS, IGNORE_OPS, perturbation_ratio)
            
            else:
                continue
            
            if augmented and len(augmented) >= 2:  # Ensure minimum size
                augmented_subgraphs.append(augmented)
    
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
