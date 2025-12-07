import torch
from itertools import combinations as comb
import os
import json
from partitioning.build_computation_graph import get_computation_graph
from partitioning.utils import to_kernel_graph
from partitioning.build_dataset import augment_partitions
import concurrent.futures

def copy_subgraph(subgraph):
    new_subgraph = {}
    for from_op, to_ops in subgraph.items():
        new_subgraph[from_op] = to_ops.copy()
    return new_subgraph

def contains_4D_tensors(op_node):
    for shape, _ in op_node.input_tensor_shapes:
        if len(shape) > 3:
            return True
    for shape, _ in op_node.output_tensor_shapes:
        if len(shape) > 3:
            return True
    return False

def get_partitions(op_node, min_num_ops, max_num_ops, all_subgraphs, UNSUPPORTED_OPS, COMPOSITE_OPS, IGNORE_OPS):
    if op_node.fn not in UNSUPPORTED_OPS.union(IGNORE_OPS):
        # handle non-matching shapes
        op_needs_broadcast = False
        input_dims = len(op_node.input_tensor_shapes[0][0])
        for s in op_node.input_tensor_shapes:
            if len(s[0]) != input_dims:
                op_needs_broadcast = True
                break
        if not op_needs_broadcast:
            if op_node.fn in COMPOSITE_OPS:
                num_ops = COMPOSITE_OPS[op_node.fn]
            else:
                num_ops = 1
            get_partitions_helper(op_node, {op_node: []}, num_ops, min_num_ops, max_num_ops, set(), all_subgraphs, UNSUPPORTED_OPS, COMPOSITE_OPS, IGNORE_OPS)

def get_partitions_helper(op_node, curr_subgraph, num_ops, min_num_ops, max_num_ops, visited, all_subgraphs, UNSUPPORTED_OPS, COMPOSITE_OPS, IGNORE_OPS):
    # if it is a composite operator, return a subgraph with only that one operator
    if id(op_node.name) in visited:
        return
    if num_ops > max_num_ops:
        return
    if contains_4D_tensors(op_node):
        return
    
    # assume op_node already in curr_subgraph
    visited.add(id(op_node.name))
    if num_ops >= min_num_ops:
        all_subgraphs.append(copy_subgraph(curr_subgraph))

    def find_valid_output_ops(output_op, orig_out_id, prev_out_id, valid_output_ops, visited):
        if contains_4D_tensors(output_op):
            return
        if id(output_op.name) in visited:
            return
        visited.add(id(output_op.name))
        
        # .union(COMPOSITE_OPS) ensures that no second operator of a subgraph is composite op
        if output_op.fn in UNSUPPORTED_OPS.union(IGNORE_OPS):
            return

        input_dims = len(output_op.input_tensor_shapes[0][0])
        if input_dims == 0:
            return
        for s in output_op.input_tensor_shapes:
            if len(s[0]) != input_dims:
                return
        
        # WARNING: this messes up the node structure and causes input/output_ops and tensor_ids to not correspond to each other anymore
        # could work because partition_graph does not use tensor_ids and to_kernel_graph does not use input/output_ops
        if output_op.fn in IGNORE_OPS:
            return
        #     for out_op in output_op.output_ops:
        #         if out_op.output_tensor_shapes:
        #             assert len(out_op.output_tensor_shapes) == 1
        #             find_valid_output_ops(out_op, orig_out_id, out_op.output_tensor_shapes[0][1], valid_output_ops, visited)
        # elif output_op.fn not in UNSUPPORTED_OPS:
        #     # find in the inputs of output_op the tensor whose id is prev_out_id, replace with orig_out_id
        #     for i in range(len(output_op.input_tensor_shapes)):
        #         if output_op.input_tensor_shapes[i][1] == prev_out_id:
        #             output_op.input_tensor_shapes[i] = (output_op.input_tensor_shapes[i][0], orig_out_id)
        #             break
        #     valid_output_ops.append(output_op)
        valid_output_ops.append(output_op)
        
    valid_output_ops = []
    for output_op in op_node.output_ops:
        find_valid_output_ops(output_op, op_node.output_tensor_shapes[0][1], op_node.output_tensor_shapes[0][1], valid_output_ops, set())
    for choose_k in range(1, len(valid_output_ops) + 1):
        curr_subgraph_copy = copy_subgraph(curr_subgraph)
        for comb_outputs in comb(valid_output_ops, choose_k):
            visited_copy = visited.copy()
            for output_node in comb_outputs:
                if output_node not in curr_subgraph_copy:
                    curr_subgraph_copy[output_node] = []
                curr_subgraph_copy[op_node].append(output_node)
                if output_node.fn in COMPOSITE_OPS:
                    num_ops += COMPOSITE_OPS[output_node.fn]
                else:
                    num_ops += 1
                get_partitions_helper(output_node, copy_subgraph(curr_subgraph_copy), num_ops, min_num_ops, max_num_ops, visited_copy, all_subgraphs, UNSUPPORTED_OPS, COMPOSITE_OPS, IGNORE_OPS)

def partition_graph(model, 
                    dummy_input, 
                    min_num_ops=2, 
                    max_num_ops=4, 
                    UNSUPPORTED_OPS=set(), # these are operators not supported by Mirage
                    COMPOSITE_OPS=dict(),
                    IGNORE_OPS=set()): # these are operators that performs no operations on the tensors
    unique_operators = {}
    operators, _ = get_computation_graph(model, dummy_input, unique_operators, "onnx")

    all_subgraphs = []
    for _, op_node in operators.items():
        get_partitions(op_node, min_num_ops, max_num_ops, all_subgraphs, UNSUPPORTED_OPS, COMPOSITE_OPS, IGNORE_OPS)
    return all_subgraphs, unique_operators

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
    operators, _ = get_computation_graph(model, dummy_input, unique_operators, "onnx")
    print("got comp graph")
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
    # for i, subgraph in enumerate(augmented_subgraphs):
    #     print(f"Subgraph {i}: {len(subgraph)} operators")
    print(f"Generated {len(augmented_subgraphs)} total subgraphs (including originals)")

    return augmented_subgraphs, unique_operators

def generate_all_augmented_kernels(input_configs, model, root_dir, dataset_name, min_num_ops=2, max_num_ops=4, aug_factor=5, UNSUPPORTED_OPS=set(), COMPOSITE_OPS=set(), IGNORE_OPS=set(), timeout_minutes=5):
    # Collect subgraphs from all input configurations
    all_subgraphs = []
    for batch_size, seq_len in input_configs:
        print(f"\nProcessing config: batch={batch_size}, seq={seq_len}")
        dummy_input = torch.ones(batch_size, seq_len, dtype=int)
        
        subgraphs, _ = partition_graph_with_sampling(model, dummy_input, min_num_ops, max_num_ops, aug_factor, UNSUPPORTED_OPS, COMPOSITE_OPS, IGNORE_OPS)
        all_subgraphs.extend(subgraphs)
        print(f"  Collected {len(subgraphs)} subgraphs from this config")
    
    print(f"\nTotal subgraphs collected: {len(all_subgraphs)}")
    
    kernel_input_dims = []
    all_kernels = []
    
    done = [int(f.split("_")[1].split(".")[0]) for f in os.listdir(root_dir) if f.startswith("original_")]
    hashes = set(done)
    
    performance = json.load(open(os.path.join(root_dir, f"{dataset_name}_performance.json"), "r")) if os.path.exists(os.path.join(root_dir, f"{dataset_name}_performance.json")) else {}
    print("Found ", performance)
    timeout_seconds = timeout_minutes * 60
    
    for i, subgraph in enumerate(all_subgraphs):
        print(f"\n[{i+1}/{len(all_subgraphs)}] Processing subgraph...")
        
        produced_in_subgraph = set()
        consumed_in_subgraph = set()
        for op in subgraph:
            for _, tid in op.output_tensor_shapes:
                produced_in_subgraph.add(tid)
            for _, tid in op.input_tensor_shapes:
                consumed_in_subgraph.add(tid)
        
        output_ids = [tid for tid in produced_in_subgraph if tid not in consumed_in_subgraph]
        
        try:
            kernel_graph, dims = to_kernel_graph(subgraph, output_ids)
        except NotImplementedError:
            print(f"  ⚠ Skipped: NotImplementedError in to_kernel_graph")
            continue
        
        # check for duplicate subgraphs
        graph_hash = kernel_graph.get_owner_independent_hash()
        if graph_hash in hashes:
            print(f"  ⚠ Skipped: Duplicate hash {graph_hash}")
            continue
        hashes.add(graph_hash)
        
        # save original mugraph
        kernel_graph.to_json(os.path.join(root_dir, f"original_{graph_hash}.json"))
        
        # Superoptimize with timeout
        try:
            print(f"  Superoptimizing {graph_hash} (timeout: {timeout_minutes} min)...")
            
            # Use ThreadPoolExecutor with timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(kernel_graph.superoptimize)
                try:
                    result = future.result(timeout=timeout_seconds)
                    
                    # Handle both tuple and single value returns
                    if isinstance(result, tuple):
                        optimized_graph, best_perf = result
                    else:
                        optimized_graph = result
                        best_perf = -1  # Unknown performance
                    
                    print(f"  ✓ Optimization complete (perf: {best_perf})")
                    
                except concurrent.futures.TimeoutError:
                    print(f"  ✗ TIMEOUT after {timeout_minutes} minutes - skipping this graph")
                    # Remove the hash so we can retry later if needed
                    hashes.remove(graph_hash)
                    # Delete the original file since we didn't optimize it
                    os.remove(os.path.join(root_dir, f"original_{graph_hash}.json"))
                    continue
                
        except Exception as e:
            print(f"  ✗ Superoptimize failed with error: {e}")
            continue
        
        performance[graph_hash] = best_perf
        
        # save optimized mugraph
        optimized_graph.to_json(os.path.join(root_dir, f"optimized_{graph_hash}.json"))
        all_kernels.append(optimized_graph)
        kernel_input_dims.append(dims)
        
        # save performance after each successful optimization
        json.dump(performance, open(os.path.join(root_dir, f"{dataset_name}_performance.json"), "w"))
        print(f"  ✓ Saved optimized kernel {graph_hash}")
    
    print(f"\n{'='*60}")
    print(f"FINAL STATS:")
    print(f"  Total subgraphs processed: {len(all_subgraphs)}")
    print(f"  Successfully optimized: {len(all_kernels)}")
    print(f"  Success rate: {len(all_kernels)/len(all_subgraphs)*100:.1f}%")
    print(f"{'='*60}")
    
    return all_kernels, kernel_input_dims
