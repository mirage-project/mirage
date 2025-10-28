import torch
from itertools import combinations as comb
import time
import mirage as mi
import json
import numpy as np
import warnings
from op import Operator
from build_computation_graph import get_computation_graph
from build_dataset import augment_partitions, serialize_subgraphs_to_json
import os
from generate_dag import solve_partitions, cost_function
from graph_splitter import process_operator_graph

from cost_model.in_ctx_partition import InCtxPartitioner
# from visualize_augs import visualize_partition, compare_augmentations

CAST_ID_TO_DTYPE = {
    1: torch.float32,
    2: torch.uint8,
    3: torch.int8,
    4: torch.uint16,
    5: torch.int16,
    6: torch.int32,
    7: torch.int64,
    9: torch.bool,
    10: torch.float16,
    11: torch.float64,
    12: torch.uint32,
    13: torch.uint64,
    14: torch.complex64,
    15: torch.complex128,
}

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
    operators = get_computation_graph(model, dummy_input, unique_operators, "onnx")

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
        UNSUPPORTED_OPS=UNSUPPORTED_OPS,
        IGNORE_OPS=IGNORE_OPS,
        augmentation_factor=augmentation_factor,
        perturbation_strategies=['expand', 'contract']
    )

    # compare_augmentations(valid_partitions[0], augmented_subgraphs[:3])
    
    print(f"Generated {len(augmented_subgraphs)} total subgraphs (including originals)")
    print(f"Augmentation ratio: {len(augmented_subgraphs) / len(valid_partitions):.1f}x")
    
    # Serialize the results
    serialize_subgraphs_to_json(augmented_subgraphs, 
                               filename='scripts/augmented_partitions.json')
    
    return augmented_subgraphs, unique_operators

def function_map(graph, func, inputs, kwargs={}):
    match func.fn:
        case "MatMul": return graph.matmul(*inputs, **kwargs)
        case "ReduceSum": return graph.reduction(*inputs, **kwargs)
        case "Exp": return graph.exp(*inputs, **kwargs)
        case "Gelu": return graph.gelu(*inputs, **kwargs)
        case "Relu": return graph.relu(*inputs, **kwargs)
        case "Clip": return graph.clamp(*inputs, **kwargs)
        case "Add": return graph.add(*inputs, **kwargs)
        case "Mul": return graph.mul(*inputs, **kwargs)
        case "Div": return graph.div(*inputs, **kwargs)
        case "Reciprocal": return graph.div(*inputs, **kwargs)
        case "Sqrt": return graph.sqrt(*inputs, **kwargs)
        case "Pow": return graph.pow(*inputs, **kwargs)
        case "Square": return graph.square(inputs[0], **kwargs)
        case "RMSNormalization": return graph.rms_norm(*inputs, **kwargs) # Onnx doesn't support different normalized shape
        case _: 
            raise NotImplementedError(f"{func.fn} not implemented")

# Take in an adjacency list formatted subgraph and generate a mirage kernel graph
def to_kernel_graph(subgraph):
    graph = mi.new_kernel_graph()
    dims = []
    # stores output tensors of operations + their reference counts based on ID
    intermediates = {}
    for op, _ in subgraph.items():
        inputs = []
        for (shape, tensor_id) in op.input_tensor_shapes:
            if tensor_id not in intermediates:
                dims.append((shape, "V"))
                inputs.append(graph.new_input(dims=shape, dtype=mi.float16))
            else:
                inputs.append(intermediates[tensor_id][0])
                intermediates[tensor_id][1] += 1
        for arg, value in op.additional_params.items():
            if arg == "arg0":
                shape = shape = op.output_tensor_shapes[0][0]
                dims = [(shape, "C", value)] + dims
                inputs = [graph.new_input(dims=shape, dtype=mi.float16)] + inputs
            elif arg == "arg1":
                shape = shape = op.output_tensor_shapes[0][0]
                dims.append((shape, "C", value))
                inputs.append(graph.new_input(dims=shape, dtype=mi.float16))
            else:
                assert False, f"Unknown additional param {arg} for op {op.name} with fn {op.fn}"
        
        kwargs = op.kwargs
        res = function_map(graph, op, inputs, kwargs)
        if type(res) == list:
            for i, tensor in enumerate(res):
                intermediates[op.output_tensor_shapes[i][1]] = [tensor, 0]
        else:
            intermediates[op.output_tensor_shapes[0][1]] = [res, 0]
    for _, tsr_cnt in intermediates.items():
        if tsr_cnt[1] == 0: graph.mark_output(tsr_cnt[0])
    return graph, dims
        
def generate_all_kernels(model, dummy_inputs, root_dir, dataset_name, min_num_ops=2, max_num_ops=4, aug_factor=5, UNSUPPORTED_OPS=set(), COMPOSITE_OPS=set(), IGNORE_OPS=set()):
    subgraphs, _ = partition_graph_with_sampling(model, dummy_inputs, min_num_ops, max_num_ops, aug_factor, UNSUPPORTED_OPS, COMPOSITE_OPS, IGNORE_OPS)
    kernel_input_dims = []
    all_kernels = []
    
    done = [int(f.split("_")[1].split(".")[0]) for f in os.listdir(root_dir) if f.startswith("original_")]
    hashes = set(done)

    performance = json.load(open(os.path.join(root_dir, f"{dataset_name}_performance.json"), "r")) if os.path.exists(os.path.join(root_dir, f"{dataset_name}_performance.json")) else {}
    for subgraph in subgraphs:
        kernel_graph, dims = to_kernel_graph(subgraph)
    
        # check for duplicate subgraphs
        graph_hash = kernel_graph.get_owner_independent_hash()
        if graph_hash in hashes:
            continue
        hashes.add(graph_hash)

        # save original mugraph
        kernel_graph.to_json(os.path.join(root_dir, f"original_{graph_hash}.json"))
        
        try:
            print(f"Superoptimizing {graph_hash}")
            optimized_graph, best_perf = kernel_graph.superoptimize()
        except Exception as e:
            print(f"Subgraph {graph_hash} superoptimize failed with error: {e}")
            continue
                
        performance[graph_hash] = best_perf

        # save optimized mugraph
        optimized_graph.to_json(os.path.join(root_dir, f"optimized_{graph_hash}.json"))
        all_kernels.append(optimized_graph)
        kernel_input_dims.append(dims)
    
        # save performance
        json.dump(performance, open(os.path.join(root_dir, f"{dataset_name}_performance.json"), "w"))
    
    return all_kernels, kernel_input_dims

class HybridModel:
    """
    Executable wrapper for Mirage + PyTorch hybrid execution.
    
    Automatically routes operations to either Mirage-optimized kernels 
    or PyTorch fallback implementations.
    """
    
    def __init__(self, execution_plan, input_tensor_ids, output_tensor_ids, parameter_tensors):
        self.execution_plan = execution_plan
        self.input_tensor_ids = input_tensor_ids
        self.output_tensor_ids = output_tensor_ids
        self.parameter_tensors = parameter_tensors  # dict: tensor_id -> parameter tensor

        self._param_cache = {}
        self._const_cache = {}
    
    def _get_params_on(self, device, dtype):
        """Return dict of parameters placed on (device, dtype), cached."""
        key = (device, dtype)
        params = self._param_cache.get(key)
        if params is None:
            # Move once, cache for subsequent calls on same (device, dtype)
            moved = {}
            for tid, p in self.parameter_tensors.items():
                tp = p
                if tp.device != device or tp.dtype != dtype:
                    tp = tp.to(device=device, dtype=dtype, non_blocking=True)
                moved[tid] = tp
            self._param_cache[key] = moved
            params = moved
        return params

    def _get_const(self, device, shape, value, dtype=torch.float16):
        """Return a constant tensor for Mirage kernels, cached per (device, shape, value, dtype)."""
        key = (device, tuple(shape), float(value), dtype)
        t = self._const_cache.get(key)
        if t is None:
            t = torch.full(shape, value, device=device, dtype=dtype)
            self._const_cache[key] = t
        return t
    
    def __call__(self, *inputs):
        if len(inputs) != len(self.input_tensor_ids):
            raise ValueError(f"Expected {len(self.input_tensor_ids)} input(s), got {len(inputs)}")
        
        device = inputs[0].device if inputs else torch.device('cpu')
        dtype = inputs[0].dtype if inputs else torch.float16
        intermediates = {tid: tensor for tid, tensor in zip(self.input_tensor_ids, inputs)}

        params_on_target = self._get_params_on(device, dtype)
        intermediates.update(params_on_target)

        with torch.inference_mode():
            for step_type, payload in self.execution_plan:
                if step_type == "mirage":
                    kernel, input_ids, output_ids, const_dims = payload
                    
                    # Prepare inputs in fp16 ONLY if needed; avoid redundant casts
                    kernel_inputs = []
                    for tid in input_ids:
                        t = intermediates[tid]
                        if t.dtype is not torch.float16:
                            warnings.warn(f"Casting input tensor id {tid} from {t.dtype} to torch.float16. Consider providing inputs in float16 to avoid this overhead.")
                            t = t.to(torch.float16)
                        kernel_inputs.append(t)

                    # Append cached constants (fp16) without recreating them every call
                    dev = kernel_inputs[0].device if kernel_inputs else device
                    for shape, value in const_dims:
                        kernel_inputs.append(self._get_const(dev, shape, value, dtype=torch.float16))

                    outputs = kernel(inputs=kernel_inputs)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    # Convert outputs back to original dtype
                    for tid, tensor in zip(output_ids, outputs):
                        intermediates[tid] = tensor.to(dtype)
                elif step_type == "pytorch":
                    op, input_ids, output_id = payload
                    inputs = [intermediates[tid] for tid in input_ids]
                    result = self._execute_pytorch_op(op, inputs)
                    intermediates[output_id] = result
        
        outputs = [intermediates[tid] for tid in self.output_tensor_ids]
        return outputs[0] if len(outputs) == 1 else outputs
    
    def _execute_pytorch_op(self, op, inputs):
        fn = op.fn
        if fn in ["Add", "Sub", "Mul", "Div"]:
            # Handle operations with scalar constants in additional_params
            if len(inputs) == 1 and hasattr(op, 'additional_params') and 'arg1' in op.additional_params:
                scalar = op.additional_params['arg1']
                if fn == "Add":
                    return inputs[0] + scalar
                elif fn == "Sub":
                    return inputs[0] - scalar
                elif fn == "Mul":
                    return inputs[0] * scalar
                elif fn == "Div":
                    return inputs[0] / scalar
            elif len(inputs) >= 2:
                return getattr(torch, fn.lower())(inputs[0], inputs[1])
            else:
                raise ValueError(f"{fn} requires 2 inputs or additional_params, got {len(inputs)} inputs")
        elif fn == "MatMul":
            return torch.matmul(inputs[0], inputs[1])
        elif fn == "Gemm":
            # General Matrix Multiply: Y = alpha * A @ B + beta * C
            # For Linear layers: Y = A @ B^T + C (default alpha=1, beta=1)
            matmul_result = torch.matmul(inputs[0], inputs[1].T)
            if len(inputs) > 2:  # Has bias
                return matmul_result + inputs[2]
            return matmul_result
        elif fn == "Relu":
            return torch.relu(inputs[0])
        elif fn == "Sigmoid":
            return torch.sigmoid(inputs[0])
        elif fn == "Exp":
            return torch.exp(inputs[0])
        elif fn == "Neg":
            return torch.neg(inputs[0])
        elif fn == "Reciprocal":
            return torch.reciprocal(inputs[0])
        elif fn == "Transpose":
            return inputs[0].transpose(-2, -1)
        elif fn == "Reshape":
            return inputs[0].reshape(op.output_tensor_shapes[0][0])
        elif fn == "Expand":
            out_shape = torch.broadcast_shapes(inputs[0].shape, tuple(op.output_tensor_shapes[0][0]))
            return torch.broadcast_to(inputs[0], (out_shape))
        elif fn == "Gather":
            axis = op.kwargs.get("axis", 0)
            return torch.gather(inputs[0], axis, inputs[1])
        elif fn == "Cast" or fn == "CastLike":
            to_dtype = CAST_ID_TO_DTYPE.get(op.kwargs.get("to"), None)
            if to_dtype is not None:
                return inputs[0].to(dtype=to_dtype)
            else:
                raise NotImplementedError(f"Cast to dtype id '{op.kwargs.get('to')}' not implemented")
        else:
            raise NotImplementedError(f"PyTorch fallback for '{fn}' not implemented")

def partition_graph_with_dp(model, 
                          dummy_input, 
                          IGNORE_OPS=None, 
                          UNSUPPORTED_OPS=None,
                          max_nodes_per_partition=4,
                          dry_run=False,
                          ):
    """
    
    Returns a HybridModel that combines Mirage-optimized kernels with PyTorch fallback.
    
    Args:
        dry_run: If True, skip superoptimize and use original kernel graphs for fast testing
        
    Returns:
        HybridModel: Executable model with hybrid Mirage+PyTorch execution
    """
    
    print("Building computation graph...")
    unique_operators = {}
    operators = get_computation_graph(model, dummy_input, unique_operators, "onnx")
    
    # Load parameters from ONNX model
    import onnx
    onnx_model = onnx.load("scripts/onnx/inferred_model.onnx")
    parameter_dict = {init.name: torch.from_numpy(onnx.numpy_helper.to_array(init)) 
                     for init in onnx_model.graph.initializer}
    
    print("Splitting graph into supported/unsupported subgraphs...")
    if IGNORE_OPS is None:
        IGNORE_OPS = {"Identity", "Cast", "CastLike", "Constant", "Dropout"}
    if UNSUPPORTED_OPS is None:
        UNSUPPORTED_OPS = set()
        
    subgraphs, deps, sorted_ops = process_operator_graph(operators, IGNORE_OPS, UNSUPPORTED_OPS)
    
    print("Applying dynamic programming partitioning to large Mirage subgraphs...")
    
    fine_grained_partitions = []
    
    for sg_id, (sg_dict, sg_type) in enumerate(subgraphs):
        if sg_type == "mirage" and len(sg_dict) > max_nodes_per_partition:
            print(f"  Partitioning subgraph {sg_id} ({len(sg_dict)} ops)...")
            
            # Extract operators in topological order
            ops_list = [op for op in sorted_ops if op in sg_dict] 
            n = len(ops_list)
            adj = np.zeros((n, n), dtype=int)
            
            # Build adjacency matrix
            for i, op in enumerate(ops_list):
                for output_op in op.output_ops:
                    if output_op in ops_list:
                        j = ops_list.index(output_op)
                        adj[i][j] = 1
            
            # Apply DAG partitioning with connectivity constraint
            def cost_function_with_adj(nodes):
                return cost_function(nodes, adj)
            partition_boundaries = solve_partitions(list(range(n)), cost_function_with_adj, max_nodes_per_partition, adj)
            
            # Convert partitions to subgraphs
            for p_id, boundary in enumerate(partition_boundaries):
                start = 0 if p_id == 0 else partition_boundaries[p_id-1] 
                partition_ops = ops_list[start:boundary]
                
                print(f"Partition {p_id}: {[op.name for op in partition_ops]}")
                
                partition_subgraph = {op: True for op in partition_ops}
                fine_grained_partitions.append((partition_subgraph, "mirage"))
                print(f"    Created partition {p_id}: {len(partition_ops)} ops")
        else:
            fine_grained_partitions.append((sg_dict, sg_type))
    
    # Build execution plan
    print(f"\nOptimizing {sum(1 for _, t in fine_grained_partitions if t == 'mirage')} Mirage partitions...")
    execution_plan = []
    
    for pid, (sg_dict, sg_type) in enumerate(fine_grained_partitions):
        # Print partition info
        ops_in_partition = [op for op in sorted_ops if op in sg_dict]
        print(f"\n=== Partition {pid} ({sg_type}): {len(sg_dict)} ops ===")
        print(f"  Operators: {[op.name for op in ops_in_partition]}")
        
        if sg_type == "mirage":
            kernel_graph, dims = to_kernel_graph(sg_dict)
            try:
                if dry_run:
                    # Dry run mode: compile original kernel without superoptimize
                    # Create dummy inputs for compilation (Mirage uses float16)
                    dummy_inputs = []
                    for dim_info in dims:
                        if dim_info[1] == "V":  # Variable input
                            dummy_inputs.append(torch.randn(dim_info[0], dtype=torch.float16, device="cuda"))
                        elif dim_info[1] == "C":  # Constant input
                            dummy_inputs.append(torch.full(dim_info[0], dim_info[2], dtype=torch.float16, device="cuda"))
                    
                    # Compile the kernel
                    kernel_graph.compile(inputs=dummy_inputs)
                    optimized_kernel = kernel_graph
                    print(f"  → Result: Dry run mode - compiled original kernel (no optimization)")
                else:
                    result = kernel_graph.superoptimize()
                    # Handle both tuple (normal optimization) and single value (cached) returns
                    if isinstance(result, tuple):
                        optimized_kernel, _ = result
                    else:
                        optimized_kernel = result
                    print(f"  ✓ Result: Mirage kernel optimized successfully")
                
                # Extract tensor IDs from subgraph
                produced_in_subgraph = set()
                consumed_in_subgraph = set()
                for op in sg_dict:
                    for _, tid in op.output_tensor_shapes:
                        produced_in_subgraph.add(tid)
                    for _, tid in op.input_tensor_shapes:
                        consumed_in_subgraph.add(tid)
                
                # Input tensors: consumed but not produced in subgraph
                input_ids = []
                for op in sg_dict:
                    for _, tid in op.input_tensor_shapes:
                        if tid not in produced_in_subgraph and tid not in input_ids:
                            input_ids.append(tid)
                
                # Output tensors: produced but not consumed in subgraph (exported to outside)
                output_ids = [tid for tid in produced_in_subgraph if tid not in consumed_in_subgraph]
                const_dims = [(d[0], d[2]) for d in dims if d[1] == "C"]
                
                execution_plan.append(("mirage", (optimized_kernel, input_ids, output_ids, const_dims)))
            except Exception as e:
                print(f"  ✗ Result: Superoptimize failed - {str(e)[:80]}...")
                print(f"     Fallback: Using PyTorch execution")
                for op in sg_dict:
                    execution_plan.append(("pytorch", (op, [t for _, t in op.input_tensor_shapes], op.output_tensor_shapes[0][1])))
        else:
            print(f"  → Result: Using PyTorch execution (unsupported operations)")
            for op in sg_dict:
                execution_plan.append(("pytorch", (op, [t for _, t in op.input_tensor_shapes], op.output_tensor_shapes[0][1])))
    
    # Topological sort execution plan
    def get_plan_io(step):
        step_type, payload = step
        if step_type == "mirage":
            return payload[1], payload[2]  # input_ids, output_ids
        else:  # pytorch
            return payload[1], [payload[2]]  # input_ids, [output_id]
    
    sorted_plan = []
    all_produced = {tid for op in sorted_ops for _, tid in op.output_tensor_shapes}
    available = set(tid for op in sorted_ops for _, tid in op.input_tensor_shapes if tid not in all_produced)
    remaining = execution_plan.copy()
    
    while remaining:
        for step in remaining[:]:
            inputs, outputs = get_plan_io(step)
            if all(tid in available for tid in inputs):
                sorted_plan.append(step)
                available.update(outputs)
                remaining.remove(step)
        if len(sorted_plan) == len(execution_plan):
            break
    
    execution_plan = sorted_plan
    print(f"\n✓ Sorted execution plan: {len(execution_plan)} steps")
    
    # Infer model inputs/outputs
    all_produced = {tid for op in sorted_ops for _, tid in op.output_tensor_shapes}
    all_input_tensor_ids = [tid for op in sorted_ops for _, tid in op.input_tensor_shapes if tid not in all_produced]
    all_input_tensor_ids = list(dict.fromkeys(all_input_tensor_ids))
    
    # Separate real inputs from parameters (ONNX orders: real_inputs + parameters)
    num_real_inputs = 1 if not isinstance(dummy_input, (tuple, list)) else len(dummy_input)
    input_tensor_ids = all_input_tensor_ids[:num_real_inputs]
    param_names = list(parameter_dict.keys())
    parameter_tensors = {tid: parameter_dict[param_names[i]] 
                        for i, tid in enumerate(all_input_tensor_ids[num_real_inputs:])}
    
    all_consumed = {tid for op in sorted_ops for _, tid in op.input_tensor_shapes}
    output_tensor_ids = [tid for op in sorted_ops for _, tid in op.output_tensor_shapes if tid not in all_consumed]
    
    print(f"✓ Loaded {len(parameter_tensors)} parameters, {len(input_tensor_ids)} real inputs, {len(output_tensor_ids)} outputs")
    return HybridModel(execution_plan, input_tensor_ids, output_tensor_ids, parameter_tensors)

def partition_graph_with_in_ctx_partitions(model, 
                                           dummy_input, 
                                           dataset_root,
                                           mirage_root,
                                           model_name="gpt-oss:120b",
                                           scale=1.0,
                                           IGNORE_OPS=None, 
                                           UNSUPPORTED_OPS=None,
                                           max_nodes_per_partition=4,
                                           dry_run=False,
                                           ):
    """
    
    Returns a HybridModel that combines Mirage-optimized kernels with PyTorch fallback.
    
    Args:
        dry_run: If True, skip superoptimize and use original kernel graphs for fast testing
        
    Returns:
        HybridModel: Executable model with hybrid Mirage+PyTorch execution
    """
    
    print("Building computation graph...")
    unique_operators = {}
    operators = get_computation_graph(model, dummy_input, unique_operators, "onnx")
    
    # Load parameters from ONNX model
    import onnx
    onnx_model = onnx.load("scripts/onnx/inferred_model.onnx")
    parameter_dict = {init.name: torch.from_numpy(onnx.numpy_helper.to_array(init)) 
                     for init in onnx_model.graph.initializer}
    
    print("Splitting graph into supported/unsupported subgraphs...")
    if IGNORE_OPS is None:
        IGNORE_OPS = {"Identity", "Cast", "CastLike", "Constant", "Dropout"}
    if UNSUPPORTED_OPS is None:
        UNSUPPORTED_OPS = set()
        
    subgraphs, deps, sorted_ops = process_operator_graph(operators, IGNORE_OPS, UNSUPPORTED_OPS)
    
    print("Applying in-context learning partitioning to large Mirage subgraphs...")

    partitioner = InCtxPartitioner(dataset_root, mirage_root, model_name=model_name, max_nodes_per_partition=max_nodes_per_partition, scale=scale)
    fine_grained_partitions = []

    for sg_id, (sg_dict, sg_type) in enumerate(subgraphs):
        if sg_type == "mirage" and len(sg_dict) > max_nodes_per_partition:
            print(f"  Partitioning subgraph {sg_id} ({len(sg_dict)} ops)...")
            
            partitions = partitioner.partition(sg_dict)

            # TODO: currently we need to create a mapping from names to nodes, as the in-context partitioner
            # does not preserve the original Operator objects, this can be optimized
            name_to_node = {op.name: op for op in sg_dict}
            
            for p_id, partition in enumerate(partitions):
                partition_ops = [name_to_node[name] for name in partition]
                
                print(f"Partition {p_id}: {[op.name for op in partition_ops]}")
                
                partition_subgraph = {op: True for op in partition_ops}
                fine_grained_partitions.append((partition_subgraph, "mirage"))
                print(f"    Created partition {p_id}: {len(partition_ops)} ops")
        else:
            fine_grained_partitions.append((sg_dict, sg_type))

    # Build execution plan
    print(f"\nOptimizing {sum(1 for _, t in fine_grained_partitions if t == 'mirage')} Mirage partitions...")
    execution_plan = []
    
    for pid, (sg_dict, sg_type) in enumerate(fine_grained_partitions):
        # Print partition info
        ops_in_partition = [op for op in sorted_ops if op in sg_dict]
        print(f"\n=== Partition {pid} ({sg_type}): {len(sg_dict)} ops ===")
        print(f"  Operators: {[op.name for op in ops_in_partition]}")
        
        if sg_type == "mirage":
            kernel_graph, dims = to_kernel_graph(sg_dict)
            try:
                if dry_run:
                    # Dry run mode: compile original kernel without superoptimize
                    # Create dummy inputs for compilation (Mirage uses float16)
                    dummy_inputs = []
                    for dim_info in dims:
                        if dim_info[1] == "V":  # Variable input
                            dummy_inputs.append(torch.randn(dim_info[0], dtype=torch.float16, device="cuda"))
                        elif dim_info[1] == "C":  # Constant input
                            dummy_inputs.append(torch.full(dim_info[0], dim_info[2], dtype=torch.float16, device="cuda"))
                    
                    # Compile the kernel
                    kernel_graph.compile(inputs=dummy_inputs)
                    optimized_kernel = kernel_graph
                    print(f"  → Result: Dry run mode - compiled original kernel (no optimization)")
                else:
                    result = kernel_graph.superoptimize()
                    # Handle both tuple (normal optimization) and single value (cached) returns
                    if isinstance(result, tuple):
                        optimized_kernel, _ = result
                    else:
                        optimized_kernel = result
                    print(f"  ✓ Result: Mirage kernel optimized successfully")
                
                # Extract tensor IDs from subgraph
                produced_in_subgraph = set()
                consumed_in_subgraph = set()
                for op in sg_dict:
                    for _, tid in op.output_tensor_shapes:
                        produced_in_subgraph.add(tid)
                    for _, tid in op.input_tensor_shapes:
                        consumed_in_subgraph.add(tid)
                
                # Input tensors: consumed but not produced in subgraph
                input_ids = []
                for op in sg_dict:
                    for _, tid in op.input_tensor_shapes:
                        if tid not in produced_in_subgraph and tid not in input_ids:
                            input_ids.append(tid)
                
                # Output tensors: produced but not consumed in subgraph (exported to outside)
                output_ids = [tid for tid in produced_in_subgraph if tid not in consumed_in_subgraph]
                const_dims = [(d[0], d[2]) for d in dims if d[1] == "C"]
                
                execution_plan.append(("mirage", (optimized_kernel, input_ids, output_ids, const_dims)))
            except Exception as e:
                print(f"  ✗ Result: Superoptimize failed - {str(e)[:80]}...")
                print(f"     Fallback: Using PyTorch execution")
                for op in sg_dict:
                    execution_plan.append(("pytorch", (op, [t for _, t in op.input_tensor_shapes], op.output_tensor_shapes[0][1])))
        else:
            print(f"  → Result: Using PyTorch execution (unsupported operations)")
            for op in sg_dict:
                execution_plan.append(("pytorch", (op, [t for _, t in op.input_tensor_shapes], op.output_tensor_shapes[0][1])))
    
    # Topological sort execution plan
    def get_plan_io(step):
        step_type, payload = step
        if step_type == "mirage":
            return payload[1], payload[2]  # input_ids, output_ids
        else:  # pytorch
            return payload[1], [payload[2]]  # input_ids, [output_id]
    
    sorted_plan = []
    all_produced = {tid for op in sorted_ops for _, tid in op.output_tensor_shapes}
    available = set(tid for op in sorted_ops for _, tid in op.input_tensor_shapes if tid not in all_produced)
    remaining = execution_plan.copy()
    
    while remaining:
        for step in remaining[:]:
            inputs, outputs = get_plan_io(step)
            if all(tid in available for tid in inputs):
                sorted_plan.append(step)
                available.update(outputs)
                remaining.remove(step)
        if len(sorted_plan) == len(execution_plan):
            break
    
    execution_plan = sorted_plan
    print(f"\n✓ Sorted execution plan: {len(execution_plan)} steps")
    
    # Infer model inputs/outputs
    all_produced = {tid for op in sorted_ops for _, tid in op.output_tensor_shapes}
    all_input_tensor_ids = [tid for op in sorted_ops for _, tid in op.input_tensor_shapes if tid not in all_produced]
    all_input_tensor_ids = list(dict.fromkeys(all_input_tensor_ids))
    
    # Separate real inputs from parameters (ONNX orders: real_inputs + parameters)
    num_real_inputs = 1 if not isinstance(dummy_input, (tuple, list)) else len(dummy_input)
    input_tensor_ids = all_input_tensor_ids[:num_real_inputs]
    param_names = list(parameter_dict.keys())
    parameter_tensors = {tid: parameter_dict[param_names[i]] 
                        for i, tid in enumerate(all_input_tensor_ids[num_real_inputs:])}
    
    all_consumed = {tid for op in sorted_ops for _, tid in op.input_tensor_shapes}
    output_tensor_ids = [tid for op in sorted_ops for _, tid in op.output_tensor_shapes if tid not in all_consumed]
    
    print(f"✓ Loaded {len(parameter_tensors)} parameters, {len(input_tensor_ids)} real inputs, {len(output_tensor_ids)} outputs")
    return HybridModel(execution_plan, input_tensor_ids, output_tensor_ids, parameter_tensors)

def time_kernels(kernels, input_dims, device, iterations=1):
    times = []
    for kernel, dims in zip(kernels, input_dims):
        total_time = 0
        for _ in range(iterations):
            inputs = []
            for dim in dims:
                if (dim[1] == "V"):
                    inputs.append(torch.randn(dim[0], requires_grad=True).to(device))
                elif (dim[1] == "C"):
                    inputs.append(torch.full(dim[0], dim[2]).to(device))
            start = time.time()
            _ = kernel(inputs=inputs)
            total_time += time.time() - start
        times.append(total_time / iterations)
    return times


"""
Example usage:

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

from partition_graph import partition_graph

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Define dummy tensors and run it through the model to get the computational graph
dummy_input_ids = torch.randint(0, 30522, (2, 128)).to(device)
dummy_attention_mask = torch.ones((2, 128), requires_grad=True).to(device)
dummy_labels = torch.tensor([0, 1], dtype=torch.long).to(device)

dummy_outputs = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask, labels=dummy_labels)
dummy_loss = dummy_outputs.loss

subgraphs, unique_operators = partition_graph(dummy_loss)
all_kernels, kernel_input_dims = generate_all_kernels(subgraphs)
times = time_kernels(all_kernels, kernel_input_dims, device)
"""

"""
Format of subgraphs:

The format of the subgraphs are in an adjacency list format, the keys are Operator objects each
containing the name, grad_fn, the list of input/output operators that the operator recieves
tensors from/sends tensors to, as well as the shapes of the input and output tensors to this operator.
The values are lists of Operator objects.

The entry op1: [op2, op3] means that op1 outputs its tensors to op2 and op3, i.e.

op1 --> op2
|
v
op3

"""
