import torch
import os
from pathlib import Path
import mirage as mi
import numpy as np
import warnings
from build_computation_graph import get_computation_graph
from utils import to_kernel_graph
from generate_dag import solve_partitions
from graph_splitter import process_operator_graph
from gnn_xgboost import GNNXGBoost

# Get the directory where this file is located
_RUNTIME_DIR = Path(__file__).parent

CAST_ID_TO_DTYPE = {
    1: torch.float16, # supposed to be float32 but we use float16 throughout
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

def is_connected(nodes, adj):
  if len(nodes) <= 1: return True
  visited = set([nodes[0]])
  queue = [nodes[0]]
  while queue:
    node = queue.pop(0)
    for next_node in nodes:
      if next_node not in visited and (adj[node][next_node] or adj[next_node][node]):
        visited.add(next_node)
        queue.append(next_node)
  return len(visited) == len(nodes)


def check_subgraph_exceeds_mirage_constraint(sg_dict, max_mirage_ops):
    """
    Check if a subgraph exceeds the mirage constraint:
    num_inputs + num_outputs + num_operators > max_mirage_ops
    
    Args:
        sg_dict: Dictionary representing the subgraph (keys are operators)
        max_mirage_ops: Maximum allowed value for the constraint
    
    Returns:
        True if constraint is exceeded, False otherwise
    """
    # Count operators
    num_operators = len(sg_dict)
    
    # Collect all tensor IDs produced and consumed in the subgraph
    produced_in_subgraph = set()
    consumed_in_subgraph = set()
    for op in sg_dict:
        for _, tid in op.output_tensor_shapes:
            produced_in_subgraph.add(tid)
        for _, tid in op.input_tensor_shapes:
            consumed_in_subgraph.add(tid)
    
    # Input tensors: consumed but not produced in subgraph
    num_inputs = len(consumed_in_subgraph - produced_in_subgraph)
    
    # Output tensors: produced but not consumed in subgraph
    num_outputs = len(produced_in_subgraph - consumed_in_subgraph)
    
    # Check if constraint is exceeded
    return num_inputs + num_outputs + num_operators > max_mirage_ops


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
        self._expanded_cache = {}  # Cache for dimension-expanded tensors (e.g., 2D weight -> 3D)
        self._cast_cache = {}
        
        # Initialize operation handlers
        self._init_op_handlers()
    
    def _handle_binary_op_with_scalar(self, op, inputs, fn_name):
        """Handle binary ops that can work with scalars from additional_params"""
        if len(inputs) == 1 and hasattr(op, 'additional_params') and 'arg1' in op.additional_params:
            scalar = op.additional_params['arg1']
            op_map = {
                "Add": lambda x, s: x + s,
                "Sub": lambda x, s: x - s,
                "Mul": lambda x, s: x * s,
                "Div": lambda x, s: x / s,
            }
            return op_map[fn_name](inputs[0], scalar)
        elif len(inputs) >= 2:
            return getattr(torch, fn_name.lower())(inputs[0], inputs[1])
        else:
            raise ValueError(f"{fn_name} requires 2 inputs or additional_params, got {len(inputs)} inputs")
    
    def _handle_gemm(self, inputs):
        """ONNX Gemm: Y = alpha * A * B^T + beta * C (default transB=1)"""
        matmul_result = torch.matmul(inputs[0], inputs[1].T)
        if len(inputs) > 2:  # Has bias
            return matmul_result + inputs[2]
        return matmul_result
    
    def _handle_cast(self, op, inputs):
        """Handle Cast and CastLike operations"""
        to_dtype = CAST_ID_TO_DTYPE.get(op.kwargs.get("to"), None)
        if to_dtype is not None:
            if inputs[0].dtype != to_dtype:
                # check in cache
                cache_key = (id(inputs[0]), to_dtype)
                if cache_key in self._cast_cache:
                    return self._cast_cache[cache_key]
                self._cast_cache[cache_key] = inputs[0].to(dtype=to_dtype)
                return self._cast_cache[cache_key]
            return inputs[0]
        else:
            raise NotImplementedError(f"Cast to dtype id '{op.kwargs.get('to')}' not implemented")
    
    def _init_op_handlers(self):
        """Initialize the operation dispatch table"""
        self._op_handlers = {
            "Add": lambda op, inputs: self._handle_binary_op_with_scalar(op, inputs, "Add"),
            "Sub": lambda op, inputs: self._handle_binary_op_with_scalar(op, inputs, "Sub"),
            "Mul": lambda op, inputs: self._handle_binary_op_with_scalar(op, inputs, "Mul"),
            "Div": lambda op, inputs: self._handle_binary_op_with_scalar(op, inputs, "Div"),
            "Pow": lambda op, inputs: self._handle_binary_op_with_scalar(op, inputs, "Pow"),
            "MatMul": lambda op, inputs: torch.matmul(inputs[0], inputs[1]),
            "Gemm": lambda op, inputs: self._handle_gemm(inputs),
            "Relu": lambda op, inputs: torch.relu(inputs[0]),
            "Tanh": lambda op, inputs: torch.tanh(inputs[0]),
            "Sigmoid": lambda op, inputs: torch.sigmoid(inputs[0]),
            "Exp": lambda op, inputs: torch.exp(inputs[0]),
            "Neg": lambda op, inputs: torch.neg(inputs[0]),
            "Sqrt": lambda op, inputs: torch.sqrt(inputs[0]),
            "Reciprocal": lambda op, inputs: torch.reciprocal(inputs[0]),
            "ReduceSum": lambda op, inputs: torch.sum(inputs[0], **op.kwargs, keepdims=True),
            "Transpose": lambda op, inputs: torch.permute(inputs[0], dims=op.kwargs["perm"]),
            "Reshape": lambda op, inputs: inputs[0].reshape(op.output_tensor_shapes[0][0]),
            "Abs": lambda op, inputs: torch.abs(inputs[0]),
            "Expand": lambda op, inputs: torch.broadcast_to(inputs[0], torch.broadcast_shapes(inputs[0].shape, tuple(op.output_tensor_shapes[0][0]))),
            "Gather": lambda op, inputs: torch.nn.functional.embedding(inputs[1], inputs[0]),
            "Unsqueeze": lambda op, inputs: torch.unsqueeze(inputs[0], dim=0),
            "Cast": lambda op, inputs: self._handle_cast(op, inputs),
            "CastLike": lambda op, inputs: self._handle_cast(op, inputs),
            "Constant": lambda op, inputs: op.kwargs["t"],
            "Identity": lambda op, inputs: inputs[0],
        }
    
    def _get_params_on(self, device, dtype):
        """Return dict of parameters placed on device, keeping their original dtype."""
        # Note: We only cache by device, not dtype, because parameters should keep their original dtype
        # (e.g. embedding weights stay float32/float16 even if input tokens are int64)
        key = (device,)
        params = self._param_cache.get(key)
        if params is None:
            # Move to device once, keep original dtype, cache for subsequent calls
            moved = {}
            for tid, p in self.parameter_tensors.items():
                tp = p
                if tp.device != device:
                    warnings.warn(f"Moving parameter tensor id {tid} from {tp.device} to {device}. Consider initializing model parameters on the target device to avoid this overhead.")
                    tp = tp.to(device=device, non_blocking=True)
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
    
    def __call__(self, *inputs, debug=False):
        if len(inputs) != len(self.input_tensor_ids):
            raise ValueError(f"Expected {len(self.input_tensor_ids)} input(s), got {len(inputs)}")
        device = inputs[0].device if inputs else torch.device('cuda')
        # Always use float16 for Mirage computations (inputs may be int64 for embeddings)
        dtype = torch.float16
        intermediates = {tid: tensor for tid, tensor in zip(self.input_tensor_ids, inputs)}
        params_on_target = self._get_params_on(device, dtype)
        intermediates.update(params_on_target)

        with torch.inference_mode():
            for i, (step_type, payload) in enumerate(self.execution_plan):
                if step_type == "mirage":
                    kernel, input_ids, output_ids, const_dims, expected_shapes = payload
                    
                    # Prepare inputs in fp16 ONLY if needed; avoid redundant casts
                    kernel_inputs = []
                    for j, tid in enumerate(input_ids):
                        if tid not in intermediates:
                            raise ValueError(f"Input tensor id {tid} not found in intermediates")
                        t = intermediates[tid]
                        
                        if t.dtype is not torch.float16:
                            if debug:
                                _ = float(t.abs().sum()) # if this isn't here the cast to float16 might error out
                            warnings.warn(f"Casting input tensor id {tid} from {t.dtype} to torch.float16. Consider providing inputs in float16 to avoid this overhead.")                            
                            t = t.to(torch.float16)
                        
                        # Match dimensions with expected shape (for MatMul broadcasting compatibility)
                        expected_shape = expected_shapes[j]
                        needs_expand = len(t.shape) < len(expected_shape)
                        
                        if needs_expand:
                            # Need to expand dimensions (e.g., 2D weight -> 3D for batch matmul)
                            # Check cache first to avoid repeated expand+contiguous
                            cache_key = (tid, tuple(expected_shape))
                            if cache_key in self._expanded_cache:
                                # Use cached expanded+contiguous tensor
                                t = self._expanded_cache[cache_key]
                            else:
                                # First time: expand and will be made contiguous below
                                batch_size = expected_shape[0]
                                t = t.unsqueeze(0).expand(batch_size, *t.shape)
                        
                        # Ensure tensor is contiguous for Mirage kernels
                        if not t.is_contiguous():
                            t = t.contiguous()
                            # If this was an expanded tensor, cache the contiguous version
                            if needs_expand:
                                cache_key = (tid, tuple(expected_shape))
                                self._expanded_cache[cache_key] = t
                        
                        kernel_inputs.append(t)

                    # Append cached constants (fp16) without recreating them every call
                    dev = kernel_inputs[0].device if kernel_inputs else device
                    for shape, value in const_dims:
                        c = self._get_const(dev, shape, value, dtype=torch.float16)
                        kernel_inputs.append(c)

                    outputs = kernel(inputs=kernel_inputs)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    # Mirage outputs are always float16, keep them as-is
                    for tid, tensor in zip(output_ids, outputs):
                        if debug:
                            print(f"Mirage step {i}: tensor id {tid}, shape {tensor.shape}, dtype {tensor.dtype}")
                            _ = float(tensor.abs().sum())
                        # Keep Mirage outputs in float16, don't convert to input dtype
                        intermediates[tid] = tensor
                elif step_type == "pytorch":
                    op, input_ids, output_id = payload
                    
                    if not all(tid in intermediates for tid in input_ids):
                        missing = [tid for tid in input_ids if tid not in intermediates]
                        raise ValueError(f"Input tensor ids {missing} not found in intermediates for PyTorch op {op.name}")
                    inputs = [intermediates[tid] for tid in input_ids]
                    
                    result = self._execute_pytorch_op(op, inputs)
                    if debug:
                        print(f"PyTorch step {i}: tensor id {output_id}, shape {result.shape}, dtype {result.dtype}")
                        _ = float(result.abs().sum())
                    
                    intermediates[output_id] = result
        
        outputs = [intermediates[tid] for tid in self.output_tensor_ids]
        return outputs[0] if len(outputs) == 1 else outputs
    
    def _execute_pytorch_op(self, op, inputs):
        """Execute a PyTorch operation using the dispatch table"""
        fn = op.fn
        
        # Lookup and execute the operation
        handler = self._op_handlers.get(fn)
        if handler is None:
            raise NotImplementedError(f"PyTorch fallback for '{fn}' not implemented")
        
        return handler(op, inputs)

def partition_graph_with_dp(model, 
                          dummy_input, 
                          IGNORE_OPS, 
                          UNSUPPORTED_OPS,
                          cost_model,
                          max_nodes_per_partition=4,
                          max_mirage_ops=9,
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
    operators, tensor_id_to_name = get_computation_graph(model, dummy_input, unique_operators, "onnx")
    print(tensor_id_to_name)

    print(f"Unique operators in graph: {unique_operators}")
    
    # Load parameters from ONNX model
    import onnx
    onnx_model = onnx.load("scripts/onnx/inferred_model.onnx")
    parameter_dict = {init.name: torch.from_numpy(onnx.numpy_helper.to_array(init)).to(torch.device("cuda")).to(torch.float16)
                     for init in onnx_model.graph.initializer}
    
    # Get actual output tensor IDs from ONNX model
    output_tensor_names = [output.name for output in onnx_model.graph.output]
    name_to_tensor_id = {name: tid for tid, name in tensor_id_to_name.items()}
    onnx_output_tensor_ids = [name_to_tensor_id[name] for name in output_tensor_names if name in name_to_tensor_id]
    
    print("Splitting graph into supported/unsupported subgraphs...")
        
    subgraphs, _, sorted_ops = process_operator_graph(operators, IGNORE_OPS, UNSUPPORTED_OPS)
    
    # Print original subgraph statistics
    print("\n" + "="*60)
    print("Original Subgraph Details:")
    print("="*60)
    for sg_id, (sg_dict, sg_type) in enumerate(subgraphs):
        print(f"  Subgraph {sg_id} ({sg_type}): {len(sg_dict)} ops")
    
    print("\n" + "="*60)
    print("Applying dynamic programming partitioning to large Mirage subgraphs...")
    print("="*60)
    
    print(f"Initializing cost model: {cost_model}")
    if cost_model == "gnn-xgboost":
        cm = GNNXGBoost(
            encoder_ckpt=str(_RUNTIME_DIR / "cost_model/models/11_25_exec_time_gine_best_full_lr3e-03.pt"),
            encoder_cfg={"hidden": 128, "layers": 8, "dropout": 0.2},
            xgb_model_path=str(_RUNTIME_DIR / "cost_model/models/11_25_exec_time_xgb_best_xgb.json")
        )
    elif cost_model == "dnn-abacus":
        raise NotImplementedError("DNNAbacus support not available")
    else:
        raise NotImplementedError(f"Cost model not implemented: {cost_model}")
        
    def cost_function(nodes, adj=None):
        nodes_idx = [i for _, i in nodes]
        nodes = [op for op, _ in nodes]
        if adj is not None and not is_connected(nodes_idx, adj):
            return float('inf')
        return cm(nodes)

    fine_grained_partitions = []
    
    for sg_id, (sg_dict, sg_type) in enumerate(subgraphs):
        if sg_type == "mirage" and check_subgraph_exceeds_mirage_constraint(sg_dict, max_mirage_ops):
            print(f"  Partitioning subgraph {sg_id} ({len(sg_dict)} ops)...")
            
            # Extract operators in topological order
            ops_list = [op for op in sorted_ops if op in sg_dict] 
            n = len(ops_list)
            adj = np.zeros((n, n), dtype=int)
            ops_idx_list = [(op, i) for i, op in enumerate(ops_list)]
            
            # Build adjacency matrix
            for i, op in enumerate(ops_list):
                for output_op in op.output_ops:
                    if output_op in ops_list:
                        j = ops_list.index(output_op)
                        adj[i][j] = 1
            
            # Apply DAG partitioning with connectivity constraint
            def cost_function_with_adj(nodes):
                return cost_function(nodes, adj)
            partition_boundaries = solve_partitions(ops_idx_list, cost_function_with_adj, max_nodes_per_partition, adj, max_mirage_ops)
            
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
    
    # Print final partition statistics
    mirage_count = sum(1 for _, t in fine_grained_partitions if t == "mirage")
    pytorch_count = sum(1 for _, t in fine_grained_partitions if t == "pytorch")
    print("\n" + "="*60)
    print("Final Partition Statistics:")
    print("="*60)
    print(f"  Total partitions: {len(fine_grained_partitions)}")
    print(f"  - Mirage partitions: {mirage_count}")
    print(f"  - PyTorch partitions: {pytorch_count}")
    print(f"  Original subgraphs: {len(subgraphs)} → Final partitions: {len(fine_grained_partitions)}")
    
    # Build execution plan
    print(f"\nOptimizing {sum(1 for _, t in fine_grained_partitions if t == 'mirage')} Mirage partitions...")
    execution_plan = []
    cached_mirage_kernels = {}
    
    for pid, (sg_dict, sg_type) in enumerate(fine_grained_partitions):
        # Print partition info
        ops_in_partition = [op for op in sorted_ops if op in sg_dict]
        print(f"\n=== Partition {pid} ({sg_type}): {len(sg_dict)} ops ===")
        print(f"  Operators: {[op.name for op in ops_in_partition]}")
        
        if sg_type == "mirage":
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
            
            # Output tensors to export from this partition:
            # - Any tensor produced in this subgraph that is consumed OUTSIDE the subgraph
            # - Plus any tensor that is a FINAL output of the whole graph (consumed by no op globally)
            all_inputs_outside = {tid for op in sorted_ops if op not in sg_dict for _, tid in op.input_tensor_shapes}
            all_consumed_global = {tid for op in sorted_ops for _, tid in op.input_tensor_shapes}
            exported_to_outside = produced_in_subgraph.intersection(all_inputs_outside)
            final_outputs = {tid for tid in produced_in_subgraph if tid not in all_consumed_global}
            output_ids = sorted(exported_to_outside.union(final_outputs))

            kernel_graph, dims = to_kernel_graph(sg_dict, output_ids)
            dummy_inputs = []
            for dim_info in dims:
                if dim_info[1] == "V":  # Variable input
                    dummy_inputs.append(torch.randn(dim_info[0], dtype=torch.float16, device="cuda"))
                elif dim_info[1] == "C":  # Constant input
                    dummy_inputs.append(torch.full(dim_info[0], dim_info[2], dtype=torch.float16, device="cuda"))
            
            try:
                h = str(kernel_graph.get_owner_independent_hash())
                # TODO: Sqrt doesn't work with superoptimizer currently
                if dry_run:
                    # # Dry run mode: compile original kernel without superoptimize
                    # # Create dummy inputs for compilation (Mirage uses float16)

                    # The dummy inputs are not actually used in dry run mode
                    optimized_kernel = kernel_graph
                    print(f"  → Result: Dry run mode - use original kernel (no optimization)")
                else:        
                    if os.path.isfile("optimized_" + h + ".json"):
                        print(f"  → Result: Loading cached optimized kernel for hash {h}")
                        optimized_kernel = mi.new_kernel_graph()
                        optimized_kernel.from_json("optimized_" + h + ".json")
                    else:
                        # Superoptimize and check for success
                        kernel_graph = kernel_graph.superoptimize()
                        
                        # Check if superoptimization succeeded
                        if kernel_graph is None:
                            raise RuntimeError("Superoptimization failed: returned None (0 mugraphs found)")
                        
                        # Handle both tuple (normal optimization) and single value (cached) returns
                        if isinstance(kernel_graph, tuple):
                            optimized_kernel, _ = kernel_graph
                        else:
                            optimized_kernel = kernel_graph
                        
                        # Verify we got a valid kernel
                        if optimized_kernel is None:
                            raise RuntimeError("Superoptimization failed: no valid kernel graph")
                        
                        optimized_kernel.to_json("optimized_" + h + ".json")
                        print(f"  ✓ Result: Mirage kernel cached with hash {h}")
                    print(f"  ✓ Result: Mirage kernel optimized successfully")
                kernel_cache_key = (h, tuple(dummy_input.shape for dummy_input in dummy_inputs))
                if kernel_cache_key in cached_mirage_kernels:
                    optimized_kernel = cached_mirage_kernels[kernel_cache_key]
                    print(f"     ✓ Using cached compiled kernel")
                else:
                    optimized_kernel.compile(inputs=dummy_inputs)
                    cached_mirage_kernels[kernel_cache_key] = optimized_kernel
                    print(f"     ✓ Compiled kernel and cached")
                # check for kernel validity
                print(f"Checking kernel validity...")
                output = optimized_kernel(inputs=dummy_inputs)
                _ = float(output[0].abs().sum())
                print(f"     ✓ Kernel validity verified")

                const_dims = [(d[0], d[2]) for d in dims if d[1] == "C"]
                # Save expected input shapes for runtime dimension matching
                expected_input_shapes = [d[0] for d in dims if d[1] == "V"]
                execution_plan.append(("mirage", (optimized_kernel, input_ids, output_ids, const_dims, expected_input_shapes)))
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
        changed = False
        for step in remaining[:]:
            inputs, outputs = get_plan_io(step)
            if all(tid in available for tid in inputs):
                changed = True
                sorted_plan.append(step)
                available.update(outputs)
                remaining.remove(step)
        if not changed:
            raise Exception("Cannot sort out tensor sizes")
        print(f"Sorted {len(sorted_plan)}/{len(execution_plan)} tensors")
        if len(sorted_plan) == len(execution_plan):
            break
    
    execution_plan = sorted_plan
    print(f"\n✓ Sorted execution plan: {len(execution_plan)} steps")
    
    # Infer model inputs/outputs
    all_produced = {tid for op in sorted_ops for _, tid in op.output_tensor_shapes}
    all_input_tensor_ids = [tid for op in sorted_ops for _, tid in op.input_tensor_shapes if tid not in all_produced]
    
    # Separate real inputs from parameters (ONNX orders: real_inputs + parameters)
    input_tensor_ids = []
    # Safer loop with index checking
    parameter_tensors = {}
    for tid in all_input_tensor_ids:
        pname = tensor_id_to_name[tid]
        if pname not in parameter_dict:
            input_tensor_ids.append(tid)
        else:
            parameter_tensors[tid] = parameter_dict[pname]
    
    # Use the actual ONNX output tensor IDs instead of inferring from unconsumed tensors
    output_tensor_ids = onnx_output_tensor_ids
    
    print(f"✓ Loaded {len(parameter_tensors)} parameters, {len(input_tensor_ids)} real inputs, {len(output_tensor_ids)} outputs")
    return HybridModel(execution_plan, input_tensor_ids, output_tensor_ids, parameter_tensors)
