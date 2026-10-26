"""Utility functions for graph partitioning and kernel operations.

This module provides helper functions for converting subgraphs to Mirage kernel graphs,
mapping operations to their corresponding Mirage graph methods, and benchmarking kernel
execution times.
"""

import mirage as mi
import torch
import time

def function_map(graph, func, inputs, kwargs={}):
    """Map operation functions to their corresponding Mirage graph methods.
    
    Args:
        graph: Mirage kernel graph object to apply operations on.
        func: Operation object with 'fn' attribute specifying the operation type.
        inputs: List of input tensors for the operation.
        kwargs: Additional keyword arguments to pass to the operation (default: {}).
    
    Returns:
        Output tensor(s) from the applied operation.
    
    Raises:
        NotImplementedError: If the operation type is not supported.
    """
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

def to_kernel_graph(subgraph, output_ids=[]):
    """Convert an adjacency list formatted subgraph to a Mirage kernel graph.
    
    Args:
        subgraph: Dictionary representing the subgraph in adjacency list format,
                 where keys are operations and values are their connections.
        output_ids: List of tensor IDs to mark as outputs. If empty, all tensors
                   with zero reference count are marked as outputs (default: []).
    
    Returns:
        tuple: A tuple containing:
            - graph: The generated Mirage kernel graph.
            - dims: List of tuples describing input dimensions, where each tuple
                   contains (shape, type, [value]) with type being 'V' for variable
                   or 'C' for constant.
    """
    graph = mi.new_kernel_graph()
    dims = []
    # stores output tensors of operations + their reference counts based on ID
    intermediates = {}
    for op, _ in subgraph.items():
        inputs = []
        for (shape, tensor_id) in op.input_tensor_shapes:
            if tensor_id not in intermediates:
                dims.append((shape, "V"))
                new_input = graph.new_input(dims=shape, dtype=mi.float16)
                inputs.append(new_input)
                # Record this input tensor in intermediates to avoid duplicates
                intermediates[tensor_id] = [new_input, 0]
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
    if len(output_ids) > 0:
        for out_id in output_ids:
            graph.mark_output(intermediates[out_id][0])
    else:
        for _, tsr_cnt in intermediates.items():
            if tsr_cnt[1] == 0: graph.mark_output(tsr_cnt[0])
    return graph, dims

def time_kernels(kernels, input_dims, device, iterations=1):
    """Benchmark execution time of multiple kernels.
    
    Args:
        kernels: List of compiled kernel functions to benchmark.
        input_dims: List of dimension specifications for each kernel, where each
                   specification is a list of tuples (shape, type, [value]).
        device: PyTorch device to run the kernels on (e.g., 'cuda', 'cpu').
        iterations: Number of times to run each kernel for averaging (default: 1).
    
    Returns:
        list: Average execution times in seconds for each kernel.
    
    Notes:
        - For dimensions with type 'V', random tensors are generated.
        - For dimensions with type 'C', constant tensors are created with the
          specified value.
    """
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

def is_connected(nodes, adj):
    """Check if a set of nodes forms a connected subgraph.
    
    Uses breadth-first search (BFS) to determine if all nodes in the given set
    are reachable from each other through the adjacency matrix.
    
    Args:
        nodes: List of node identifiers to check for connectivity.
        adj: 2D adjacency matrix where adj[i][j] is True if there's an edge
             between node i and node j.
    
    Returns:
        bool: True if all nodes are connected (form a single connected component),
              False otherwise. Returns True for single node or empty node lists.
    """
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
