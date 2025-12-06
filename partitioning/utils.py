import mirage as mi

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
def to_kernel_graph(subgraph, output_ids=[]):
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
