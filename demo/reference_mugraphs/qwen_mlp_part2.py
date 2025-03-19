import mirage as mi
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)


def torch_qwen_mlp(X, Y, W):
    silu = torch.nn.SiLU()
    O = torch.matmul(silu(X) * Y, W)

    return O

if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(1, 18944), dtype=mi.bfloat16)
    Y = graph.new_input(dims=(1, 18944), dtype=mi.bfloat16)
    W = graph.new_input(dims=(18944, 3584), dtype=mi.bfloat16)
    D = graph.mul(graph.silu(X), Y)
    O = graph.matmul(D, W)
    graph.mark_output(O)
    optimized_graph = graph.superoptimize(config="mlp")

    input_tensors = [
        torch.randn(1, 1, 18944,dtype=torch.bfloat16, device='cuda:0'),
        torch.randn(1, 1, 18944,dtype=torch.bfloat16, device='cuda:0'),
        torch.randn(18944, 3584,  dtype=torch.bfloat16, device='cuda:0'),
    ]
    
    input_strides = []
    dtensors = optimized_graph.cygraph.get_input_dtensors()
    assert len(dtensors) == len(
        input_tensors
    ), "Given number of inputs do not match the uGraph's inputs"
    for i in range(len(dtensors)):
        dims, strides = optimized_graph.cygraph.get_input_dtensor_shape_and_stride(dtensors[i])
        input_strides.append(strides)

    # input_strides = [tensor.stride() for tensor in input_tensors]
    p = mi.generate_cuda_program(optimized_graph.cygraph, target_cc=86, input_strides=input_strides)
    print(p["code"])

    outputs = optimized_graph(inputs=input_tensors)
    print(outputs[0])
    print(torch_qwen_mlp(input_tensors[0], input_tensors[1], input_tensors[2]))
