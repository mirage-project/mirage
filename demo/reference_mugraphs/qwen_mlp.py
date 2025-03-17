import mirage as mi
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)


def torch_qwen_mlp(X, G, W):
    variance = X.pow(2).mean(-1, keepdim=True)
    X = X * torch.rsqrt(variance)
    X = torch.mul(X, G)
    O = torch.matmul(X, W)

    return O

if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(1, 3584), dtype=mi.bfloat16)
    G = graph.new_input(dims=(1, 3584), dtype=mi.bfloat16)
    W = graph.new_input(dims=(3584, 2*18944), strides=(1, 3584), dtype=mi.bfloat16)
    D = graph.rms_norm(X, normalized_shape=(3584,))
    D = graph.mul(D, G)
    O = graph.matmul(D, W)
    graph.mark_output(O)
    optimized_graph = graph.superoptimize(config="mlp")

    input_tensors = [
        torch.randn(1, 1, 3584,dtype=torch.bfloat16, device='cuda:0'),
        torch.randn(3584,  dtype=torch.bfloat16, device='cuda:0'),
        torch.randn(3584, 2*18944,  dtype=torch.bfloat16, device='cuda:0'),
    ]
    input_tensors[2] = torch.as_strided(input_tensors[2], (3584, 37888), (1, 3584))
    
    input_strides = []
    dtensors = optimized_graph.cygraph.get_input_dtensors()
    assert len(dtensors) == len(
        input_tensors
    ), "Given number of inputs do not match the uGraph's inputs"
    for i in range(len(dtensors)):
        input_strides.append(optimized_graph.cygraph.get_input_dtensor_layout(dtensors[i]))

    # input_strides = [tensor.stride() for tensor in input_tensors]
    p = mi.generate_cuda_program(optimized_graph.cygraph, target_cc=86, input_strides=input_strides)
    print(p["code"])

    outputs = optimized_graph(inputs=input_tensors)
    print(f"Shape1: {input_tensors[0].shape}")       
    print(f"Stride1: {input_tensors[0].stride()}")   
    print(f"Dtype1: {input_tensors[0].dtype}")       


    print(f"Shape2: {input_tensors[1].shape}")       
    print(f"Stride2: {input_tensors[1].stride()}")   
    print(f"Dtype2: {input_tensors[1].dtype}")       


    print(f"Shape3: {input_tensors[2].shape}")       
    print(f"Stride3: {input_tensors[2].stride()}")   
    print(f"Dtype3: {input_tensors[2].dtype}")       


    print("input", input_tensors[0])
    print(outputs[0])
    print(torch_qwen_mlp(input_tensors[0], input_tensors[1], input_tensors[2]))

