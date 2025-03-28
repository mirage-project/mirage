import mirage as mi
import numpy as np
import torch
torch.set_printoptions(sci_mode=False)


def torch_qwen_prenorm(X, G, W):
    variance = X.pow(2).mean(-1, keepdim=True)
    X = X * torch.rsqrt(variance)
    X = torch.mul(X, G)
    O = torch.matmul(X, W)

    return O

if __name__ == "__main__":

    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(1, 2048), dtype=mi.bfloat16)
    G = graph.new_input(dims=(1, 2048), dtype=mi.bfloat16)
    W = graph.new_input(dims=(2048, 2560), strides=(1, 2048), dtype=mi.bfloat16)
    D = graph.rms_norm(X, normalized_shape=(2048,))
    D = graph.mul(D, G)
    O = graph.matmul(D, W)
    graph.mark_output(O)
    opt_kernel = graph.superoptimize(config="mlp")

    input_tensors = [
        torch.randn(1, 1, 2048,dtype=torch.bfloat16, device='cuda:0'),
        torch.randn(2048,  dtype=torch.bfloat16, device='cuda:0'),
        torch.randn(2048, 2560,  dtype=torch.bfloat16, device='cuda:0'),
    ]

    input_tensors[2] = torch.as_strided(input_tensors[2], (2048, 2560), (1, 2048))
    outputs = opt_kernel(inputs=input_tensors)
    print(outputs[0])
    print(torch_qwen_prenorm(input_tensors[0], input_tensors[1], input_tensors[2]))


