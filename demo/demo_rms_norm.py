import mirage as mi
import numpy as np
import torch

if __name__ == "__main__":
    mi.set_gpu_device_id(7)
    mirage_dtype = mi.bfloat16
    torch_dtype = mi.convert_dtype_to_torch_type(mirage_dtype)
    for _ in range(10):
        graph = mi.new_kernel_graph()
        X = graph.new_input(dims=(1, 7168), dtype=mirage_dtype)
        #G = graph.new_input(dims=(1, 7168), dtype=mirage_dtype)
        W = graph.new_input(dims=(7168, 16384), dtype=mirage_dtype)
        D = graph.rms_norm(X, normalized_shape=(7168,))
        #D = graph.mul(D, G)
        O = graph.matmul(D, W)
        graph.mark_output(O)
        optimized_graph = graph.superoptimize(config="mlp")

    input_tensors = [
        torch.randn(1, 1, 7168, dtype=torch_dtype, device='cuda:7'),
        #torch.randn(1, 7168, dtype=torch_dtype, device='cuda:7'),
        torch.randn(7168, 16384, dtype=torch_dtype, device='cuda:7'),
    ]

    outputs = optimized_graph(inputs=input_tensors)
    output = outputs[0]

    for _ in range(16):
        optimized_graph(inputs=input_tensors)

    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(1000):
        optimized_graph(inputs=input_tensors)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    mean_syn = curr_time / 1000

    print("Best muGraph run time (ms): ", mean_syn)
