import mirage as mi
import numpy as np
import torch

if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(16, 4096), dtype=mi.float16)
    W = graph.new_input(dims=(4096, 6144), dtype=mi.float16)
    D = graph.rms_norm(X, normalized_shape=(4096,))
    O = graph.matmul(D, W)
    graph.mark_output(O)
    optimized_graph = graph.superoptimize(config="mlp", backend="triton", 
            warmup_iters=2, profile_iters=6)

    with open("triton_rms_generated.py", "w") as f:
        f.write(mi.generate_triton_program(optimized_graph.cygraph, target_cc=10)["code"])
    
    print("Start running optimized graph...")
    input_tensors = [
        torch.randn(2, 4096, dtype=torch.float16, device=torch.device('cuda')),
        torch.randn(4096, 6144, dtype=torch.float16, device=torch.device('cuda')),
    ]

    outputs = optimized_graph(inputs=input_tensors)
    output = outputs[0]
    print(output.shape)
    print(output.stride(0), output.stride(1))

    for _ in range(2):
        print("Warmup run...")
        optimized_graph(inputs=input_tensors)

    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(6):
        print("Profile run...")
        optimized_graph(inputs=input_tensors)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    mean_syn = curr_time / 6

    print("Best muGraph run time (ms): ", mean_syn)

