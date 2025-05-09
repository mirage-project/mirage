import mirage as mi
import numpy as np
import torch

rms_norm = torch.nn.RMSNorm(4096, device='cuda:0', dtype=torch.float16)
#@torch.compile
def torch_rms_norm(X, W):
    D = rms_norm(X)
    E = torch.matmul(D, W)
    return E

if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(16, 4096), dtype=mi.float16)
    W = graph.new_input(dims=(4096, 4096), dtype=mi.float16)
    tb_graph = mi.new_threadblock_graph(grid_dim=(64,1,1), block_dim=(128,1,1), forloop_range=64, reduction_dimx=64)
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW = tb_graph.new_input(dtensor=W, input_map=(1, -1, -1), forloop_dim=0)
    tM = tb_graph.matmul(tX, tW)
    tAccX = tb_graph.forloop_accum(tX, "rms")
    tAccM = tb_graph.forloop_accum(tM)
    tO = tb_graph.div(tAccM, tAccX)
    tb_graph.new_output(stensor=tO, output_map=(1, -1, -1))
    O = graph.customized([X, W], tb_graph)
    graph.mark_output(O[0])
    graph.visualize("rms_norm")

    graph1 = mi.new_kernel_graph()
    X = graph1.new_input(dims=(16, 4096), dtype=mi.float16)
    W = graph1.new_input(dims=(4096, 4096), dtype=mi.float16)
    D = graph1.rms_norm(X, normalized_shape=(4096,))
    O = graph1.matmul(D, W)
    graph1.mark_output(O)

    prob_optimized_graph = graph1.superoptimize(config="mlp", is_formal_verified=False)
    prob_optimized_graph.visualize("prob_optimized_rms_norm")

    formal_optimized_graph = graph1.superoptimize(config="mlp", is_formal_verified=True)
    formal_optimized_graph.visualize("formal_optimized_rms_norm")
    
    # input_tensors = [
    #     torch.randn(16, 4096, dtype=torch.float16, device='cuda:0'),
    #     torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0'),
    # ]

    # original_output = graph(inputs=input_tensors)
    # formal_opt_output = formal_optimized_graph(inputs=input_tensors)
    # prob_opt_output = prob_optimized_graph(inputs=input_tensors)

    # def compare_outputs(name, output1, output2, atol=1e-3, rtol=1e-3):
    #     if torch.allclose(output1[0], output2[0], atol=atol, rtol=rtol):
    #         print(f"{name} matches the original graph ✔️")
    #     else:
    #         print(f"{name} does not match the original graph ❌")
    #         print(f"Original output: {output1[0]}")
    #         print(f"Optimized output: {output2[0]}")

    # compare_outputs("Formal Optimized Graph vs Original Graph", formal_opt_output, original_output)
    # compare_outputs("Probabilistic Optimized Graph vs Original Graph", prob_opt_output, original_output)


    