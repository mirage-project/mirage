import mirage as mi
import numpy as np
import torch

silu = torch.nn.SiLU()

def torch_gated_mlp(X, W1, W2):
    D = torch.matmul(X, W1)
    E = torch.matmul(X, W2)
    O = torch.mul(silu(D), E)
    return O

if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(16, 4096), dtype=mi.float16)
    W1 = graph.new_input(dims=(4096, 4096), dtype=mi.float16)
    W2 = graph.new_input(dims=(4096, 4096), dtype=mi.float16)
    
    tb_graph = mi.new_threadblock_graph(grid_dim=(64,1,1), block_dim=(128,1,1), forloop_range=32, reduction_dimx=64)
    tX = tb_graph.new_input(dtensor=X, input_map=(-1, -1, -1), forloop_dim=1)
    tW1 = tb_graph.new_input(dtensor=W1, input_map=(1, -1, -1), forloop_dim=0)
    tW2 = tb_graph.new_input(dtensor=W2, input_map=(1, -1, -1), forloop_dim=0)
    
    tD1 = tb_graph.matmul(tX, tW1)
    tD2 = tb_graph.matmul(tX, tW2)
    tA1 = tb_graph.forloop_accum(tD1)
    tA2 = tb_graph.forloop_accum(tD2)
    tS = tb_graph.silu(tA1)
    tO = tb_graph.mul(tS, tA2)
    
    tb_graph.new_output(stensor=tO, output_map=(1, -1, -1))
    O = graph.customized([X, W1, W2], tb_graph)
    graph.mark_output(O[0])

    graph1 = mi.new_kernel_graph()
    X = graph1.new_input(dims=(16, 4096), dtype=mi.float16)
    W1 = graph1.new_input(dims=(4096, 4096), dtype=mi.float16)
    W2 = graph1.new_input(dims=(4096, 4096), dtype=mi.float16)
    O1 = graph1.matmul(X, W1)
    O2 = graph1.matmul(X, W2)
    O1 = graph1.silu(O1)
    O = graph1.mul(O1, O2)
    graph1.mark_output(O)

    prob_optimized_graph = graph1.superoptimize(config="mlp")
    formal_optimized_graph = graph1.superoptimize(config="mlp", is_formal_verified=True)

    input_tensors = [
        torch.randn(16, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0')
    ]

    original_output = graph(inputs=input_tensors)
    prob_opt_output = prob_optimized_graph(inputs=input_tensors)
    formal_opt_output = formal_optimized_graph(inputs=input_tensors)

    # def compare_outputs(name, output1, output2, atol=1e-2, rtol=1e-2):
    #     if torch.allclose(output1[0], output2[0], atol=atol, rtol=rtol):
    #         print(f"{name} matches the original graph ✔️")
    #     else:
    #         print(f"{name} does not match the original graph ❌")
    
    # compare_outputs("Probabilistic Optimized Graph vs Original Graph", prob_opt_output, original_output)
    # compare_outputs("Formal Optimized Graph vs Original Graph", formal_opt_output, original_output)

    graph.visualize("gated_mlp")
    prob_optimized_graph.visualize("prob_optimized_gated_mlp")
    formal_optimized_graph.visualize("formal_optimized_gated_mlp")