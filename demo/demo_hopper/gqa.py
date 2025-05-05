import mirage as mi
import torch

def group_query_attention():
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(2, 256, 64), dtype=mi.float16)
    K = graph.new_input(dims=(2, 64, 1024), dtype=mi.float16)
    V = graph.new_input(dims=(2, 1024, 64), dtype=mi.float16)
    
    tbgraph1 = mi.new_threadblock_graph(grid_dim=(2,16,4), block_dim=(256,1,1), forloop_range=4, reduction_dimx=64)
    bQ = tbgraph1.new_input(dtensor=Q, input_map=(0, -1, 1), forloop_dim=-1)
    bK = tbgraph1.new_input(dtensor=K, input_map=(0, 2, -1), forloop_dim=2)
    bV = tbgraph1.new_input(dtensor=V, input_map=(0, 1, -1), forloop_dim=1)
    bA = tbgraph1.matmul(bQ, bK)
    bE = tbgraph1.exp(bA)
    bS = tbgraph1.matmul(bE, bV)
    bO1 = tbgraph1.forloop_accum(bS)
    bO2 = tbgraph1.forloop_accum(bE, "sum")
    tbgraph1.new_output(stensor=bO1, output_map=(0, 2, 1))
    tbgraph1.new_output(stensor=bO2, output_map=(0, 2, 1))
    O = graph.customized([Q, K, V], tbgraph1)

    tbgraph2 = mi.new_threadblock_graph(grid_dim=(2,16,1), block_dim=(128,1,1), forloop_range=1, reduction_dimx=64)
    bA = tbgraph2.new_input(dtensor=O[0], input_map=(0, 1, -1), forloop_dim=-1)
    bB = tbgraph2.new_input(dtensor=O[1], input_map=(0, 1, -1), forloop_dim=-1)
    bA = tbgraph2.forloop_accum(bA, "sum_todimx")
    bB = tbgraph2.forloop_accum(bB, "sum")
    bO = tbgraph2.div(bA, bB)
    tbgraph2.new_output(stensor=bO, output_map=(0, 1, -1))
    O = graph.customized(O, tbgraph2)
    
    graph.mark_output(O[0])
    graph.visualize("gqa")
    return graph

def group_query_attention_blockwise_scaling():
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(2, 256, 64), dtype=mi.float16)
    K = graph.new_input(dims=(2, 64, 1024), dtype=mi.float16)
    V = graph.new_input(dims=(2, 1024, 64), dtype=mi.float16)
    
    # input scaling factors
    Qs = graph.new_input(dims=(2, 4, 1), dtype=mi.float16)
    Ks = graph.new_input(dims=(2, 1, 16), dtype=mi.float16)
    # Es = graph.new_input(dims=(2, 4, 16), dtype=mi.float16)
    Vs = graph.new_input(dims=(2, 16, 1), dtype=mi.float16)
    
    tbgraph1 = mi.new_threadblock_graph(grid_dim=(2,16,4), block_dim=(256,1,1), forloop_range=1, reduction_dimx=64)
    bQ = tbgraph1.new_input(dtensor=Q, input_map=(0, -1, 1), forloop_dim=-1)
    bK = tbgraph1.new_input(dtensor=K, input_map=(0, 2, -1), forloop_dim=-1)
    bV = tbgraph1.new_input(dtensor=V, input_map=(0, 1, -1), forloop_dim=-1)
    
    # input scaling factors in tiles (all of size 1)
    bQs = tbgraph1.new_input(dtensor=Qs, input_map=(0, -1, 1), forloop_dim=-1)
    bKs = tbgraph1.new_input(dtensor=Ks, input_map=(0, 2, -1), forloop_dim=-1)
    # bEs = tbgraph1.new_input(dtensor=Es, input_map=(0, 2, 1), forloop_dim=-1)
    bVs = tbgraph1.new_input(dtensor=Vs, input_map=(0, 1, -1), forloop_dim=-1)

    bA = tbgraph1.matmul_e4m3(bQ, bK, bQs, bKs)  
    bE = tbgraph1.exp(bA)

    bS = tbgraph1.matmul_e4m3(bE, bV, None, bVs) # bEs
    # bS = tbgraph1.matmul(bE, bV)
    bO1 = tbgraph1.forloop_accum(bS)
    bO2 = tbgraph1.forloop_accum(bE, "sum")
    tbgraph1.new_output(stensor=bO1, output_map=(0, 2, 1))
    tbgraph1.new_output(stensor=bO2, output_map=(0, 2, 1))
    O = graph.customized([Q, K, V, Qs, Ks, Vs], tbgraph1) # Es

    tbgraph2 = mi.new_threadblock_graph(grid_dim=(2,16,1), block_dim=(128,1,1), forloop_range=1, reduction_dimx=64)
    bA = tbgraph2.new_input(dtensor=O[0], input_map=(0, 1, -1), forloop_dim=-1)
    bB = tbgraph2.new_input(dtensor=O[1], input_map=(0, 1, -1), forloop_dim=-1)
    bA = tbgraph2.forloop_accum(bA, "sum_todimx")
    bB = tbgraph2.forloop_accum(bB, "sum")
    bO = tbgraph2.div(bA, bB)
    tbgraph2.new_output(stensor=bO, output_map=(0, 1, -1))
    O = graph.customized(O, tbgraph2)
    
    graph.mark_output(O[0])
    graph.visualize("gqa_blockwise_scale")
    return graph

if __name__ == "__main__": 
    mirage_dtype = mi.bfloat16
    torch_dtype = mi.convert_dtype_to_torch_type(mirage_dtype)
    
    input_tensors = [
        torch.rand(2, 256, 64 , dtype=torch.float16, device='cuda:0'),
        torch.rand(2, 64, 1024, dtype=torch.float16, device='cuda:0'),
        torch.rand(2, 1024, 64, dtype=torch.float16, device='cuda:0')
    ]
    
    input_scales = [
        torch.rand(2, 4, 1, dtype=torch.float16, device='cuda:0'),
        torch.rand(2, 1, 16,dtype=torch.float16, device='cuda:0'),
        # torch.rand(2, 4, 16,dtype=torch.float16, device='cuda:0'), # Es
        torch.rand(2, 16, 1,dtype=torch.float16, device='cuda:0'),
    ]
    
    # gqa = group_query_attention()
    # outputs = gqa(inputs=input_tensors)
    
    gqa_ptq = group_query_attention_blockwise_scaling()
    input_strides = [tensor.stride() for tensor in input_tensors + input_scales]
    p = mi.generate_cuda_program(gqa_ptq.cygraph, target_cc=86, input_strides=input_strides)
    print(p["code"])
    # run generated cuda program
    outputs = gqa_ptq(inputs=(input_tensors + input_scales))
