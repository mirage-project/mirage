import mirage as mi
import torch

def group_query_attention():
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(2, 256, 64), dtype=mi.float16)
    K = graph.new_input(dims=(2, 64, 4096), dtype=mi.float16)
    V = graph.new_input(dims=(2, 4096, 64), dtype=mi.float16)
    
    tbgraph1 = mi.new_threadblock_graph(grid_dim=(2,16,4), block_dim=(128,1,1), forloop_range=4, reduction_dimx=64)
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

def group_query_attention_ptq(): 
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(2, 256, 64), dtype=mi.float16)
    K = graph.new_input(dims=(2, 64, 1024), dtype=mi.float16)
    V = graph.new_input(dims=(2, 1024, 64), dtype=mi.float16)
    
    # ptq tbgraph1
    tbgraph1 = mi.new_threadblock_graph(grid_dim=(2,16,4), block_dim=(128,1,1), forloop_range=1, reduction_dimx=64)
    bQ = tbgraph1.new_input(dtensor=Q, input_map=(0, -1, 1), forloop_dim=-1)
    bK = tbgraph1.new_input(dtensor=K, input_map=(0, 2, -1), forloop_dim=-1) #2)
    bV = tbgraph1.new_input(dtensor=V, input_map=(0, 1, -1), forloop_dim=-1) # 1)
    ## mark bQ bK bA as quantized
    bA = tbgraph1.matmul(bQ, bK)
    bE = tbgraph1.exp(bA)
    ## mark bE bV bS as quantized
    bS = tbgraph1.matmul(bE, bV)
    bO1 = tbgraph1.forloop_accum(bS)
    bO2 = tbgraph1.forloop_accum(bE, "sum")
    bAq = tbgraph1.forloop_accum(bA)
    tbgraph1.new_output(stensor=bO1, output_map=(0, 2, 1))
    tbgraph1.new_output(stensor=bO2, output_map=(0, 2, 1))
    ## record bQ, bK, bA, bE, bV, bS for PTQ
    # bQq = tbgraph1.forloop_accum(bQ)
    # bKq = tbgraph1.forloop_accum(bK)
    # bEq = tbgraph1.forloop_accum(bE)
    # bVq = tbgraph1.forloop_accum(bV)
    # bSq = tbgraph1.forloop_accum(bS)
    
    # tbgraph1.new_output(stensor=bQq, output_map=(0, -1, 1), forloop_dim=-1) # forloop dim comes from bQ
    # tbgraph1.new_output(stensor=bKq, output_map=(0, 2, -1), forloop_dim=2) # forloop dim comes from bK
    tbgraph1.new_output(stensor=bAq, output_map=(0, 2, 1))
    # tbgraph1.new_output(stensor=bEq, output_map=(0, 2, 1),  forloop_dim=2) # forloop dim comes from bK
    # tbgraph1.new_output(stensor=bVq, output_map=(0, 1, -1), forloop_dim=1) # forloop dim comes from bV
    # tbgraph1.new_output(stensor=bSq, output_map=(0, 2, -1),  forloop_dim=-1) # forloop dim comes from 
    
    O = graph.customized([Q, K, V], tbgraph1)
    ## record Q, K, A, E, V, S for PTQ
    # Qq = graph.mark_output(O[2])
    # Kq = graph.mark_output(O[3])
    Aq = graph.mark_output(O[2]) # O[4]
    # Eq = graph.mark_output(O[5])
    # Vq = graph.mark_output(O[6])
    # Sq = graph.mark_output(O[7])
    
    tbgraph2 = mi.new_threadblock_graph(grid_dim=(2,16,1), block_dim=(128,1,1), forloop_range=1, reduction_dimx=64)
    bA = tbgraph2.new_input(dtensor=O[0], input_map=(0, 1, -1), forloop_dim=-1)
    bB = tbgraph2.new_input(dtensor=O[1], input_map=(0, 1, -1), forloop_dim=-1)
    bA = tbgraph2.forloop_accum(bA, "sum_todimx")
    bB = tbgraph2.forloop_accum(bB, "sum")
    bO = tbgraph2.div(bA, bB)
    tbgraph2.new_output(stensor=bO, output_map=(0, 1, -1))
    O = graph.customized(O[:2], tbgraph2)
    
    graph.mark_output(O[0])
    graph.visualize("gqa_ptq")
    
    qrecords = [Aq] # Qq, Kq, Aq, Eq, Vq, Sq]
    return graph, qrecords

# Per-tensor scaling
def group_query_attention_quantized():
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(2, 256, 64), dtype=mi.float16)
    K = graph.new_input(dims=(2, 64, 4096), dtype=mi.float16)
    V = graph.new_input(dims=(2, 4096, 64), dtype=mi.float16)
    ## scale factor for input
    Qs = graph.new_input(dims=(1, 1, 1), dtype=mi.float16)
    Ks = graph.new_input(dims=(1, 1, 1), dtype=mi.float16)
    As = graph.new_input(dims=(1, 1, 1), dtype=mi.float16)
    Es = graph.new_input(dims=(1, 1, 1), dtype=mi.float16)
    Vs = graph.new_input(dims=(1, 1, 1), dtype=mi.float16)
    Ss = graph.new_input(dims=(1, 1, 1), dtype=mi.float16)
    
    tbgraph1 = mi.new_threadblock_graph(grid_dim=(2,16,4), block_dim=(128,1,1), forloop_range=4, reduction_dimx=64)
    bQ = tbgraph1.new_input(dtensor=Q, input_map=(0, -1, 1), forloop_dim=-1)
    bK = tbgraph1.new_input(dtensor=K, input_map=(0, 2, -1), forloop_dim=2)
    bV = tbgraph1.new_input(dtensor=V, input_map=(0, 1, -1), forloop_dim=1)
    ## forward scale factor into tbgraph1
    bQs = tbgraph1.new_input(dtensor=Qs, input_map=(-1, -1, -1), forloop_dim=-1)
    bKs = tbgraph1.new_input(dtensor=Ks, input_map=(-1, -1, -1), forloop_dim=-1)
    bAs = tbgraph1.new_input(dtensor=As, input_map=(-1, -1, -1), forloop_dim=-1)
    bEs = tbgraph1.new_input(dtensor=Es, input_map=(-1, -1, -1), forloop_dim=-1)
    bVs = tbgraph1.new_input(dtensor=Vs, input_map=(-1, -1, -1), forloop_dim=-1)
    bSs = tbgraph1.new_input(dtensor=Ss, input_map=(-1, -1, -1), forloop_dim=-1)
    
    ## quantize dtensor
    def quantize(tbgraph, X, xs):
        return tbgraph.div(X, xs)
    def dequantize(tbgraph, X, xs):
        return tbgraph.mul(X, xs)

    ############################
    bQq = quantize(tbgraph1, bQ, bQs)
    bKq = quantize(tbgraph1, bK, bKs)
    bAq = tbgraph1.matmul(bQq, bKq)
    bA = dequantize(tbgraph1, bAq, bAs)
    ############################
    bE = tbgraph1.exp(bA)
    ############################
    bEq = quantize(tbgraph1, bE, bEs)
    bVq = quantize(tbgraph1, bV, bVs)
    bSq = tbgraph1.matmul(bE, bV)
    bS = dequantize(tbgraph1, bSq, bSs)
    ############################
    bO1 = tbgraph1.forloop_accum(bS)
    bO2 = tbgraph1.forloop_accum(bE, "sum")
    tbgraph1.new_output(stensor=bO1, output_map=(0, 2, 1))
    tbgraph1.new_output(stensor=bO2, output_map=(0, 2, 1))
    ## take in scale factors
    O = graph.customized([Q, K, V, Qs, Ks, As, Es, Vs, Ss], tbgraph1)

    tbgraph2 = mi.new_threadblock_graph(grid_dim=(2,16,1), block_dim=(128,1,1), forloop_range=1, reduction_dimx=64)
    bA = tbgraph2.new_input(dtensor=O[0], input_map=(0, 1, -1), forloop_dim=-1)
    bB = tbgraph2.new_input(dtensor=O[1], input_map=(0, 1, -1), forloop_dim=-1)
    bA = tbgraph2.forloop_accum(bA, "sum_todimx")
    bB = tbgraph2.forloop_accum(bB, "sum")
    bO = tbgraph2.div(bA, bB)
    tbgraph2.new_output(stensor=bO, output_map=(0, 1, -1))
    O = graph.customized(O, tbgraph2)
    
    graph.mark_output(O[0])
    graph.visualize("gqa_quantized")
    return graph


if __name__ == "__main__": 
    mirage_dtype = mi.bfloat16
    torch_dtype = mi.convert_dtype_to_torch_type(mirage_dtype)
    
    input_tensors = [
        torch.rand(2, 256, 64 , dtype=torch.float16, device='cuda:0'),
        torch.rand(2, 64, 1024, dtype=torch.float16, device='cuda:0'),
        torch.rand(2, 1024, 64, dtype=torch.float16, device='cuda:0')
    ]
    # record actual computation at kernel level
    Qt = input_tensors[0]
    Kt = input_tensors[1]
    At = torch.matmul(Qt, Kt)
    print(f"At shape: {At.shape}")
    Et = torch.exp(At)
    Vt = input_tensors[2]
    St = torch.matmul(Et, Vt)
    
    # record tensor involved with matmul operator for PTQ
    graph_ptq, qrecords = group_query_attention_ptq()
    # input_strides = [tensor.stride() for tensor in input_tensors]
    # p = mi.generate_cuda_program(graph_ptq.cygraph, target_cc=86, input_strides=input_strides)
    # print(p["code"])
    
    outputs = graph_ptq(inputs=input_tensors, outputs=qrecords)
    # Q = outputs[0]
    # K = outputs[1]

    A = outputs[0]
    print(f"A shape: {A.shape}")
    # E = outputs[3]
    # V = outputs[4]
    # S = outputs[5]
    
    # check validity
    # assert torch.allclose(Qt, Q), "Q record seems wrong!"
    # assert torch.allclose(Kt, K), "K record seems wrong!"
    print(At)
    print(A)
    assert torch.allclose(At, A), "A record seems wrong!"
    # assert torch.allclose(Et, E), "E record seems wrong!"
    # assert torch.allclose(Vt, V), "V record seems wrong!"
    # assert torch.allclose(St, S), "S record seems wrong!"
    
    # graph               = group_query_attention()
    
    
    
    
    
    
    # graph_quantized     = group_query_attention_quantized()
