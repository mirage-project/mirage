import mirage as mi
import os
import argparse
import torch

if __name__ == "__main__":
    graph = mi.new_kernel_graph()
    input0 = graph.new_input(dims=(12, 251), dtype=mi.float16)
    input1 = graph.new_input(dims=(12, 251), dtype=mi.float16)
    mid0 = graph.add(input0, input1)
    output0 = graph.add(input0, mid0)
    output1 = graph.exp(mid0)
    
    input_tensors = [
        torch.randn(12, 251, dtype=torch.float16, device='cuda:0'), 
        torch.randn(12, 251, dtype=torch.float16, device='cuda:0')
    ]
    
    outputs = graph(inputs=input_tensors, outputs=[output0, output1])
    
    # ========= Correctness check =========
    mid0_correct = input_tensors[0] + input_tensors[1]
    output0_correct = input_tensors[0] + mid0_correct
    output1_correct = torch.exp(mid0_correct)
    
    print('mirage output[0]:', outputs[0], sep='\n')
    print()
    print('correct output[0]:', output0_correct, sep='\n')
    
    # check the correctness of the output tensors
    print('Correctness of output[0]:', torch.allclose(outputs[0], output0_correct, atol=1e-3))
    print('Correctness of output[1]:', torch.allclose(outputs[1], output1_correct, atol=1e-3))
    
