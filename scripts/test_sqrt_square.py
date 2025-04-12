import torch
from mirage import new_kernel_graph, float16

def test_operators():
    graph = new_kernel_graph()
    A = graph.new_input((6, 6), dtype=float16)
    
    C_sqrt = graph.sqrt(A)
    C_square = graph.square(A)
    
    graph.mark_output(C_sqrt)
    graph.mark_output(C_square)
    
    input_A = torch.randn(6, 6, dtype=torch.float16, device='cuda')
    
    outputs = graph(inputs=[input_A])
    
    print("\nDebugging sqrt and square:")
    print("Input A:")
    print(input_A)
    print("\nMirage sqrt output:")
    print(outputs[0])
    print("\nPyTorch sqrt output:")
    print(torch.sqrt(input_A))
    print("\nMirage square output:")
    print(outputs[1])
    print("\nPyTorch square output:")
    print(torch.square(input_A))
    
    print("\nComparing with PyTorch implementations:")
    print("sqrt:", torch.allclose(outputs[0], torch.sqrt(input_A), rtol=1e-3))
    print("square:", torch.allclose(outputs[1], torch.square(input_A), rtol=1e-3))

if __name__ == "__main__":
    test_operators() 