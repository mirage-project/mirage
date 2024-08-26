:tocdepth: 1
**********************************
Superoptimizing RMSNorm and Linear
**********************************

Introduction
============

This tutorial demonstates how to superoptimize RMSNorm (https://arxiv.org/pdf/1910.07467) following by a linear layer. Instead of launching separate kernels for these two layers, Mirage is able to automatically generate a customized kernel that fuses the computation of RMSNorm and the following matrix multiplication. The uGraph of the customized kernel is shown as follows.

.. code-block:: Python
    import mirage as mi
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(8, 4096), dtype=mi.float16)
    G = graph.new_input(dims=(4096), dtype=mi.float16)
    W = graph.new_input(dims=(406, 4096), dtype=mi.float16)
    O = graph.matmul(graph.rms_norm(X, G), W)
    graph.mark_output(O)
    optimized_graph = mi.superoptimize(graph)


As we have demonstrated in other tutorials, optimized ugraphs can be directly executed as a fuction, and doing so will automatically launch the customized kernel discovered by Mirage.

.. code-block:: Python
    import torch
    input_tensors = [
        torch.randn(8, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0')
    ]
    optimized_graph(input_tensors)
