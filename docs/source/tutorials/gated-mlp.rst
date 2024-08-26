:tocdepth: 1
*************************
Superoptimizing Gated MLP
*************************

Introduction
============

Gated MLP layers are currently used in many large language models (e.g., LLAMA-2, LLAMA-3, and their varients). The computation graph for gated MLP is shown as follows.

.. image:: /tutorials/images/gated_mlp_computation_graph.png
   :alt: Computation Graph for Gated MLP
   :align:center

.. code-block:: Python

    import torch
    import mirage as mi
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(8, 4096), dtype=mi.float16)
    W1 = graph.new_input(dims=(4096, 4096), dtype=mi.float16)
    W2 = graph.new_input(dims=(4096, 4096), dtype=mi.float16)
    D1 = graph.matmul(X, W1)
    D2 = graph.matmul(X, W2)
    O = graph.mul(graph.silu(D1, D2))
    graph.mark_output(O)
    optimized_graph = mi.superoptimize(graph)

    input_tensors = [
        torch.randn(8, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0')
    ]
    optimized_graph(input_tensors)

In the above code snippet, we first construct a kernel graph in Mirage that corresponds to the gated MLP computation and superoptimize the graph. The optimized graph returned by Mirage can be directly called as a function, which launches the optimized kernels for gated MLP.

