:tocdepth: 1
**********************************
Superoptimizing RMSNorm and Linear
**********************************

Introduction
============

This tutorial demonstates how to generate fast kernels that compute RMSNorm (https://arxiv.org/pdf/1910.07467) following by a linear layer. The forward pass for RMSNorm can be represented as follows.

.. math::
   y_i = \frac{ x_i * g_i }{ \sqrt{\frac{1}{n} \sum_{i=1}^{n}{x_i^2}} }

Instead of launching separate kernels for these two layers, Mirage is able to automatically generate a customized kernel that fuses the computation of RMSNorm and the following matrix multiplication. The following code snippet shows how to use Mirage to automatically generate an optimized kernel for RMSNorm + Linear.

.. code-block:: Python

    import mirage as mi
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(16, 4096), dtype=mi.float16)
    G = graph.new_input(dims=(1, 4096), dtype=mi.float16)
    W = graph.new_input(dims=(406, 4096), dtype=mi.float16)
    O = graph.matmul(graph.rms_norm(X, G), W)
    graph.mark_output(O)
    optimized_graph = mi.superoptimize(graph)

The uGraph of the optimized kernel is shown as follows.

.. image:: /tutorials/images/rms_norm_linear_ugraph.png
   :alt: uGraph for RMSNorm and Linear
   :align: center

The object `optimized_graph` can run as a function, and doing so will let Mirage transpile the uGraph into CUDA code, compile the code for execution, and launch the compiled kernel.

.. code-block:: Python

    import torch
    input_tensors = [
        torch.randn(16, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(1, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0')
    ]
    optimized_graph(input_tensors)


.. image:: /tutorials/images/rms_norm_linear_performance.png
   :alt: Performance comparison with PyTorch
   :align: center
