:tocdepth: 1
**********************************
Superoptimizing RMSNorm and Linear
**********************************

Introduction
============

This tutorial demonstates how to generate fast kernels that compute RMSNorm (https://arxiv.org/pdf/1910.07467) following by a linear layer, whose computation graph is shown as follows.

.. image:: /tutorials/images/rms_norm_linear_original.png
    :alt: RMSNorm and Linear
    :align: center

The forward pass for RMSNorm can be represented as follows.

.. math::
   y_i = \frac{ x_i * g_i }{ \sqrt{\frac{1}{n} \sum_{i=1}^{n}{x_i^2}} }

Existing ML systems generally launch RMSNorm and the following MatMul as two separate kernels, since RMSNorm includes reduction/broadcast, making it hard to directly fuse it with other computation. We can use Mirage to automatically generate a highly optimized kernel that fuses RMSNorm and MatMul:

.. code-block:: Python

    import mirage as mi
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(16, 4096), dtype=mi.float16)
    G = graph.new_input(dims=(1, 4096), dtype=mi.float16)
    W = graph.new_input(dims=(406, 4096), dtype=mi.float16)
    O = graph.matmul(graph.rms_norm(X, G), W)
    graph.mark_output(O)
    optimized_graph = mi.superoptimize(graph)

Mirage introduces uGraph, a multi-level graph structure to represent the computation of a GPU kernel. The uGraph of the discovered kernel is as follows. Instead of launching separate kernels, the Mirage-discovered kernel leverages the commutativity of the division in RMSNorm and the multiplication in MatMul, moving division after MatMul. This allows Mirage to load the input tensor X once to compute both the denominator of RMSNorm and MatMul. The optimized kernel is 1.5-1.7x faster than the original kernel.

.. image:: /tutorials/images/rms_norm_linear_ugraph.png
   :alt: uGraph for RMSNorm and Linear
   :align: center

.. image:: /tutorials/images/rms_norm_linear_performance.png
   :alt: Performance comparison with PyTorch
   :width: 400
   :align: center

The object `optimized_graph` can directly run as a function, and doing so will let Mirage transpile the uGraph into CUDA code, compile the code for execution, and launch the compiled kernel.

.. code-block:: Python

    import torch
    input_tensors = [
        torch.randn(16, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(1, 4096, dtype=torch.float16, device='cuda:0'),
        torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0')
    ]
    optimized_graph(input_tensors)
