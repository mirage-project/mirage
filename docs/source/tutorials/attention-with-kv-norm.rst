:tocdepth: 1
***********************************************
Superoptimizing Attention with QK Normalization
***********************************************

Introduction
============

Recent work has introduced several variants to attention, include group-query attention, multi-latent attention, and group-query attention with query-key normalization (https://arxiv.org/abs/2405.09818). The following code snippet demonstrate the attention kernel used in Chameleon, a multi-modal foundation models for understanding and generating images and text in any arbitrary sequence. The key difference is that Chameleon performs normalizes query and key before computing attention.

.. code-block:: Python

    import torch
    import mirage as mi
    graph = mi.new_kernel_graph()
    Q = graph.new_input(dims=(2, 256, 64), dtype=mi.float16)
    K = graph.new_input(dims=(2, 64, 4096), dtype=mi.float16)
    V = graph.new_input(dims=(2, 4096, 64), dtype=mi.float16)
    Q = graph.rms_norm(Q)
    K = graph.rms_norm(K)
    A = graph.matmul(Q, K)
    E = graph.exp(A)
    S = graph.reduction(E, 2)
    D = graph.div(E, S)
    O = graph.matmul(D, V)
    graph.mark_output(O)
    optimized_graph = mi.superoptimize(graph)

