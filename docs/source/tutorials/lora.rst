:tocdepth: 1
***********************
Superoptimizing Low-Rank Adaptation
***********************

Introduction
============

Low-rank adaption (LoRA) has been widely used to adapt a pre-trained model to specialized domains and tasks. These LoRA adapters are generally inserted into the linear layers of a model, leveraging the low-rank decomposition concept to construct trainable parameters inserted into the original model weights.

.. image:: /tutorials/images/lora_kernel_graph.png
   :alt: Computation Graph for LoRA
   :align: center


The following code snippet demonstrates superoptimizing LoRA adapters in Mirage.

.. code-block:: Python

   import torch
   graph = mi.new_graph()
   X = graph.new_input(dims=(16, 256), dtype=mi.float16)
   W = graph.new_input(dims=(256, 4096), dtype=mi.float16)
   A = graph.new_input(dims=(256, 16), dtype=mi.float16)
   B = graph.new_input(dims=(16, 4096), dtype=mi.float16)
   D = graph.matmul(X, A)
   E = graph.matmul(D, B)
   C = graph.matmul(X, W)
   O = graph.add(C, E)



