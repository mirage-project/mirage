:tocdepth: 1
********************************
Multi-Level Graph Representation
********************************

Mirage uses a uGraph to specify the execution of a tensor program on GPUs. A uGraph contains hierarchical graphs at multiple levels to represent computation at the kernel, block, and thread levels.

GPU Hierarchy
=============

.. image:: /images/gpu_hierarchy.png
   :alt: GPU compute and memory hierarchy
   :align: center
   :width: 600

The above figure shows the hierarchy of today’s GPUs. Computation on GPUs is organized as kernels, each of which is a function executed simultaneously on multiple GPU cores in a single-program-multiple-data (SPMD) fashion. A kernel includes a grid of thread blocks, each of which is executed on one GPU streaming multiprocessor and includes multiple threads to perform computation on individual data elements. Each thread is associated with a per-thread register file, and all threads within a thread block can access shared memory to enable collective operations. Finally, all inputs and outputs of a kernel are stored in GPU device memory.


uGraph Representation
=====================

The following graph shows a uGraph for computing group-query attention. We use it as the running example to explain the key components of a uGraph.

.. image:: /images/mugraph_gqa.png
   :alt: uGraph for group-query attention
   :align: center

Kernel Graph
------------

Each tensor program corresponds to one kernel graph, where each node represents a kernel running on an entire GPU and each edge is a tensor shared between kernels. All tensors in a kernel graph are stored in GPU device memory since different kernels cannot share data in register file or shared memory. Each node in a kernel graph can be a pre-defined kernel operator supported by existing kernel libraries such as convolution by cuDNN and matrix multiplication by cuBLAS. In addition, to enable fine-grained inter-kernel optimizations such as kernel fusion, a node in a kernel graph can also be a graph-defined kernel operator, whose semantic and behavior are defined by a lower-level (i.e., block) graph. As an example, both kernel operators in the above figure are graph-defined operators, each of which is specified by a block graph.

Thread Block Graph
------------------

A block graph specifies computation associated with a thread block, where each node denotes a block operator which specifies computation within a thread block, and each edge is a tensor shared between thread block operators. Mirage saves all intermediate tensors within a block graph in GPU shared memory for two considerations. First, GPU shared memory offers much higher bandwidth than device memory, and this design allows Mirage to reduce device memory access by maximally saving intermediate results in shared memory. Second, for tensors whose sizes exceed the shared memory capacity and must be stored in the device memory, Mirage uses these tensors to split computation into multiple block graphs, each of which only contains tensors in shared memory. This separation does not introduce additional access to device memory.

Each block graph is also associated with a few properties to specify its execution, which we introduce as follows.

Grid Dimensions
---------------

All thread blocks within a kernel are organized by a mesh with up to 3 dimensions, identified as x, y, and z. Correspondingly, a block graph is associated with up to three grid dimensions that specify the number of blocks along the x, y, and z dimensions. The two block graphs in
the above figure launch 80 (i.e., 8 × 10) and 64 (i.e., 8 × 8) blocks.

First, for each input tensor to a graph-defined kernel operator (e.g., Q, K, and V in the kernel graph), the associated block graph contains an imap, which specifies how the input tensor is partitioned into sub-tensors for individual blocks. For each grid dimension (i.e., x, y, or z), the imap maps it to (1) a data dimension of the input tensor or (2) a special replica dimension 𝜙. For (1), the mapped data dimension is equally partitioned across blocks along the grid dimension. For (2), the input tensor is replicated across these thread blocks.

Second, for each output tensor of a block graph, the block graph includes an omap, which specifies how the outputs of all blocks are concatenated to construct the final output of the kernel operator. In an omap, each grid dimension must map to a data dimension of the output tensor, since different blocks must save to disjoint tensors in device memory. For B of shape [h=1, s=8, d=64] in the above figure, its omap={x<->h, y<->d} indicates that blocks with the same x index are concatenated along the h dimension and that blocks with the same y index are concatenated along the d dimension, resulting in a tensor B of shape [h=8, s=8, d=640].

For-loop Dimensions
-------------------

To fit large input tensors in shared memory and allow cache reuse, a second property associated with each block graph is for-loop dimensions, which collectively specify how many times the block graph is executed to complete a kernel. Correspondingly, each input tensor to a block graph is first sent to an input iterator that loads a part of the tensor from device memory to shared memory. Each input iterator is associated with an fmap to specify which part of the input tensor to load in each iteration. Formally, the fmap maps each for-loop dimension to (1) a data dimension of the input tensor or (2) the replica dimension 𝜙. Similar to the semantic of imap, the input tensor is equally partitioned along that dimension for (1) and replicated for (2).

In addition, a block graph contains output accumulators to accumulate its output across iterations in shared memory and save the final results back to device memory. Similar to an input iterator, an output accumulator is also associated with an fmap to specify how the output tensors of different iterations are combined to produce the final results. Specifically, the fmap maps each for-loop dimension to either a data dimension, which results in concatenation of the output along that dimension, or the replica dimension 𝜙, which results in the output being accumulated in shared memory.

Thread Graph
------------

A thread graph further reduces the scope of computation from a block to a single thread. Similar to a block graph, each thread graph is also associated with block dimensions, which specify the organization of threads within the block, and for-loop dimensions, which define the total number of iterations to finish the defined computation. Each thread graph includes input iterators, each of which loads an input tensor from GPU shared memory to register file, and output accumulators, each of which saves an output tensor from register file back to shared memory. A thread graph is the lowest level graph in a uGraph and only contains pre-defined thread operators.

