The CUDA Transpiler
===================

Introduction to Mirage’s CUDA Transpiler
-----------------------------------

The CUDA transpiler is a key component of Mirage. It is responsible for
translating the (optimized) computation graph produced by Mirage into
efficient CUDA code.

It is not easy to write a transpiler. You should ensure both the
correctness and the performance, which is both an algorithm challenge
(e.g. how to plan the layout of every tensor) and an engineering
challenge (e.g. how to design the transpiler in an extensible and
maintainable way). There are also numerous details and corner cases to
consider, like what to do when the shape of the matrix is not divisible
by the MMA size.

In this document, we will first introduce the architecture of the
transpiler, then dive into the details of some key algorithms and
components, and finally discuss some problems and solutions on corner
cases.

Architecture
------------

The transpiler has two crucial components: the “Transpiler” and the
“Runtime”.

The “Transpiler” is a part of Mirage’s codebase (under
``src/transpiler``). It exposes an interface called ``transpile``, which
takes a Mirage’s computational graph as input and returns the generated
CUDA code. Here is an example of a piece of generated code:

.. code-block:: cpp

    #define NUM_GPUS 1
    #define USE_NVSHMEM 0
    #include "runtime.h"
    using namespace cute;

    static void _init() {
    }

    static void _execute_mugraph(std::vector<void const *> input_tensors, std::vector<void*> output_tensors, void* buf) {
      {
        // OP type: kn_matmul_op
        half_t *dtensor10000000 = (half_t*)input_tensors.at(0);
        half_t *dtensor10000001 = (half_t*)input_tensors.at(1);
        half_t *dtensor10000002 = (half_t*)output_tensors.at(0);
        kn::gemm<CUBLAS_COMPUTE_16F>(dtensor10000002,dtensor10000000,dtensor10000001, 6,4,9, 16,1, 8,1, 1,8, 3, 128,128,64);
      }
    }

The “Runtime” is a header-only library that provides a set of useful
kernels & device functions for the transpiled code. Although it’s
located at ``include/mirage/transpiler/runtime``, code in Mirage does
not include it. Instead, the generated code will include the “Runtime”
library (``runtime.h``), and can call various functions inside it (like
we called ``kn::gemm`` in the example above). It’s the joint effort of
the “Transpiler” and the “Runtime” that makes the generated code
efficient and correct.

Besides, the transpiler also contains a set of tests to ensure the
correctness and performance of the generated code. It’s located under
``tests/transpiler``.

Prerequisites
-------------

Before we dive into the details of the transpiler, we need some
prerequisites: - You need to master **CUDA Programming**, of course. -
To leverage compile-time computation, we use **C++ Template
Metaprogramming** extensively. - We also make use of the **CuTe
Library**. It’s a library inside the `Cutlass
library <https://github.com/NVIDIA/cutlass>`__ and provides a set of
useful primitives for tensor core matrix multiplication and data
movement. CuTe has its own
`document <https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md>`__
but it’s poorly written. If you speak Chinese (or you are good at Google
Translate), you can refer to `these series of
blogs <https://zhuanlan.zhihu.com/p/661182311>`__ for a better
understanding. - We suppose you have a basic understanding of **the
Mirage project**, particularly, what ``DTensor``, ``STensor``, “kernel
graph”, “kernel operator”, “threadblock graph”, “threadblock operator”
mean.

Transpiler Overview
-------------------

The Transpiler can be divided into several steps:

1. **Threadblock Level Data Reuse Logic (Fusion Logic).** In threadblock
   level code, sometimes we can avoid reading and writing data by fusing
   the next kernel with the current one. An example is that if a matmul
   is followed by an elementwise exponentiation, we can directly perform
   the elementwise exponentiation on the result of the matmul, instead
   of writing the matmul result back to shared memory, and then reading
   it again. We called this threadblock level data reuse. The
   Transpiling process begins with planning which operators to fuse
   since it has an impact on nearly everything (layout, scheduing,
   etc.).
2. **Layout Resolution.** Plan the layout (strides of every dimension)
   of every ``DTensor`` and ``STensor``.
3. **Kernel-Level Memory Planning.** Plan the memory for every
   ``DTensor``.
4. **Kernel-Level Transpile.** Transpile each kernel operator one by
   one, that is:

-  If the operator is a pre-defined kernel operator (say, a Matmul), we
   call the corresponding function in the Runtime.
-  If the operator is a custom operator (``KN_CUSTOMIZED_OP``), we call
   the function ``transpile_kn_custom_op`` which transpiles the custom
   operator into a CUDA kernel (a function marked with ``__global__``).
   This includes:

   1. **TB Graph Scheduling.** Choose the order of operators to perform,
      which we called “TB Sched”. From another perspective, this step
      translates the threadblock-level graph (which can be seen as a
      graph IR) into a sequence of operators (similar to a linear IR).
   2. **Swizzle Planning.** Plan how to swizzle the layout of every
      ``STensor`` to avoid bank conflicts.
   3. **Memory Planning.** Allocate memory for every ``STensor``.
   4. **Threadblock-Level Transpilation.** Transpile the threadblock
      level code based on the TB graph scheduling and memory planning.

Algorithms
----------

Threadblock Level Data Reuse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In threadblock level code, an operator (which will be transpiled to a
device function) reads data from shared memory, performs computation,
and writes the result back to shared memory. However, sometimes we can
avoid this reading-writing by “fusing” the next kernel (in this section,
“kernel” refers to threadblock level operators) with the current one.
For example, if a matmul is followed by an elementwise exponentiation,
we can directly perform the elementwise exponentiation on the result of
the matmul, instead of writing the matmul result back to shared memory,
and then reading it again. We called this threadblock level data reuse.

Achieving threadblock level data reuse is not easy. If the latter kernel
is not a simple elementwise operation, you need to align the data layout
(for every CUDA thread, which part of data does it hold?) between the
former kernel, which is really complex. Besides, if the latter kernel is
not a unary (only has one input tensor) kernel, more problems will
arise. For example, Assume you have two threadblock level ops, A and B,
and an elementwise addition operator which takes the output of A and B
as input, and assume that you want to perform A first, then you need to
choose between storing the output of A in shared memory or registers.
That’s a complex decision.

Currently, the Transpiler only considers fusion where the latter
operator is an elementwise unary operator (e.g. exp or accumulate). This
problem is much easier since we can always fuse an elementwise unary
operator with its former operator without any concerns. We may consider
more complex scenarios in the future.

In the Runtime, the operator fusion is implemented as “epilogues”. An
epilogue is a chain of actions performed on a single element. Every node
on the chain can be: - A unary operator, like ``exp`` - An action that
involves memory operatorion, like “store”
(``dst[dst_layout(dst_index)] = x``) or “store-and-accumulate”
(``dst[dst_layout(dst_index)] += x``). Every chain is terminated by such
an action.

During fusion, we are actually “chaining” operators, which means that we
are dividing the original threadblock-level graph into several chains.
Every chain contains a “leading” operator, and a series (possibly zero)
elementwise-unary operators. Here is an illustration:

.. figure:: /images/tb-fusion-chain.drawio.svg
   :alt: tb-fusion-chain-example

   tb-fusion-chain-example

Layout Resolution
~~~~~~~~~~~~~~~~~

A layout is a mapping between logical coordinates and physical memory
addresses. For example, for a 2D tensor, the layout may be row-major or
column-major.

The layout is crucial for the performance of the generated code. For
example, if the layouts of both tensors are row-major during a G->S
(global memory to shared memory) copy, we can copy data in larger chunks
(say, 128 bits) instead of copying element by element. Besides,
sometimes we may swizzle the layout of STensors to avoid bank conflicts.

To resolve layouts for all tensors, we first constructs an boolean ILP
problem to decide the “innermost dimension”, the dimension with a stride
of 1, of every ``DTensor`` and ``STensor``. For every dimension of every
tensor, we have a boolean variable indicating whether it’s the innermost
dimension. After that, we add a set of restrictions (stands for
restrictions proposed by kernels in the Runtime, like cuBLAS requires
the innermost dim to be among the last two dimensions), and build the
optimization target (one operator under different layouts may have
different performance, and we want to minimize the total “cost”). After
that, we use the Z3 solver to find the optimal solution.

In the equation mentioned above, for each dimension in every
``STensor``, we also have another boolean variable indicating whether
this dimension is “swizzled”, or in other words, this dimension is not
the innermost dimension but threads may access data along this
dimension. Under this scenario, we can swizzle the layout of this
dimension to avoid bank conflicts. For more information about swizzling,
please refer to the “How to Swizzle” section in “Problems and
Solutions”.

After deciding the innermost dimension, it’s time to calculate the
strides. The stride of one dimension is the number of elements between
two consecutive elements in this dimension. For example, for a row-major
2D tensor with shape :math:`[m, n]`, it has a stride of :math:`[n, 1]`.
The physical address of an element is the dot product between its
coordinates and the stride, while in our example, an element with
coordinates :math:`(i, j)` has a physical address of
:math:`i \times n + j`.

Here we use a heuristic to calculate the strides, implemented in the
function ``calc_tensor_strides``. Assume the innermost dim of one tensor
is dimension #k, and the number of dimensions is N. We first decide an
“order” of all dimensions, which is
:math:`[k, N-1, N-2, \dots, k+1, k-1, \dots, 1, 0]`, and then assign
strides based on that order (see ``calc_tensor_strides`` for details).
We also pad the first non-1 dimension to a multiple of 8 (since there
are 8 halfs in 16 Bytes) to ensure the address of the starting element
of every dimension (with coords looks like
:math:`(0, 0, \dots, 0, M, 0, \dots, 0)`) is aligned to 16 Bytes
(comment: This doesn’t hold if shift-based swizzling is used later).

TB Graph Scheduling and Memory Planning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The TB graph scheduling problem is that, given a threadblock-level
graph, how to get an optimal “schedule” to maximize the performance?

The schedule has an impact on the performance in several aspects:

-  Minimize the number of synchronizations (``__syncthreads()``) between
   threadblocks
-  Minimize the peak shared memory usage (since different schedules
   result in different tensor lifetimes, which may affect the peak
   shared memory usage)

Those objectives are sometimes conflicting, and we need to somehow find
a balance between them. For example, consider the following
computational graph:

.. figure:: /images/tb-sched-conflict-example.drawio.svg
   :alt: tb-sched-conflict-example

   tb-sched-conflict-example

And there are two possible schedulings:

-  ``12 534 6 7``: 3 synchronizations (a space denotes a
   synchronization) with peak mem usage = 4
-  ``1234 56 7``: 2 synchronizations with peak mem usage = 5

You can see that the former schedule results in lower peak mem usage but
more synchronizations, while the latter schedule results in fewer
synchronizations but higher peak mem usage. Which one is better? It’s
hard to say.

Currently our heuristic is that, we always prioritize the number of
synchronizations, and then the peak mem usage. So we first find an order
that minimizes the number of synchronizations (if multiple orders have
the same number of synchronizations, we choose a random one), and then
try to minimize the peak mem usage under this order.

TB Graph Scheduling
~~~~~~~~~~~~~~~~~~~

To find the order with the minimum number of synchronizations, we use a
modified topology sort algorithm. We label each node (threadblock
operator) with a “depth”, which is length of the longest path from any
input operator to this node. We can calculate this depth by a dynamic
programming (DP) algorithm:

-  For input nodes, its depth is 0
-  If an operator is fused with a previous operator, it has the same
   depth as the previous one
-  Otherwise, its depth is :math:`\max_{v \in I} depth[v] + 1`, where
   :math:`I` is the set of direct input nodes of this node

After that, we sort the nodes by their depth in ascending order, and
perform a synchronization when the depth of the latter node is greater
than the former node. This is how we obtain the order of operators.

In this step, we also generate relative metadata for every operator. For
example, for input operators, we decide whether or not to use chunked
input (copying in 128 bits) and/or software pipelining. This is achieved
by the following steps:

-  First we generate metadata for every operator independently
-  Some metadata may depend on the other operators on the same chain.
   For example, we only put the accumulator into register files (instead
   of shared memory) if the “leading operator” of the chain is a matmul
   op. To deal with this case, we furthermore “refine” the metadata on
   the chain when chaining operators together. This is implemented in
   the function ``refine_opmeta_on_chain``.

Finally, we end with a linear order of operators, with metadata
attached.

From another perspective, this step effectively translates the
threadblock-level graph (which can be seen as a graph IR) into another
linear IR for further processing.

Decide How to Swizzle
~~~~~~~~~~~~~~~~~~~~~

Sometimes we need to swizzle the layout of an ``STensor`` in order to
avoid bank conflict. For example, when loading a 8x8 or 16x16 submatrix
using the ``ldmatrix`` instruction, different threads may request data
from the same bank if no swizzling is applied, leading to performance
degration.

Generally speaking, there are two methods to swizzle a layout: “xor
method” and “shift method”.

The idea of the xor method is to calculate the bitwise XOR between the
row number and the original address, and use that as the new address,
i.e. :math:`new\_addr = old\_addr \oplus row`. That’s the one used in
``cute::Swizzle``.

Let’s take an example. Assume we have a :math:`8 \times 8` tensor with
row-major layout. We also have 8 banks (physical address :math:`i` will
be mapped to bank :math:`i \mod 8`). Here is an illustration of the
original layout and the swizzled layout:

.. figure:: /images/swizzle-xor-example.drawio.svg
   :alt: swizzle-xor-example

   swizzle-xor-example

This swizzling method requires no memory overhead, but it can only be
used when the number of columns is a power of 2. For logic deciding the
swizzling parameters (:math:`B`, :math:`M`, and :math:`S`), please refer
to code in ``plan_tb_swizzle.cc``.

Another method is the shift method. The idea is to calculate the new
address by :math:`new\_addr = old\_addr + row \times shift`, where
:math:`shift` is a constant chosen by us. Intuitively, it looks like to
“enlarge” the stride of the row to “shift” banks. This method can be
used no matter how many columns the tensor has, but it requires a memory
overhead of :math:`shift \times num\_rows`. Here is an example:

.. figure:: /images/swizzle-shift-example.drawio.svg
   :alt: swizzle-shift-example

   swizzle-shift-example

According to number theory, we can totally avoid bank conflict if the
greatest common divisor (GCD) between the new stride (original stride +
shift) and the number of banks is :math:`1`. Since the number of banks
is usually a power of 2, we can always find a shift :math:`\in \{0, 1\}`
that satisfies this requirement, so the memory overhead is quite small.

And then, let’s talk about how we incorporate these swizzling methods
into the Transpiler.

1. First, some instructions require every :math:`chunk` element to be
   contiguous and in-order, e.g. when performing ``ldmatrix``
   instruction or chunked copying, every 8 halfs should be consecutive
   in memory. We calculate this “chunk size” after TB graph scheduling
   since the metadata of every operator is ready at that time.

2. After that, for every STensor, we decide how to swizzle the layout.
   If the number of chunks in the innermost dimension, we use the xor
   method. Otherwise, we use the shift method.

Memory Planning
~~~~~~~~~~~~~~~

Now we have decided the schedule (i.e. the order of operators). Now it’s
time to allocate memory for every STensor.

Let’s first introduce the Dynamic Storage Allocation (DSA) problem. The
DSA problem is that, given a list of objects (each object has a size, a
allocate time, and a free time), how to allocate memory (i.e. provide a
start address for every object) to minimize the peak memory usage?
Formally speaking, a DSA input :math:`I` consists of :math:`n` triples
of numbers, i.e.
:math:`I = \{(s_1, l_1, r_1), \dots, (s_n, l_n, r_n)\}`, where
:math:`s_i` is the size of the :math:`i`-th object, and
:math:`[l_i, r_i)` is the time interval that the :math:`i`-th object is
alive. The output is a list of :math:`n` integers, i.e.
:math:`O = \{a_1, \dots, a_n\}`, where :math:`a_i` is the start address
of the :math:`i`-th object, such that if
:math:`[l_i, r_i) \cap [l_j, r_j) \neq \phi\ (i \neq j)`, then
:math:`[a_i, a_i+s_i) \cap [a_j, a_j+s_j)` should be :math:`\phi`. This
problem also have a nice geometric interpretation: You can view each
triple :math:`(s_i, l_i, r_i)` asan axis-parallel rectangle with a
projection :math:`(l_i, r_i)` on the x-axis and a height of :math:`s_i`.
You are only allowed to slide the rectangles along the y-axis. The
objective is to pack all rectangles into a minimum height without
overlapping.

Firstly, we show how we formulate the memory planning problem as DSA
problem. For the size of every STensor, we can easily calculate it when
doing layout resolution. For the time interval (lifetime), we carefully
catorgorize STensors into the following types, and calculate the
lifetime for every STensor:

.. figure:: /images/tensor-lifecycle.drawio.svg
   :alt: tensor-lifecycle

   tensor-lifecycle

Then we can try to solve the DSA problem. Unfortunately, it’s actually a
NP-Complete problem. We just run some heuristics (first fit, best fit,
last fit, random, etc.) and choose the one with the minimum peak shared
memory usage.

Problems and Solutions
----------------------

In this section, we will discuss some problems and solutions on corner
cases.

How to Perform Threadblock Level Matrix Multiplication when the Size is not Divisible by MMA Size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prerequisites: - The ``ldmatrix`` instruction - MMA and Copy in CuTe

Life will be much easier if the size of the matrices is divisible by the
MMA size. In such scenarios, we can just copy the data from shared
memory to registers via ``ldmatrix`` (assume SM75+) and perform the
matrix multiplication.

However, the story is different when the size of the matrices is not
divisible by the MMA size. Since ``ldmatrix`` loads a 16x16 tile, it may
access out-of-bound memory, or produce incorrect results.

As the figure below shows, the black grids are 16x16 tiles, and the
green rectangle is the matrix that is not divisible by the MMA size.
Areas marked with blue are out-of-bound elements. When performing
``ldmatrix`` on those areas, we must find a valid memory address to feed
the ``ldmatrix`` instruction, instead of using the out-of-bound address.
We do not care about their value. Areas marked with red are out-of-bound
elements that may affect the final result. In addition to finding a
valid memory address, we must also ensure that those elements are zero.

.. figure:: /images/mma-non-divisible-example.drawio.svg
   :alt: mma-non-divisible-example

   mma-non-divisible-example

Let’s recap what ``ldmatrix`` does. As documented
`here <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix>`__,
it has 3 variants: ``.num = .x1``, ``.num = .x2``, and ``.num = .x4``.
For simplicity, we only consider the ``.num = .x4`` variant. In this
case, each thread provides an address which points to 8 elements in
shared memory (those 8 elements should be in consecutive in one row or
one column), and after this instruction, each thread will have 8
elements in registers, which will be fed to the MMA instruction.

So our solution is that, for each thread, we first examine whether the
start address of the 8 elements are out of bound. If it’s not, then we
can safely use the address (since the innermost dimension is padded to
multiple of 8). If it’s out-of-bound, we will feed a special address
that is guaranteed to be valid and contains zero. We may also fill the
red area with zero to ensure correctness.

Here are some discussions about other solutions:

-  Can we use ``copy_if`` in CuTe to “mask off” out-of-bound elements?
   Probably not. As pointed out by the PTX ISA document, The mandatory
   ``.aligned`` qualifier in ``ldmatrix`` indicates that all threads in
   the warp must execute the same ``ldmatrix`` instruction. In
   conditionally executed code, an ldmatrix instruction should only be
   used if it is known that all threads in the warp evaluate the
   condition identically, otherwise the behavior is undefined. Since
   ``copy_if`` is based on conditional execution, it’s not safe to use
   it in this scenario.
-  If we pad each dimension to multiple of 16, can we avoid this
   problem? Probably not. The number of warps is usually > 1, so each
   time we call ``copy``, it copies a tile with side length = 16k (where
   k is an integer). You must pad the dimension to multiple of 16k,
   which may be a waste of memory and performance.

How to Decide ``thr_layout`` when Calling ``make_tiled_mma``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prerequisites: - MMA primitives in CuTe, including ``MMAAtom`` and
``TiledMMA``

CuTe provides the primitive ``MMAAtom`` to represent one basic matrix
multiplication operation performed cooperatively by a group of threads.
For example, ``SM80_16x8x16_F16F16F16F16_TN`` is a ``MMAAtom`` that is
performed by a group of 32 threads.

Usually we have a number of threads that is a multiple of that group
size, and while performing MMA, we want to distribute the large MMA jobs
to every group, letting every group of threads perform serveral
``MMAAtom``\ s (the ``thr_layout`` parameter in ``make_tiled_mma``). For
example, assume we are using ``SM80_16x8x16_F16F16F16F16_TN`` so the
group size is 32, and we have 128 threads (4 groups). Assume the size of
the entire MMA job is 64x32x16 (mxnxk), there are 4 methods to assign
``MMAAtom``\ s to thread groups, as shown below (letters on squares
indicating which group reads/writes data in this square):

.. figure:: /images/mma-thr-layout-example.drawio.svg
   :alt: mma-thr-layout-example

   mma-thr-layout-example

We need to find an optimal ``thr_layout`` which minimize the cost of
data copying. To achieve this, we enumerate the number of groups along
the m and n axes (we usually set the number of groups along the k axis
to 1), then calculate the number of MMA atom calls and the total volume
of data copied. Finally, we choose the layout with the minimum cost.

How to Check Whether or Not We Can Use Chunked Copy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When performing G->S or S->G copying, we want to copy in chunks
(e.g. perform copy in uint128_t) to improve the performance. However,
this needs the layout to be “chunk congruent”, meaning that every 16
Bytes (8 halfs) in the DTensor are contiguous in the STensor. The
problem is, how to test against this?

Iterating every 16 bytes in the DTensor and checking whether they are
contiguous in the STensor is a correct solution, but not feasible since
it’s too slow.

To solve this, we first find the “real innermost dimension”, which is
defined as “the non-1 dimension with a stride of 1” (that dimension must
be unique). Our chunked copy is performed “along” that dimension (you
may refer to the Runtime for details). We can just check whether the
first 8 element in the real innermost dimension are contiguous in the
STensor. If they are, we can derive that every 16 Bytes in the DTensor
are contiguous in the STensor via some linear algebra, and we can use
chunked copy.

The logic mentioned above is implemented in the function
``can_use_chunked_copy``.

When and How to Store the Accumulator in Register File (instead of Shared Memory)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes we can get a large performance gain if we store the
accumulator in the register file instead of shared memory, for example,
in matrix multiplication. Then the problem is: - How to decide whether
to store the accumulator in the register file or shared memory? - How to
implement this?

(BTW, this is an example of that “the Transpiler is both an algorithm
challenge and an engineering challenge”)

Let’s start from the first question. I think the main corcern should be
the limited number of registers. According to `NVIDIA’s
document <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications>`__,
each thread can use up to 255 32-bit registers. That’s not too much.

Currently we use a simple heuristic to decide this: we first check the
per-thread register burden if we store the accumulator in the register
file. If it’s less than 192, we store the accumulator in the register
file. Otherwise, we store it in shared memory. We may use advanced
techniques in the future.

The second question is how to implement this. Currently this is only
implemented for matmul operator since we can obtain a huge performance
gain on matmul when storing the accumulator in the register file. To
implement this, three methods are implemented for the ``tb::Matmul``
kernel in the Runtime:

-  ``get_mma_rC()``: Get a fragment (tensor on register) that stores the
   accumulator
-  ``run()``: Perform the matmul operation
-  ``write_back_mma_rC()``: Write the fragment back to shared memory

If we are going to store the accumulator on register file, the program
looks like this:

.. code-block:: cpp

    auto t = get_mma_rc();
    for (...) {
      run(t, ...);
    }
    write_back_mma_rc(t, ...);

If we are going to store the accumulator in shared memory, the program
looks like this:

.. code-block:: cpp

    for (...) {
      auto t = get_mma_rc();
      run(t, ...);
      write_back_mma_rc(t, ...);
    }
