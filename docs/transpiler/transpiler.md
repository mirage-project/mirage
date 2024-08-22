# The Transpiler

## Introduction to Mirage's Transpiler

The transpiler is a key component of Mirage. It is responsible for translating the (optimized) computation graph produced by Mirage into efficient CUDA code.

It is not easy to write a transpiler. You should ensure both the correctness and the performance, which is both an algorithm challenge (e.g. how to plan the layout of every tensor) and an engineering challenge (e.g. how to design the transpiler in an extensible and maintainable way). There are also numerous details and corner cases to consider, like what to do when the shape of the matrix is not divisible by the MMA size.

In this document, we will first introduce the architecture of the transpiler, then dive into the details of some key algorithms and components, and finally discuss some problems and solutions on corner cases.

## Architecture

The transpiler has two crucial components: the "Transpiler" and the "Runtime".

The "Transpiler" is a part of Mirage's codebase (under `src/transpiler`). It exposes an interface called `transpile`, which takes a Mirage's computational graph as input and returns the generated CUDA code. Here is an example of a piece of generated code:

```cpp
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
```

The "Runtime" is a header-only library that provides a set of useful kernels & device functions for the transpiled code. Although it's located at `include/mirage/transpiler/runtime`, code in Mirage does not include it. Instead, the generated code will include the "Runtime" library (`runtime.h`), and can call various functions inside it (like we called `kn::gemm` in the example above). It's the joint effort of the "Transpiler" and the "Runtime" that makes the generated code efficient and correct.

Besides, the transpiler also contains a set of tests to ensure the correctness and performance of the generated code. It's located under `tests/transpiler`.

## Prerequisites

Before we dive into the details of the transpiler, we need some prerequisites:
- You need to master **CUDA Programming**, of course.
- To leverage compile-time computation, we use **C++ Template Metaprogramming** extensively.
- We also make use of the **CuTe Library**. It's a library inside the [Cutlass library](https://github.com/NVIDIA/cutlass) and provides a set of useful primitives for tensor core matrix multiplication and data movement. CuTe has its own [document](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md) but it's poorly written. If you speak Chinese (or you are good at Google Translate), you can refer to [these series of blogs](https://zhuanlan.zhihu.com/p/661182311) for a better understanding.
- We suppose you have a basic understanding of **the Mirage project**, particularly, what `DTensor`, `STensor`, "kernel graph", "kernel operator", "threadblock graph", "threadblock operator" mean.

## Transpiler Overview

The Transpiler can be divided into several steps:

1. **Threadblock Level Data Reuse Logic (Fusion Logic).** In threadblock level code, sometimes we can avoid reading and writing data by fusing the next kernel with the current one. An example is that if a matmul is followed by an elementwise exponentiation, we can directly perform the elementwise exponentiation on the result of the matmul, instead of writing the matmul result back to shared memory, and then reading it again. We called this threadblock level data reuse. The Transpiling process begins with planning which operators to fuse since it has an impact on nearly everything (layout, scheduing, etc.).
2. **Layout Resolution.** Plan the layout (strides of every dimension) of every `DTensor` and `STensor`.
3. **Kernel-Level Memory Planning.** Plan the memory for every `DTensor`.
4. **Kernel-Level Transpile.** Transpile each kernel operator one by one, that is:
  - If the operator is a pre-defined kernel operator (say, a Matmul), we call the corresponding function in the Runtime.
  - If the operator is a custom operator (`KN_CUSTOMIZED_OP`), we call the function `transpile_kn_custom_op` which transpiles the custom operator into a CUDA kernel (a function marked with `__global__`). This includes:
    1. **TB Graph Scheduling and Memory Planning.** Choose the order of operators to perform, and allocate memory for every intermediate `STensor`.
    2. **Threadblock-Level Transpile.** Transpile the threadblock level code based on the TB graph scheduling and memory planning.

## Algorithms

### Threadblock Level Data Reuse

In threadblock level code, an operator (which will be transpiled to a device function) reads data from shared memory, performs computation, and writes the result back to shared memory. However, sometimes we can avoid this reading-writing by "fusing" the next kernel (in this section, "kernel" refers to threadblock level operators) with the current one. For example, if a matmul is followed by an elementwise exponentiation, we can directly perform the elementwise exponentiation on the result of the matmul, instead of writing the matmul result back to shared memory, and then reading it again. We called this threadblock level data reuse.

Achieving threadblock level data reuse is not easy. If the latter kernel is not a simple elementwise operation, you need to align the data layout (for every CUDA thread, which part of data does it hold?) between the former kernel, which is really complex. Besides, if the latter kernel is not a unary (only has one input tensor) kernel, more problems will arise. For example, Assume you have two threadblock level ops, A and B, and an elementwise addition operator which takes the output of A and B as input, and assume that you want to perform A first, then you need to choose between storing the output of A in shared memory or registers. That's a complex decision.

Currently, the Transpiler only considers fusion where the latter operator is an elementwise unary operator. This problem is much easier since we can always fuse an elementwise unary operator with its former operator without any concerns. We may consider more complex scenarios in the future.

In the Runtime, the operator fusion is implemented as "epilogues". An epilogue is a chain of actions performed on a single element. Every node on the chain can be:
- A unary operator, like `exp`
- An action that involves memory operatorion, like "store" (`dst[dst_layout(dst_index)] = x`) or "store-and-accumulate" (`dst[dst_layout(dst_index)] += x`). Every chain is terminated by such an action.

### Layout Resolution

A layout is a mapping between logical coordinates and physical memory addresses. For example, for a 2D tensor, the layout may be row-major or column-major.

The layout is crucial for the performance of the generated code. For example, if the layouts of both tensors are row-major during a G->S (global memory to shared memory) copy, we can copy data in larger chunks (say, 128 bits) instead of copying element by element. Besides, sometimes we may swizzle the layout of STensors to avoid bank conflicts.

To resolve layouts for all tensors, we first constructs an boolean ILP problem to decide the "innermost dimension", the dimension with a stride of 1, of every `DTensor` and `STensor`. For every dimension of every tensor, we have a boolean variable indicating whether it's the innermost dimension. After that, we add a set of restrictions (stands for restrictions proposed by kernels in the Runtime, like cuBLAS requires the innermost dim to be among the last two dimensions), and build the optimization target (one operator under different layouts may have different performance, and we want to minimize the total "cost"). After that, we use the Z3 solver to find the optimal solution.

After deciding the innermost dimension, it's time to calculate the strides. The stride of one dimension is the number of elements between two consecutive elements in this dimension. For example, for a row-major 2D tensor with shape $[m, n]$, it has a stride of $[n, 1]$. The physical address of an element is the dot product between its coordinates and the stride, while in our example, an element with coordinates $(i, j)$ has a physical address of $i \times n + j$.

Here we use a heuristic to calculate the strides, implemented in the function `calc_tensor_strides`. Assume the innermost dim of one tensor is dimension #k, and the number of dimensions is N. We first decide an "order" of all dimensions, which is $[k, N-1, N-2, \dots, k+1, k-1, \dots, 1, 0]$, and then assign strides based on that order (see `calc_tensor_strides` for details). We also pad the first non-1 dimension to a multiple of 8 (since there are 8 halfs in 16 Bytes) to ensure the address of the starting element of every dimension (with coords looks like $(0, 0, \dots, 0, M, 0, \dots, 0)$) is aligned to 16 Bytes.

### TB Graph Scheduling and Memory Planning

The TB graph scheduling problem is that, given a threadblock-level graph, how to get an optimal "schedule" to maximize the performance? The "schedule" here includes:

- The order of operators ("schedule")
- The address of the space allocated for each intermediate tensor ("memory plan")

And in order to optimize the performance, we may need to:

- Minimize the number of synchronizations (`__syncthreads()`) between threadblocks
- Minimize the peak shared memory usage

Those objectives are maybe conflicting, and we need to somehow find a balance between them. Consider the following computational graph:

![tb-sched-conflict-example](/docs/assets/transpiler/tb-sched-conflict-example.drawio.svg)

And there are two possible schedulings:

- `12 534 6 7`: 3 synchronizations (a space denotes a synchronization) with peak mem usage = 4
- `1234 56 7`: 2 synchronizations with peak mem usage = 5

Currently our heuristic is that, we always prioritize the number of synchronizations, and then the peak mem usage. So we first find an order that minimizes the number of synchronizations (if multiple orders have the same number of synchronizations, we choose a random one).

To find the order with the minimum number of synchronizations, we use a modified topology sort algorithm. We label each node (threadblock operator) with a "depth", which is length of the longest path from any input operator to this node. We can calculate this depth by a dynamic programming (DP) algorithm:

- For input nodes, its depth is 0
- Otherwise, its depth is $\max_{v \in I} depth[v] + 1$, where $I$ is the set of direct input nodes of this node

After that, we sort the nodes by their depth in ascending order, and perform a synchronization when the depth of the latter node is greater than the former node. This is how we obtain the schedule.

For memory planning, it's actually a NP-Complete problem. We just run some heuristics (first fit, best fit, last fit, random, etc.) and choose the one with the minimum peak shared memory usage.

## Problems and Solutions

In this section, we will discuss some problems and solutions on corner cases.

### How to Perform Threadblock Level Matrix Multiplication when the Size is not Divisible by MMA Size

Prerequisites:
- The `ldmatrix` instruction
- MMA and Copy in CuTe

Life will be much easier if the size of the matrices is divisible by the MMA size. In such scenarios, we can just copy the data from shared memory to registers via `ldmatrix` (assume SM75+) and perform the matrix multiplication.

However, the story is different when the size of the matrices is not divisible by the MMA size. Since `ldmatrix` loads a 16x16 tile, it may access out-of-bound memory, or produce incorrect results.

As the figure below shows, the black grids are 16x16 tiles, and the green rectangle is the matrix that is not divisible by the MMA size. Areas marked with blue are out-of-bound elements. When performing `ldmatrix` on those areas, we must find a valid memory address to feed the `ldmatrix` instruction, instead of using the out-of-bound address. We do not care about their value. Areas marked with red are out-of-bound elements that may affect the final result. In addition to finding a valid memory address, we must also ensure that those elements are zero.

![mma-non-divisible-example](/docs/assets/transpiler/mma-non-divisible-example.drawio.svg)

Let's recap what `ldmatrix` does. As documented [here](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix), it has 3 variants: `.num = .x1`, `.num = .x2`, and `.num = .x4`. For simplicity, we only consider the `.num = .x4` variant. In this case, each thread provides an address which points to 8 elements in shared memory (those 8 elements should be in consecutive in one row or one column), and after this instruction, each thread will have 8 elements in registers, which will be fed to the MMA instruction.

So our solution is that, for each thread, we first examine whether the start address of the 8 elements are out of bound. If it's not, then we can safely use the address (since the innermost dimension is padded to multiple of 8). If it's out-of-bound, we will feed a special address that is guaranteed to be valid and contains zero. We may also fill the red area with zero to ensure correctness.

Here are some discussions about other solutions:

- Can we use `copy_if` in CuTe to "mask off" out-of-bound elements? Probably not. As pointed out by the PTX ISA document, The mandatory `.aligned` qualifier in `ldmatrix` indicates that all threads in the warp must execute the same `ldmatrix` instruction. In conditionally executed code, an ldmatrix instruction should only be used if it is known that all threads in the warp evaluate the condition identically, otherwise the behavior is undefined. Since `copy_if` is based on conditional execution, it's not safe to use it in this scenario.
- If we pad each dimension to multiple of 16, can we avoid this problem? Probably not. The number of warps is usually > 1, so each time we call `copy`, it copies a tile with side length = 16k (where k is an integer). You must pad the dimension to multiple of 16k, which may be a waste of memory and performance.

### How to Decide `thr_layout` when Calling `make_tiled_mma`

Prerequisites:
- MMA primitives in CuTe, including `MMAAtom` and `TiledMMA`

CuTe provides the primitive `MMAAtom` to represent one basic matrix multiplication operation performed cooperatively by a group of threads. For example, `SM80_16x8x16_F16F16F16F16_TN` is a `MMAAtom` that is performed by a group of 32 threads.

Usually we have a number of threads that is a multiple of that group size, and while performing MMA, we want to distribute the large MMA jobs to every group, letting every group of threads perform serveral `MMAAtom`s (the `thr_layout` parameter in `make_tiled_mma`). For example, assume we are using `SM80_16x8x16_F16F16F16F16_TN` so the group size is 32, and we have 128 threads (4 groups). Assume the size of the entire MMA job is 64x32x16 (mxnxk), there are 4 methods to assign `MMAAtom`s to thread groups, as shown below (letters on squares indicating which group reads/writes data in this square):

![mma-thr-layout-example](/docs/assets/transpiler/mma-thr-layout-example.drawio.svg)

We need to find an optimal `thr_layout` which minimize the cost of data copying. To achieve this, we enumerate the number of groups along the m and n axes (we usually set the number of groups along the k axis to 1), then calculate the number of MMA atom calls and the total volume of data copied. Finally, we choose the layout with the minimum cost.

### How to Check Whether or Not We Can Use Chunked Copy

When performing G->S or S->G copying, we want to copy in chunks (e.g. perform copy in uint128_t) to improve the performance. However, this needs the layout to be "chunk congruent", meaning that every 16 Bytes (8 halfs) in the DTensor are contiguous in the STensor. The problem is, how to test against this?

Iterating every 16 bytes in the DTensor and checking whether they are contiguous in the STensor is a correct solution, but not feasible since it's too slow.

To solve this, we first find the "real innermost dimension", which is defined as "the non-1 dimension with a stride of 1" (that dimension must be unique). Our chunked copy is performed "along" that dimension (you may refer to the Runtime for details). We can just check whether the first 8 element in the real innermost dimension are contiguous in the STensor. If they are, we can derive that every 16 Bytes in the DTensor are contiguous in the STensor via some linear algebra, and we can use chunked copy.

The logic mentioned above is implemented in the function `can_use_chunked_copy`. 