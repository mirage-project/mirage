# The NKI Transpiler

## Introduction to Mirage's NKI Transpiler

The NKI Transpiler transpiles a muGraph into a NKI program. To cope with the layout requirements of the Trainium devices (i.e., each tensor in sbuf is associated with a partition dim and a free dim), the transpiler requires that only the last two dimensions of a stensor can be of size larger than 1, and all other dimensions must be equal to 1. Meanwhile, the inputs and outputs to NKI APIs are ``tiles'', which are tensors whose first dimension is the partition dim. Based on these constraints, we imply the following assumptions when transpiling a muGraph to a NKI program:

- Assumption #1: for all stensors of a thread block graph (i.e., running on a single NeuronCore), the tensor should only have at most two non-one dimensions and these dimensions must be the last dimensions. That is, only the last two dimensions of a stensor can be of size larger than 1, and all other dimensions must be equal to 1. Based on this assumption, all stensors in the transpiled NKI programs are at most two dimensional --- all other dimensions are omitted.

- Assumption #2: for all stensors, the partition dimension is the first dimension, and this order may be different from the stensor's metadata. For example, for a tensor `A` of shape `[128, 256]`, if the transpile picks the second dimension as the partition dim (i.e., `partition_dim = 1`), the tensor in the transpiled code will have shape `[256, 128]`, whose first dimension is the partition dim. This assumption makes all stensors `tiles` that can be directly used as inputs to NKI operators, while requiring additional `nki.language.transpose` to convert layouts.

## Matrix multiplication

We use [NKI's built-in matmul API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.isa.nc_matmul.html) to perform matrix multiplication. NeuronCore impose the following constraints for matrix multiplication.

To perform a matrix multiplication of shapes `[M, K]` and `[K, N]`, Tensor Engine (the engine performing this operation) requires the `K` dimension to be mapped to the partition dimension in SBUF for both input matrices. Therefore, you need to pass shapes `[K, M]` and `[K, N]` into the nki.isa.nc_matmul API, as the partition dimension is always the left-most dimension for an input tile to any NKI compute API.

The first and second oprands of matmul depends on the partition dim chosen by the transpiler through the ILP algorithm. Specifically, for a matmul operator in a thread block graph, let `A` and `B` be the two input tensors of shape `[Asize, K]` and `[K, Bsize]`, respectively. We need to pick `K` as the partition dim for both inputs to satisfy the above layout constraints. The output stensor `C`'s shape is `[Asize, Bsize]`. We determine the operands based on which dim is the partition dim. If `Asize` is the partition dim of the output tensor, we make `A` the stationary and `B` the moving tile. Otherwise, if `Bsize` is the partition dim of the output tensor, we make `B` the stationary and `A` the moving tile.
