# Mirage: A Multi-level Superoptimizer for Tensor Algebra

Mirage is a tensor algebra superoptimizer that automatically discovers highly-optimized tensor programs for DNNs. Mirage automatically identifies and verifies sophisticated optimizations, many of which require joint optimization at the kernel, thread block, and thread levels of the GPU compute hierarchy. For an input DNN, Mirage searches the space of potential tensor programs that are functionally equivalent to the given DNN to discover highly-optimized candidates. This approach allows Mirage to find new custom kernels that outperform existing expert-designed ones. 

## Quick Installation

The quickest way to try Mirage is installing the latest stable release from pip:
```bash
pip install mirage-project
```
You can also [install Mirage from source](INSTALL.md).

## Quickstart

As a tensor algebra superoptimizer, Mirage can be used to accelerate arbitrary DNNs. We use two examples to show how to use Mirage to automatically generate GPU kernels that outperform existing hand-written alternatives. More examples are available in [tutorials](https://mirage-project.readthedocs.io/en/latest/tutorials/index.html).

### Group-query attention (GQA)

The follow code snippet shows how to use Mirage to automatically generate highly-optimized CUDA programs for group-query attention (GQA) in LLAMA-3-70B. We assume the model is served in half precision and is tensor model parallelized across 4 GPUs to fit in GPU device memory. Therefore, the GQA operator computes attention across 8 query heads and 2 key-value heads.

First, we define the computation graph for GQA, which takes three input tensors `Q`, `K`, and `V`, and produces a single output tensor `O` that contains the attention result:

```python
import mirage as mi
graph = mi.new_kernel_graph()
Q = graph.new_input(dims=(2, 256, 64), dtype=mi.float16)
K = graph.new_input(dims=(2, 64, 4096), dtype=mi.float16)
V = graph.new_input(dims=(2, 4096, 64), dtype=mi.float16)
A = graph.matmul(Q, K)
E = graph.exp(A)
S = graph.reduction(E, 2)
D = graph.div(E, S)
O = graph.matmul(D, V)
```

Second, we will use `mi.superoptimize` to superoptimize GQA. Mirage will automatically search the space of potential mugraphs that are functionally equivalent to the input graph to discover highly-optimized CUDA programs. MuGraphs are a new multi-level graph representation in Mirage that specifies computation at the kernel, thread block, and thread levels of the GPU compute hierarchy. An introduction to uGraph is available [here](https://mirage-project.readthedocs.io/en/latest/mugraph.html). Mirage can automatically find uGraphs that represent today's expert-designed GPU optimizations such as FlashAttention, FlashDecoding, and FlashInfer. In addition, Mirage also discovers other uGraphs that outperform these expert-designed implementations for certain cases.

```python
optimized_graph = graph.superoptimize(config="attention")
```

The `superoptimize` function returns the best uGraph discovered by Mirage. The object `optimized_graph` can directly run as a function, and doing so will let Mirage transpile the uGraph into CUDA code, compile the code for execution, and launch the compiled kernel. This allows users to directly run Mirage-generated kernels in their Python programs.

```python
import torch
input_tensors = [
    torch.randn(64, 1, 128, dtype=torch.float16, device='cuda:0'),
    torch.randn(64, 128, 4096, dtype=torch.float16, device='cuda:0'),
    torch.randn(64, 4096, 128, dtype=torch.float16, device='cuda:0')
]
# Launch the Mirage-generated kernel to perform attention
output = optimized_graph(input_tensors)
```

### RMSNorm + MatMul

This example demonstates how to generate fast kernels that compute [RMSNorm](https://arxiv.org/pdf/1910.07467) following by a linear layer. We can use Mirage to automatically generate a highly optimized kernel that fuses RMSNorm and MatMul.
```python
import mirage as mi
graph = mi.new_kernel_graph()
X = graph.new_input(dims=(16, 4096), dtype=mi.float16)
W = graph.new_input(dims=(406, 4096), dtype=mi.float16)
O = graph.matmul(graph.rms_norm(X), W)
graph.mark_output(O)
optimized_graph = graph.superoptimize()

import torch
input_tensors = [
    torch.randn(16, 4096, dtype=torch.float16, device='cuda:0'),
    torch.randn(4096, 4096, dtype=torch.float16, device='cuda:0')
]
optimized_graph(input_tensors)
```
The `optimized_graph` is 1.5-1.7x faster than running the two operators sequentially and can be directly used in your Python program.

## Contribution
Please let us know if you encounter any bugs or have any suggestions by [submitting an issue](https://github.com/mirage-project/mirage/issues).

We welcome all contributions to Mirage from bug fixes to new features and extensions.

## Citation
A paper describing Mirage's techniques is available [on arxiv](https://arxiv.org/abs/2405.05751). Please cite Mirage as:

``` bibtex
@misc{wu2024mirage,
      title={A Multi-Level Superoptimizer for Tensor Programs}, 
      author={Mengdi Wu and Xinhao Cheng and Oded Padon and Zhihao Jia},
      eprint={2405.05751},
      archivePrefix={arXiv},
      year={2024},
}
```

## License
Mirage uses Apache License 2.0.
