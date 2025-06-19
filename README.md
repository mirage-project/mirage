# Mirage Persistent Kernel: Compiling LLMs into a MegaKernel

*Latest News* 🔥
* [2025/06] We released [Mirage Persistent Kernel (MPK)](https://github.com/mirage-project/mirage/tree/mpk), a compiler and runtime that automatically transforms multi-GPU LLM inference into a high-performance megakernel.

## About

MPK is a compiler and runtime that automatically transforms LLM inference into a single megakernel — a fused GPU kernel that performs all necessary computation and communication in one launch. This end-to-end GPU fusion approach reduces LLM inference latency by 1.2-6.7x with minimal developer effort.

## Quick Installation

The quickest way to try MPK is installing the latest version from source code:
```bash
git clone --recursive --branch mpk https://www.github.com/mirage-project/mirage
cd mirage
pip install -e . -v
```

* [2025/06/19] We are actively working on preparing pre-built binary wheels for MPK and will update the build instructions soon.

## Quickstart
Mirage can compile LLMs from the HuggingFace Transformer model zoo into a megakernel with just a few dozen lines of Python code--mainly to specify the megakernel's inputs and outputs. [This demo](https://github.com/mirage-project/mirage/blob/mpk/demo/qwen3/demo.py) shows how to convert the Qwen3-8B model into a megakernel. You can run the demo using the native Triton + FlashInfer kernels as follows.
```python
python demo/qwen3/demo.py
```

You can let MPK compile the LLM into a megakernel and run the kernel by pass the `--use-mirage` argument.
```python
python demo/qwen3/demo.py --use-mirage
```

Meanwhile, we also provide a profiling interface to profile the performance of megakernels by visualizing the execution timeline of each task. You can enable profiling by sending an extra `--profiling` argument.
```python
python demo/qwen3/demo.py --use-mirage
```

## How MPK Works
After importing the mirage python package, you can create a megakernel (a.k.a, persistent kernel) through the following API.
```python
import mirage as mi
mpk = mi.PersistentKernel(
    world_size=world_size,
    mpi_rank=rank,
    num_workers=96,
    num_local_schedulers=48,
    num_remote_schedulers=0,
    meta_tensors=[step, tokens],
    profiler_tensor=profiler_tensor,
)
```
where `world_size` and `mpi_rank` specifies the number of GPUs (i.e., tensor parallel size) and the rank of the current GPU. The arguments `num_workers`, `num_local_schedulers`, and `num_remote_schedulers` specifies the number of workers, local schedulers, and remote schedulers. MPK requires that these numbers match the total number of physical SMs (i.e., `num_workers` + (`num_local_schedulers` + `num_remote_schedulers`) / 4 == number of physical SMs). The megakernel currently requires two meta tensors: `step` is an array of integer that specify the current decoding step, and MPK will increase that value by one after each iteration; `tokens` is a tensor of shape `num_requests` x `seq_length` that stores all prompt tokens. MPK will write generated tokens back to the `tokens` tensor.

Next, users can attach `PyTorch.Tensor` to the megakernel by calling `mpk.attach_input`
```python
x = mpk.attach_input(torch_tensor=torch_tensor, name="torch_tensor_name")
```
where `name` is the name that is used by MPK to refer to the tensor in the generated megakernel.

Alternatively, users can create a new tensor for the megakernel using `mpk.new_tensor`
```python
y = mpk.new_tensor(
    dims=(batch_size, hidden_size),
    dtype=mi.bfloat16,
    name="embed_out",
    io_category="cuda_tensor",
)
```
where `dims` and `dtype` specifies the dimensions and data type of the tensor. Similarly to the previous API, `name` will be used by MPK to refer to this new tensor. Finally, `io_category` indicates how the tensor is allocated and must be one of `cuda_tensor` and `nvshmem_tensor`. Note that tensors that will be accessed by remote GPUs (e.g., during allreduce) must be allocated as an `nvshmem_tensor`.

The next step is to define the computation graph of the LLM. For example, `mpk.rmsnorm_linear_layer` adds an RMSNorm layer followed by a Linear layer and fuses them together.
```python
mpk.rmsnorm_linear_layer(
    input=x,
    weight_norm=w_norm,
    weight_linear=w_qkv,
    output=attn_in,
    grid_dim=(96, 1, 1),
    block_dim=(128, 1, 1),
)
```
`weight_norm` and `weight_linear` are the weight tensors for RMSNorm and Linear; `input` and `output` are the input and output tensors of this fused layer; finally, `grid_dim` and `block_dim` specifies the number of thread blocks (i.e., number of tasks in the task graph) and number of thread within each thread block. To minimize latency, it is suggested that the total number of thread blocks is a multiplier of the number of workers to avoid outliers.

Finally, you can compile the LLM into a megakernel by invoking `mpk.compile()`, which produces an optimized task graph and CUDA implementations for each task. You can execute the generated megakernel by calling mpk as a function `mpk()`.

## Contribution
Please let us know if you encounter any bugs or have any suggestions by [submitting an issue](https://github.com/mirage-project/mirage/issues).

We welcome all contributions to Mirage from bug fixes to new features and extensions.

## Citation
A paper describing Mirage's techniques is available [on arxiv](https://arxiv.org/abs/2405.05751). Please cite Mirage as:

``` bibtex
@inproceedings {wu2024mirage,
title={Mirage: A Multi-Level Superoptimizer for Tensor Programs}, 
author={Mengdi Wu and Xinhao Cheng and Shengyu Liu and Chunan Shi and Jianan Ji and Kit Ao and Praveen Velliengiri and Xupeng Miao and Oded Padon and Zhihao Jia},
booktitle = {19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 25)},
year = {2025},
address = {Boston, MA},
publisher = {USENIX Association},
month = jul
}
```

## License
Mirage uses Apache License 2.0.
