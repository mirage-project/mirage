<div align="center">

# Mirage Persistent Kernel: Compiling LLMs into a MegaKernel
    
| [Join Slack](https://join.slack.com/t/miragesystem/shared_invite/zt-37reobr1i-SKjxeYF3GXdPDoCvtVbjTQ) | [Roadmap](https://github.com/mirage-project/mirage/issues/325) | [Blog Post](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17) | 

</div>

*Latest News* 🔥
* [2025/06] We released [Mirage Persistent Kernel (MPK)](https://github.com/mirage-project/mirage/tree/mpk), a compiler and runtime that automatically transforms multi-GPU LLM inference into a high-performance megakernel.

## About

**Mirage Persistent Kernel (MPK)** is a compiler and runtime system that automatically transforms LLM inference into a single megakernel—a fused GPU kernel that performs all necessary computation and communication within a single kernel launch. This end-to-end GPU fusion approach reduces LLM inference latency by 1.2× to 6.7×, all while requiring minimal developer effort.

## Quick Installation

The fastest way to try MPK is to install it directly from source:
```bash
git clone --recursive --branch mpk https://www.github.com/mirage-project/mirage
cd mirage
pip install -e . -v
export MIRAGE_HOME=$(pwd)
```

> 🔧[2025/06/19] We are working on pre-built binary wheels for MPK and will update the installation instructions once they are available.

## Quickstart
Mirage allows you to compile LLMs from the Hugging Face model zoo into a megakernel using just a few dozen lines of Python—mainly to define the kernel’s inputs and outputs. See [this demo script](https://github.com/mirage-project/mirage/blob/mpk/demo/qwen3/demo.py) that compiles the Qwen3-8B model into a megakernel.

We start by running the demo with native Triton and FlashInfer kernels:
```bash
python demo/qwen3/demo.py
```

To compile and execute the megakernel using MPK:
```bash
python demo/qwen3/demo.py --use-mirage
```

To enable profiling (which visualizes the execution timeline of each task):
```bash
python demo/qwen3/demo.py --use-mirage --profiling
```

## How MPK Works
Once you've imported the Mirage package, you can instantiate a persistent kernel using the following API:
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
* `world_size` and `mpi_rank`: number of GPUs and current GPU rank.
* `num_workers`, `num_local_schedulers`, `num_remote_schedulers`: the number of workers, local schedulers, and remote schedulers. They must match the number of physical SMs (`num_workers` + (`num_local_schedulers` + `num_remote_schedulers`) / 4).
* The megakernel currently requires two meta tensors: `step` is an array of integer tracking the current decoding step, and is incremented by MPK after each decoding iteration; `tokens` is a tensor of shape [`num_requests`, `seq_length`] storing prompts and MPK generated tokens.

To attach an existing `PyTorch.Tensor`:
```python
x = mpk.attach_input(torch_tensor=torch_tensor, name="torch_tensor_name")
```
* `name` is used by MPK to refer to the tensor in the generated megakernel in CUDA.

To allocate a new tensor:
```python
y = mpk.new_tensor(
    dims=(batch_size, hidden_size),
    dtype=mi.bfloat16,
    name="embed_out",
    io_category="cuda_tensor",
)
```
* `dims` and `dtype` specify the dimensions and data type of the tensor. 
* `name` is used by MPK to refer to this new tensor in the megakernel. 
* `io_category` indicates how the tensor is allocated and must be `cuda_tensor` or `nvshmem_tensor` (the latter is required for remote GPU access, e.g., during all-reduce).

### Defining the Computation Graph
You can compose the LLM’s computation graph by chaining fused operations. For example: `rmsnorm_linear_layer` fuses an RMSNorm layer and a Linear layer in the megakernel.
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
* `weight_norm` and `weight_linear` are the weight tensors for RMSNorm and Linear.
* `input` and `output` are the input and output tensors of this fused layer. 
* `grid_dim` and `block_dim` specifies the number of thread blocks (i.e., number of tasks in the task graph) and number of thread within each thread block. To minimize latency, it is suggested that the total number of thread blocks is a multiplier of the number of workers to avoid outliers.

### Compilation & Execution
Once the computation graph is defined, compile it with:
```python
mpk.compile()
```
Then, run the optimized megakernel as:
```python
mpk()
```

## Contribution
We welcome feedback, contributions, and collaborations from the community! Please join our [Slack channel](https://join.slack.com/t/mirage-ag11870/shared_invite/zt-37reobr1i-SKjxeYF3GXdPDoCvtVbjTQ).

Please let us know if you encounter any bugs or have any suggestions by [submitting an issue](https://github.com/mirage-project/mirage/issues).

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
