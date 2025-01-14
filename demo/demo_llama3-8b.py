import mirage as mi
import argparse
import os
import torch
import flashinfer

n_local_heads = 32
n_local_kv_heads = 8
head_dim = 128
intermediate_size = 14336
num_tokens = 8
num_kv_tokens = 4096

silu = torch.nn.SiLU()
def get_rms_linear_kernel(num_tokens, output_dim):
    graph = mi.new_kernel_graph()
    X = graph.new_input(dims=(num_tokens, 4096), dtype=mi.float16)
    W = graph.new_input(dims=(4096, output_dim), dtype=mi.float16)
    D = graph.rms_norm(X, normalized_shape=(4096,))
    O = graph.matmul(D, W)
    graph.mark_output(O)
    return graph.superoptimize(config="mlp")
   
def mirage_llama(X, Wqkv, Wo, W13, W2, Kcache, Vcache, kernels):
    func = kernels[0]
    outputs = func(inputs=[X, Wqkv])
    Xqkv = outputs[0]
    Xq = Xqkv[:, : (n_local_heads * head_dim)]
    output_shape = Xq.shape
    Xkv = Xqkv[:, (n_local_heads * head_dim) :]
    Xk, Xv = Xkv.chunk(2, 1)
    Xq = Xq.view(Xq.shape[0], n_local_heads, head_dim)
    Xk = Xk.view(Xk.shape[0], n_local_kv_heads, head_dim)
    Xv = Xv.view(Xv.shape[0], n_local_kv_heads, head_dim)
    output = flashinfer.single_prefill_with_kv_cache(Xq, Kcache, Vcache, causal=True)
    X = torch.matmul(output.reshape(output_shape), Wo)
    func = kernels[1]
    outputs = func(inputs=[X, W13])
    X13 = outputs[0]
    X1, X3 = X13.chunk(2, -1)
    output = torch.matmul(X1, W2)
    return output

if __name__ == "__main__":
    X = torch.randn(num_tokens, 4096, dtype=torch.float16, device='cuda:0')
    Wqkv = torch.randn(4096, n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim, dtype=torch.float16, device='cuda:0')
    Wo = torch.randn(n_local_heads * head_dim, 4096, dtype=torch.float16, device='cuda:0')
    W13 = torch.randn(4096, intermediate_size * 2, dtype=torch.float16, device='cuda:0')
    W2 = torch.rand(14336, 4096, dtype=torch.float16, device='cuda:0')
    Kcache = torch.rand(num_kv_tokens, n_local_kv_heads, head_dim, dtype=torch.float16, device='cuda:0')
    Vcache = torch.rand(num_kv_tokens, n_local_kv_heads, head_dim, dtype=torch.float16, device='cuda:0')

    k1 = get_rms_linear_kernel(num_tokens, Wqkv.shape[-1])
    k2 = get_rms_linear_kernel(num_tokens, W13.shape[-1])
    kernels = [k1, k2]

    for _ in range(16):
        mirage_llama(X, Wqkv, Wo, W13, W2, Kcache, Vcache, kernels)
    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    starter.record()
    for rep in range(repetitions):
        mirage_llama(X, Wqkv, Wo, W13, W2, Kcache, Vcache, kernels)

    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)

    mean_syn = curr_time / 1000
    print(mean_syn)
