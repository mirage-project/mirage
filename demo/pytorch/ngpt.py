import mirage as mi
import argparse
import os
import torch
import flashinfer

n_local_heads = 12
n_local_kv_heads = 12
head_dim = 128
num_tokens = 4
num_kv_tokens = 4096
intermediate_size = 4096
batch_size = 8

rms_norm1 = torch.nn.RMSNorm(n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim, device='cuda:0', dtype=torch.float16)
rms_norm2 = torch.nn.RMSNorm(intermediate_size * 2, device='cuda:0', dtype=torch.float16)
silu = torch.nn.SiLU()

def torch_ngpt(X, Wqkv, Wo, W13, W2, Kcache, Vcache, alpha):
    Xqkv = torch.matmul(X, Wqkv)
    Xqkv = rms_norm1(Xqkv)
    Xq = Xqkv[:, : (n_local_heads * head_dim)]
    output_shape = Xq.shape
    Xkv = Xqkv[:, (n_local_heads * head_dim) :]
    Xk, Xv = Xkv.chunk(2, 1)
    Xq = Xq.view(Xq.shape[0], n_local_heads, head_dim)
    Xk = Xk.view(Xk.shape[0], n_local_kv_heads, head_dim)
    Xv = Xv.view(Xv.shape[0], n_local_kv_heads, head_dim)
    output = flashinfer.single_prefill_with_kv_cache(Xq, Kcache, Vcache, causal=True)
    output = torch.matmul(output.reshape(output_shape), Wo)
    # Norm
    X = output
    X13 = torch.matmul(X, W13)
    X13 = rms_norm2(X13)
    X13 = torch.mul(X13, alpha)
    X13 = rms_norm2(X13)
    X1, X3 = X13.chunk(2, -1)
    output = torch.matmul(silu(X1) * X3, W2)
    return output

if __name__ == "__main__":
    X = torch.randn(batch_size * num_tokens, 4096, dtype=torch.float16, device='cuda:0')
    Wqkv = torch.randn(4096, n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim, dtype=torch.float16, device='cuda:0')
    Wo = torch.randn(n_local_heads * head_dim, 4096, dtype=torch.float16, device='cuda:0')
    W13 = torch.randn(4096, intermediate_size * 2, dtype=torch.float16, device='cuda:0')
    W2 = torch.rand(intermediate_size, 4096, dtype=torch.float16, device='cuda:0')
    Kcache = torch.rand(num_kv_tokens, n_local_kv_heads, head_dim, dtype=torch.float16, device='cuda:0')
    Vcache = torch.rand(num_kv_tokens, n_local_kv_heads, head_dim, dtype=torch.float16, device='cuda:0')
    alpha = torch.rand(batch_size * num_tokens, intermediate_size * 2, dtype=torch.float16, device='cuda:0')
    for _ in range(16):
        torch_ngpt(X, Wqkv, Wo, W13, W2, Kcache, Vcache, alpha)
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(1000):
        torch_ngpt(X, Wqkv, Wo, W13, W2, Kcache, Vcache, alpha)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    mean_syn = curr_time / 1000

    print("Torch nGPT run time (ms): ", mean_syn)

