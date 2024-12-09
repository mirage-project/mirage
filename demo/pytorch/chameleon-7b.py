import mirage as mi
import argparse
import os
import torch
import flashinfer

n_local_heads = 32
n_local_kv_heads = 32
head_dim = 128
num_tokens = 8
num_kv_tokens = 4096
batch_size = 8

rms_norm4k = torch.nn.RMSNorm(4096, device='cuda:0', dtype=torch.float16)
rms_norm128 = torch.nn.RMSNorm(128, device='cuda:0', dtype=torch.float16)
silu = torch.nn.SiLU()

def torch_llama(X, Wqkv, Wo, W13, W2, Kcache, Vcache):
    X = rms_norm4k(X)
    Xqkv = torch.matmul(X, Wqkv)
    Xq = Xqkv[:, : (n_local_heads * head_dim)]
    output_shape = Xq.shape
    Xkv = Xqkv[:, (n_local_heads * head_dim) :]
    Xk, Xv = Xkv.chunk(2, 1)
    Xq = Xq.view(Xq.shape[0], n_local_heads, head_dim)
    Xk = Xk.view(Xk.shape[0], n_local_kv_heads, head_dim)
    Xv = Xv.view(Xv.shape[0], n_local_kv_heads, head_dim)
    Xq = rms_norm128(Xq)
    Xk = rms_norm128(Xk)
    output = flashinfer.single_prefill_with_kv_cache(Xq, Kcache, Vcache, causal=True)
    output = torch.matmul(output.reshape(output_shape), Wo)
    # RMSNorm
    X = output
    X = rms_norm4k(X)
    X13 = torch.matmul(X, W13)
    X1, X3 = X13.chunk(2, -1)
    output = torch.matmul(silu(X1) * X3, W2)
    return output

if __name__ == "__main__":
    X = torch.randn(batch_size * num_tokens, 4096, dtype=torch.float16, device='cuda:0')
    Wqkv = torch.randn(4096, n_local_heads * head_dim + 2 * n_local_kv_heads * head_dim, dtype=torch.float16, device='cuda:0')
    Wo = torch.randn(n_local_heads * head_dim, 4096, dtype=torch.float16, device='cuda:0')
    W13 = torch.randn(4096, 11008 * 2, dtype=torch.float16, device='cuda:0')
    W2 = torch.rand(11008, 4096, dtype=torch.float16, device='cuda:0')
    Kcache = torch.rand(num_kv_tokens, n_local_kv_heads, head_dim, dtype=torch.float16, device='cuda:0')
    Vcache = torch.rand(num_kv_tokens, n_local_kv_heads, head_dim, dtype=torch.float16, device='cuda:0')
    for _ in range(16):
        torch_llama(X, Wqkv, Wo, W13, W2, Kcache, Vcache)
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(1000):
        torch_llama(X, Wqkv, Wo, W13, W2, Kcache, Vcache)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    mean_syn = curr_time / 1000

    print("Torch Chameleon-7B run time (ms): ", mean_syn)

