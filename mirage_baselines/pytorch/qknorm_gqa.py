import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', type=int, required=True)
args = parser.parse_args()
print("Batch size", args.batch)

Q = torch.rand([args.batch, 2, 256, 64], dtype=torch.float16, device='cuda')
K = torch.rand([args.batch, 2, 64, 4096], dtype=torch.float16, device='cuda')
V = torch.rand([args.batch, 2, 4096, 64], dtype=torch.float16, device='cuda')

multihead_attn = torch.nn.MultiheadAttention(embed_dim=32 * 64, num_heads = 2, batch_first=True, device='cuda', dtype=torch.float16)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
rms_norm64 = torch.nn.RMSNorm(64, device='cuda:0', dtype=torch.float16)

repetitions = 1024
timings=np.zeros((repetitions,1))

for rep in range(16):
    Q = rms_norm64(Q)
    V = rms_norm64(V)
    S = torch.matmul(Q, K)
    S = torch.softmax(S, dim=3)
    S = torch.matmul(S, V)

with torch.no_grad():
  for rep in range(repetitions):
      starter.record()
      Q = rms_norm64(Q)
      V = rms_norm64(V)
      S = torch.matmul(Q, K)
      S = torch.softmax(S, dim=3)
      S = torch.matmul(S, V)
      ender.record()
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions

print(mean_syn)

