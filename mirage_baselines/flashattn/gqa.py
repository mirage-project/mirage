from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', type=int, required=True)
args = parser.parse_args()
print("Batch size", args.batch)

Q = torch.rand([args.batch, 32, 16, 64], dtype=torch.float16, device='cuda')
K = torch.rand([args.batch, 4096, 2, 64], dtype=torch.float16, device='cuda')
V = torch.rand([args.batch, 4096, 2, 64], dtype=torch.float16, device='cuda')

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

repetitions = 1024
timings=np.zeros((repetitions,1))

for rep in range(1024):
    flash_attn_func(Q, K, V)

with torch.no_grad():
  for rep in range(repetitions):
      starter.record()
      flash_attn_func(Q, K, V)
      ender.record()
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions

print(mean_syn)

