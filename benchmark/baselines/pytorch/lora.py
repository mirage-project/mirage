import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', type=int, required=True)
args = parser.parse_args()
print("Batch size", args.batch)

W = torch.rand([256, 4096], dtype=torch.float16, device='cuda')
A = torch.rand([256, 16], dtype=torch.float16, device='cuda')
B = torch.rand([16, 4096], dtype=torch.float16, device='cuda')
X = torch.rand([args.batch, 256], dtype=torch.float16, device='cuda')

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

repetitions = 1024
timings=np.zeros((repetitions,1))

for rep in range(16):
  D = torch.matmul(X, A)
  E = torch.matmul(D, B)
  C = torch.matmul(X, W)

with torch.no_grad():
  for rep in range(repetitions):
      starter.record()
      D = torch.matmul(X, A)
      E = torch.matmul(D, B)
      C = torch.matmul(X, W)
      torch.add(C, E)
      ender.record()
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions

print(mean_syn)

