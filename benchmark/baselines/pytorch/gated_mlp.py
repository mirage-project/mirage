import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', type=int, required=True)
args = parser.parse_args()
print("Batch size", args.batch)

silu = torch.nn.SiLU()

X = torch.rand([args.batch, 4096], dtype=torch.float16, device='cuda')
A = torch.rand([4096, 4096], dtype=torch.float16, device='cuda')
B = torch.rand([4096, 4096], dtype=torch.float16, device='cuda')

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

repetitions = 1024
timings=np.zeros((repetitions,1))

for rep in range(128):
      O1 = torch.matmul(X, A)
      O2 = torch.matmul(X, B)
      O1 = silu(O1)
      O = torch.mul(O1, O2)

with torch.no_grad():
  for rep in range(repetitions):
      starter.record()
      O1 = torch.matmul(X, A)
      O2 = torch.matmul(X, B)
      O1 = silu(O1)
      O = torch.mul(O1, O2)
      ender.record()
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions

print(mean_syn)

