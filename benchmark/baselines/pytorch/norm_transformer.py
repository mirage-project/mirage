import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', type=int, required=True)
args = parser.parse_args()
print("Batch size", args.batch)

X = torch.rand([args.batch, 4096], dtype=torch.float16, device='cuda')
H = torch.rand([args.batch, 4096], dtype=torch.float16, device='cuda')
alpha = torch.rand([args.batch, 4096], dtype=torch.float16, device='cuda')
rms_norm4k = torch.nn.RMSNorm(4096, device='cuda:0', dtype=torch.float16)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

repetitions = 1024
timings=np.zeros((repetitions,1))

for rep in range(16):
    H_norm = rms_norm4k(H)
    A = torch.add(H_norm, X)
    B = torch.mul(alpha, X)
    C = torch.add(X, B)
    O = rms_norm4k(C)

with torch.no_grad():
  for rep in range(repetitions):
      starter.record()
      H_norm = rms_norm4k(H)
      A = torch.add(H_norm, X)
      B = torch.mul(alpha, X)
      C = torch.add(X, B)
      O = rms_norm4k(C)
      ender.record()
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions

print(mean_syn)

