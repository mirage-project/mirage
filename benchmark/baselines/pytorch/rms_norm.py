import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', type=int, required=True)
args = parser.parse_args()
print("Batch size", args.batch)

X = torch.rand([2 * args.batch, 4096], dtype=torch.float16, device='cuda')
W = torch.rand([4096, 4096], dtype=torch.float16, device='cuda')
rms_norm64 = torch.nn.RMSNorm(4096, device='cuda:0', dtype=torch.float16)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

repetitions = 1024
timings=np.zeros((repetitions,1))

for rep in range(16):
    S = rms_norm64(X)
    S = torch.matmul(S, W)

with torch.no_grad():
  for rep in range(repetitions):
      starter.record()
      S = rms_norm64(X)
      O = torch.matmul(S, W)
      ender.record()
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      timings[rep] = curr_time

mean_syn = np.sum(timings) / repetitions

print(mean_syn)

