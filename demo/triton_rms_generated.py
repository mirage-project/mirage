import triton
import triton.language as tl
import triton.ops as ops
import torch

@triton.jit
def custom_kernel_23(dtensor10010052: tl.tensor, dtensor10010050: tl.tensor, dtensor10010051: tl.tensor):
  stensor20214361 = tl.zeros((16,64,), dtype=tl.float32)
  # Original shape: (16,64,)
  stensor20214365 = tl.zeros((16,256,), dtype=tl.float32)
  # Original shape: (16,192,)
  for i in range(64):
    stensor20214357 = tl.load(dtensor10010050 + (tl.arange(0, 16))[:, None] * 4096 + (i * 64 + tl.arange(0, 64))[None, :] * 1, mask=None)
    stensor20214358 = tl.load(dtensor10010051 + (i * 64 + tl.arange(0, 64))[:, None] * 6144 + (tl.program_id(0) * 192 + tl.arange(0, 256))[None, :] * 1, mask=(tl.arange(0, 256)[None, :]) < 192)
    stensor20214359 = stensor20214357 * stensor20214357
    stensor20214360 = stensor20214359 * 0.000244
    stensor20214361 += stensor20214360
    stensor20214364 = tl.dot(stensor20214357, stensor20214358)
    stensor20214365 += stensor20214364
  stensor20214362 = tl.sum(stensor20214361, axis=1, keep_dims=True)
  stensor20214363 = tl.sqrt(stensor20214362)
  stensor20214366 = tl.fdiv(stensor20214365, stensor20214363)
  tl.store(dtensor10010052 + (tl.arange(0, 16))[:, None] * 6144 + (tl.program_id(0) * 192 + tl.arange(0, 256))[None, :], stensor20214366, mask=(tl.arange(0, 256)[None, :]) < 192)


if __name__ == "__main__":
  device = torch.device('cuda')
  dtensor10010050 = torch.randn((16,4096,), dtype=torch.float16).to(device=device)
  dtensor10010051 = torch.randn((4096,6144,), dtype=torch.float16).to(device=device)
  dtensor10010052 = torch.randn((16,6144,), dtype=torch.float16).to(device=device)
  custom_kernel_23[(32, 1, 1)](dtensor10010052, dtensor10010050, dtensor10010051)
