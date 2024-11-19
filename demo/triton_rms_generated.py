import triton
import triton.language as tl
import triton.ops as ops
import torch

@triton.jit
def custom_kernel_0(dtensor10009983: tl.tensor, dtensor10009981: tl.tensor, dtensor10009982: tl.tensor):
  stensor20213901 = tl.zeros((16,64,), dtype=tl.float32)
  stensor20213905 = tl.zeros((16,128,), dtype=tl.float32)
  for i in range(64):
    stensor20213897 = tl.load(dtensor10009981 + (tl.arange(0, 16))[:, None] * 64 + (i * 64 + tl.arange(0, 64))[None, :])
    stensor20213898 = tl.load(dtensor10009982 + (i * 64 + tl.arange(0, 64))[:, None] * 128 + (tl.program_id(0) * 128 + tl.arange(0, 128))[None, :])
    stensor20213899 = stensor20213897 * stensor20213897
    stensor20213900 = stensor20213899 * 0.000244
    stensor20213901 += stensor20213900
    stensor20213904 = tl.dot(stensor20213897, stensor20213898)
    stensor20213905 += stensor20213904
  stensor20213902 = tl.sum(stensor20213901, axis=1)
  stensor20213902 = stensor20213902[:,None]
  stensor20213903 = tl.sqrt(stensor20213902)
  stensor20213906 = tl.fdiv(stensor20213905, stensor20213903)
  tl.store(dtensor10009983 + (tl.arange(0, 16))[:, None] * 128 + (tl.program_id(0) * 128 + tl.arange(0, 128))[None, :], stensor20213906)


if __name__ == "__main__":
  device = torch.device('cuda')
  dtensor10009981 = torch.randn((16,4096,), dtype=torch.float16).to(device=device)
  dtensor10009982 = torch.randn((4096,6144,), dtype=torch.float16).to(device=device)
  dtensor10009983 = torch.randn((16,6144,), dtype=torch.float16).to(device=device)
  custom_kernel_0[(48, 1, 1)](dtensor10009983, dtensor10009981, dtensor10009982)
