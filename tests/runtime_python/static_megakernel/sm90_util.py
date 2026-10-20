"""What the sm90 static-megakernel tests share: the build flags of the MPK Hopper task bodies, and an MXFP4 decoder."""
import torch
from mirage.mpk.static_megakernel import sm90_flags

SM90_FLAGS = sm90_flags()
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device="cuda")


def decode(packed, scales):
    v = torch.stack((E2M1[(packed & 15).long()], E2M1[(packed >> 4).long()]), dim=-1).flatten(-2)
    s = torch.where(scales == 0, torch.zeros_like(scales, dtype=torch.float32), torch.pow(2.0, scales.float() - 127))
    return (v.reshape(*v.shape[:-1], -1, 32) * s.unsqueeze(-1)).reshape(*v.shape[:-1], -1)
