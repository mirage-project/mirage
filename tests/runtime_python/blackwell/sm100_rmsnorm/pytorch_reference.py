"""PyTorch reference implementation for RMSNorm.

The eps value matches the existing test_rmsnorm_testmode.py (1e-5) so that
behavior is preserved when the test was moved into this folder.
"""

import torch


def rmsnorm_ref(x, weight, eps=1e-5):
    """RMSNorm: (x / RMS(x)) * weight."""
    x_f32 = x.to(torch.float32)
    rms = x_f32.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (x_f32 * rms * weight.to(torch.float32)).to(x.dtype)
