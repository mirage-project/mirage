"""Test the ``layers.AllReduce`` catalog module in single-GPU mode.

For world_size==1, AllReduce.forward is the identity (plus optional
residual). The compile path requires multi-GPU NVSHMEM; we skip if
world_size==1 (the standard test-mode case). The forward reference is
still exercised in eager PyTorch.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.allreduce import AllReduce


def test_allreduce_forward_identity():
    # AllReduce on single GPU has compile path tied to NVSHMEM (use_nvshmem
    # requires world_size > 1). We can only exercise forward() here.
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    batch_size = 2
    hidden_size = 128

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)

    m = AllReduce(prefix="test_")
    # No active PK; forward should pass without raising and return
    # identity (+residual if given).
    out = m.forward(x)
    torch.testing.assert_close(out, x, atol=1e-2, rtol=1e-2)

    out_with_res = m.forward(x, residual=residual)
    torch.testing.assert_close(
        out_with_res, x + residual, atol=1e-2, rtol=1e-2
    )

    print("SKIPPED compile path: AllReduce.compile requires multi-GPU NVSHMEM "
          "(world_size>1). forward() identity+residual reference verified.")
    print("PASSED: AllReduce forward identity+residual matches expected.")


if __name__ == "__main__":
    test_allreduce_forward_identity()
