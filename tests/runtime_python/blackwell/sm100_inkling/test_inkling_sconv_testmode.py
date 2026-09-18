"""Test-mode test for the Inkling depthwise short convolution
(TASK_INKLING_SCONV_SM100).

Builds two independent sconv layers in one task graph:
  - decode (SEQ=1), grid partitions the channel dim (G=8)
  - short extend (SEQ=3) exercising the sliding-window shift, G=4
and checks both the conv+residual output and the in-place conv_state update
against the PyTorch reference.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import inkling_sconv_ref

SCONV_K = 4


def main():
    torch.manual_seed(0)
    device = "cuda"
    hidden = 1024

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    pk = PersistentKernel(**params)

    cases = []  # (name, seq, grid_x)
    for name, seq, grid_x in (("decode", 1, 8), ("extend", 3, 4)):
        x = torch.randn(seq, hidden, dtype=torch.bfloat16, device=device)
        w = 0.5 * torch.randn(hidden, SCONV_K, dtype=torch.float32, device=device)
        state0 = torch.randn(SCONV_K - 1, hidden, dtype=torch.float32, device=device)
        state = state0.clone()  # updated in place by the kernel
        out = torch.zeros(seq, hidden, dtype=torch.bfloat16, device=device)

        x_dt = pk.attach_input(x, name=f"{name}_x")
        w_dt = pk.attach_input(w, name=f"{name}_w")
        state_dt = pk.attach_input(state, name=f"{name}_state")
        out_dt = pk.attach_input(out, name=f"{name}_out")
        pk.inkling_sconv_layer(
            x=x_dt,
            weight=w_dt,
            conv_state=state_dt,
            output=out_dt,
            grid_dim=(grid_x, 1, 1),
            block_dim=(128, 1, 1),
        )
        cases.append((name, x, w, state0, state, out))

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ok = True
    for name, x, w, state0, state, out in cases:
        ref_out, ref_state = inkling_sconv_ref(x, w, state0)
        out_diff = (out.float() - ref_out.float()).abs().max().item()
        state_diff = (state - ref_state).abs().max().item()
        print(f"[{name}] out max diff: {out_diff:.3e}, "
              f"state max diff: {state_diff:.3e}")
        try:
            torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(state, ref_state, atol=1e-5, rtol=1e-5)
        except AssertionError as e:
            print(f"[{name}] FAILED: {e}")
            ok = False

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("PASSED: inkling_sconv test_mode produces correct output")


if __name__ == "__main__":
    main()
