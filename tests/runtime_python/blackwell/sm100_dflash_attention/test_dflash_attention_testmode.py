"""Test-mode test for DFlash block attention (TASK_DFLASH_ATTENTION_SM100),
covering the mma flash path + kv-head grid split added for Kimi-K2.6.

Four layers in one task graph:
  - kimi:    production shape (64 q / 8 kv heads, B=8, grid split G=8 ->
             8 q / 1 kv per task, 64 mma rows), full attention,
             ctx_len=300 (partial 64-key tile -> -inf sentinel masking)
  - window:  same shape with sliding_window=2048 and ctx_len=2500, so whole
             KV tiles below the window get skipped
  - nosplit: mma path with G=1 (kernel loops kv heads internally),
             16 q / 2 kv heads, B=8
  - ref:     shape the mma path does not cover (B=4, GQA 4:1 -> 16 rows),
             exercising the scalar dflash_attention_sm100_ref fallback with
             a small symmetric window

Each is checked against the shared PyTorch reference (non-causal block
attention over [ctx ++ block] keys, |q_pos - key_pos| window).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import dflash_attention_ref

HEAD_DIM = 128


def main():
    torch.manual_seed(0)
    device = "cuda"

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    pk = PersistentKernel(**params)

    # (name, nq, nkv, B, grid_x, sliding_window, ctx_len)
    configs = [
        ("kimi", 64, 8, 8, 8, 0, 300),
        ("window", 64, 8, 8, 8, 2048, 2500),
        ("nosplit", 16, 2, 8, 1, 0, 100),
        ("ref", 16, 4, 4, 1, 64, 50),
    ]

    cases = []
    for name, nq, nkv, b, grid_x, sw, ctx_len in configs:
        q = torch.randn(b, nq * HEAD_DIM, dtype=torch.bfloat16, device=device)
        ctx_k = torch.randn(ctx_len, nkv * HEAD_DIM, dtype=torch.bfloat16, device=device)
        ctx_v = torch.randn(ctx_len, nkv * HEAD_DIM, dtype=torch.bfloat16, device=device)
        blk_k = torch.randn(b, nkv * HEAD_DIM, dtype=torch.bfloat16, device=device)
        blk_v = torch.randn(b, nkv * HEAD_DIM, dtype=torch.bfloat16, device=device)
        out = torch.zeros(b, nq * HEAD_DIM, dtype=torch.bfloat16, device=device)

        dts = {}
        for tname, t in (("q", q), ("ctx_k", ctx_k), ("ctx_v", ctx_v),
                         ("blk_k", blk_k), ("blk_v", blk_v), ("out", out)):
            dts[tname] = pk.attach_input(t, name=f"{name}_{tname}")

        pk.dflash_attention_layer(
            q=dts["q"], ctx_k=dts["ctx_k"], ctx_v=dts["ctx_v"],
            blk_k=dts["blk_k"], blk_v=dts["blk_v"], output=dts["out"],
            grid_dim=(grid_x, 1, 1), block_dim=(128, 1, 1),
            sliding_window=sw, head_dim=HEAD_DIM,
        )
        cases.append((name, q, ctx_k, ctx_v, blk_k, blk_v, sw, out))

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ok = True
    for name, q, ctx_k, ctx_v, blk_k, blk_v, sw, out in cases:
        ref = dflash_attention_ref(
            q, ctx_k, ctx_v, blk_k, blk_v, sw, HEAD_DIM
        )
        diff = (out.float() - ref.float()).abs().max().item()
        print(f"[{name}] out max diff: {diff:.3e}")
        try:
            torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        except AssertionError as e:
            print(f"[{name}] FAILED: {e}")
            ok = False

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("PASSED: dflash_attention test_mode produces correct output")


if __name__ == "__main__":
    main()
