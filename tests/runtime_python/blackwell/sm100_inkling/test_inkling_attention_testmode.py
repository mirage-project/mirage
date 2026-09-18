"""Test-mode test for Inkling GQA decode attention
(TASK_INKLING_ATTENTION_SM100).

Three layers in one task graph, matching the per-task shapes the builder
produces (grid.x partitions kv heads, so per-task heads = heads / G):
  - local:  4 q / 1 kv per task (GQA 4:1), sliding window 512, extent 512,
            no log scaling, ctx_len 700 (window active)
  - global: 8 q / 1 kv per task (GQA 8:1), no window, extent 1024,
            log scaling alpha=0.1 with a small n_floor so tau > 1,
            ctx_len 1200 (distances beyond extent hit the zero-bias branch)
  - edge:   ctx_len 0 (first token: only the new blk_k/blk_v attend)
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import inkling_attention_ref

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

    # (name, nq, nkv, grid_x, sw, extent, alpha, n_floor, ctx_len)
    configs = [
        ("local", 8, 2, 2, 512, 512, 0.0, 128000, 700),
        ("global", 16, 2, 2, 0, 1024, 0.1, 64, 1200),
        ("edge", 4, 1, 1, 512, 512, 0.0, 128000, 0),
    ]

    cases = []
    for name, nq, nkv, grid_x, sw, extent, alpha, n_floor, ctx_len in configs:
        max_ctx = max(ctx_len + 16, 32)
        q = 3.0 * torch.randn(1, nq * HEAD_DIM, dtype=torch.bfloat16, device=device)
        ctx_k = torch.randn(max_ctx, nkv * HEAD_DIM, dtype=torch.bfloat16, device=device)
        ctx_v = torch.randn(max_ctx, nkv * HEAD_DIM, dtype=torch.bfloat16, device=device)
        blk_k = torch.randn(1, nkv * HEAD_DIM, dtype=torch.bfloat16, device=device)
        blk_v = torch.randn(1, nkv * HEAD_DIM, dtype=torch.bfloat16, device=device)
        bias = 0.5 * torch.randn(nq, extent, dtype=torch.bfloat16, device=device)
        step = torch.tensor([ctx_len], dtype=torch.int32, device=device)
        out = torch.zeros(1, nq * HEAD_DIM, dtype=torch.bfloat16, device=device)

        dts = {}
        for tname, t in (("q", q), ("ctx_k", ctx_k), ("ctx_v", ctx_v),
                         ("blk_k", blk_k), ("blk_v", blk_v), ("bias", bias),
                         ("step", step), ("out", out)):
            dts[tname] = pk.attach_input(t, name=f"{name}_{tname}")

        pk.inkling_attention_layer(
            q=dts["q"], ctx_k=dts["ctx_k"], ctx_v=dts["ctx_v"],
            blk_k=dts["blk_k"], blk_v=dts["blk_v"], bias=dts["bias"],
            step=dts["step"], output=dts["out"],
            grid_dim=(grid_x, 1, 1), block_dim=(128, 1, 1),
            sliding_window=sw, extent=extent, head_dim=HEAD_DIM,
            log_scaling_alpha=alpha, log_scaling_n_floor=n_floor,
        )
        cases.append((name, q, ctx_k, ctx_v, blk_k, blk_v, bias, ctx_len,
                      sw, extent, alpha, n_floor, out))

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ok = True
    for (name, q, ctx_k, ctx_v, blk_k, blk_v, bias, ctx_len,
         sw, extent, alpha, n_floor, out) in cases:
        ref = inkling_attention_ref(
            q, ctx_k, ctx_v, blk_k, blk_v, bias, ctx_len, HEAD_DIM,
            sw, extent, alpha, n_floor,
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
    print("PASSED: inkling_attention test_mode produces correct output")


if __name__ == "__main__":
    main()
