"""Test-mode coverage for a broadcast bias on the SM100 linear.

A bias is the residual epilogue with a zero row stride, so the CUTE kernel is
unchanged and what needs proving is that every token reads the one stored row
and that the column slicing across tasks stays aligned.

The three shapes are GPT-OSS's real dense projections, including the router's
32-column output -- narrower than any shape in tree, since Qwen3-30B has 128
experts. Each shape is run with and without a bias: the difference must be
exactly the bias, which is what catches a bias that lands on only the first
token or drifts by a task's column offset.

Note the task counts: the SM100 output TMA needs each task's column slice
16-byte aligned, i.e. a multiple of 8 columns. hidden = 2880 over the demo
helper's 96 tasks would give 30 columns and fail, so these use 45 (64 columns)
instead.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

BATCH = 8
HIDDEN = 2880

# (name, out_features, num_tasks); every slice is a multiple of 8 columns
CONFIGS = [
    ("qkv", 5120, 64),      # (64 q + 8 kv + 8 kv) * 64, 80 columns per task
    ("oproj", HIDDEN, 45),  # 64 columns per task
    ("router", 32, 1),      # 32 columns, narrower than one MMA tile
]


def main():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        max_num_batched_tokens=BATCH,
        max_num_batched_requests=1,
    )
    pk = PersistentKernel(**params)

    x = torch.randn(BATCH, HIDDEN, dtype=dtype, device=device) * 0.1
    x_dt = pk.attach_input(x, name="x")

    cases = []
    for name, out_features, num_tasks in CONFIGS:
        assert out_features % num_tasks == 0
        w = torch.randn(out_features, HIDDEN, dtype=dtype, device=device) * 0.05
        # Per-column values, so a bias applied with the wrong column offset
        # cannot pass.
        b = torch.randn(1, out_features, dtype=dtype, device=device)
        w_dt = pk.attach_input(w, name=f"{name}_w")
        b_dt = pk.attach_input(b, name=f"{name}_b")

        out_bias = torch.zeros(BATCH, out_features, dtype=dtype, device=device)
        out_plain = torch.zeros(BATCH, out_features, dtype=dtype, device=device)

        pk.linear_layer(
            input=x_dt, weight=w_dt,
            output=pk.attach_input(out_bias, name=f"{name}_out_bias"),
            grid_dim=(num_tasks, 1, 1), block_dim=(256, 1, 1),
            bias=b_dt,
        )
        pk.linear_layer(
            input=x_dt, weight=w_dt,
            output=pk.attach_input(out_plain, name=f"{name}_out_plain"),
            grid_dim=(num_tasks, 1, 1), block_dim=(256, 1, 1),
        )
        cases.append((name, out_features, num_tasks, w, b, out_bias, out_plain))

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ok = True
    for name, out_features, num_tasks, w, b, out_bias, out_plain in cases:
        cols = out_features // num_tasks
        ref_plain = (x.float() @ w.float().T)
        ref_bias = ref_plain + b.float()

        d_plain = (out_plain.float() - ref_plain).abs().max().item()
        d_bias = (out_bias.float() - ref_bias).abs().max().item()
        # The bias must show up on EVERY token, at the right column.
        delta = (out_bias.float() - out_plain.float())
        d_delta = (delta - b.float()).abs().max().item()
        print(f"[{name}] {out_features} cols over {num_tasks} tasks "
              f"({cols}/task): no-bias {d_plain:.4f}, bias {d_bias:.4f}, "
              f"(bias - no-bias) vs b {d_delta:.4f}")

        tol = 0.05
        if d_plain >= tol:
            print(f"[{name}] FAILED: the no-bias path disagrees with torch")
            ok = False
        if d_bias >= tol:
            print(f"[{name}] FAILED: the bias path disagrees with torch")
            ok = False
        if d_delta >= tol:
            print(f"[{name}] FAILED: the added term is not the bias row")
            ok = False

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("\nPASSED: a broadcast bias reaches every token at the right column, "
          "on all three GPT-OSS dense shapes")


if __name__ == "__main__":
    main()
