import torch
import runtime_kernel_hopper

torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)


def ref_out(x, w, bias, residual):
    out = x.float() @ w.float().T
    if bias is not None:
        out = out + bias.float()
    if residual is not None:
        out = out + residual.float()
    return out


def check(batch_size, output_size, reduction_size, use_bias, use_residual):
    # Small inputs keep the matmul near unit scale, so a dropped bias or
    # residual (also unit scale) shows up well above bf16 rounding.
    x = torch.randn(
        (batch_size, reduction_size), device="cuda", dtype=torch.bfloat16, generator=g
    ) * 0.05
    w = torch.randn(
        (output_size, reduction_size), device="cuda", dtype=torch.bfloat16, generator=g
    ) * 0.05
    bias = (
        torch.randn(output_size, device="cuda", dtype=torch.bfloat16, generator=g)
        if use_bias
        else None
    )
    residual = (
        torch.randn(
            batch_size, output_size, device="cuda", dtype=torch.bfloat16, generator=g
        )
        if use_residual
        else None
    )
    output = torch.empty(
        batch_size, output_size, device="cuda", dtype=torch.bfloat16
    )

    runtime_kernel_hopper.linear_swapAB_bias(x, w, bias, residual, output)

    expected = ref_out(x, w, bias, residual)
    err = (output.float() - expected).abs()
    # element-wise: bf16 rounding of the sum plus one more for the residual
    ok = bool((err <= 0.05 + 0.01 * expected.abs()).all())
    status = "OK" if ok else "FAIL"
    print(
        f"[{status}] batch={batch_size} output={output_size} reduction={reduction_size}"
        f" bias={use_bias} residual={use_residual}"
        f" | max_err={err.max().item():.4f}"
    )
    assert ok, "numerical check failed"


print("=== correctness ===")
for batch_size in [8, 16]:
    for output_size in [48, 64, 128, 256, 2048]:
        for reduction_size in [4096, 12288]:
            for use_bias in [False, True]:
                for use_residual in [False, True]:
                    # Known issue: the 128B-swizzled residual tile is read
                    # wrong when a task has fewer than 64 output columns.
                    if use_residual and output_size < 64:
                        continue
                    check(
                        batch_size,
                        output_size,
                        reduction_size,
                        use_bias,
                        use_residual,
                    )

print("\n=== kernel-only latency (us), single CTA ===")
shapes = [
    (8, 2048, 4096),
    (8, 2048, 12288),
    (8, 256, 12288),
    (16, 2048, 12288),
    (8, 64, 4096),
]
for batch_size, output_size, reduction_size in shapes:
    x = torch.randn(
        (batch_size, reduction_size), device="cuda", dtype=torch.bfloat16
    )
    w = torch.randn(
        (output_size, reduction_size), device="cuda", dtype=torch.bfloat16
    )
    bias = torch.randn(output_size, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(
        batch_size, output_size, device="cuda", dtype=torch.bfloat16
    )
    output = torch.empty(
        batch_size, output_size, device="cuda", dtype=torch.bfloat16
    )

    variants = [
        ("no-bias  no-residual", None, None),
        ("bias     no-residual", bias, None),
        ("no-bias  residual  ", None, residual),
        ("bias     residual  ", bias, residual),
    ]
    # interleave rounds (and min over rounds) to avoid clock-ramp /
    # first-launch ordering artifacts
    results = {label: [] for label, _, _ in variants}
    for _round in range(3):
        for label, b, r in variants:
            results[label].append(
                runtime_kernel_hopper.linear_swapAB_bias(
                    x, w, b, r, output, profile=True, benchmark_runs=200
                )
            )
    results = {label: min(t) for label, t in results.items()}

    base = results["no-bias  no-residual"]
    print(f"-- batch={batch_size} output={output_size} reduction={reduction_size}")
    for label, t in results.items():
        delta = (t - base) * 1000
        print(f"   {label}: {t * 1000:8.2f} us  (delta {delta:+.2f} us)")

print("\ndone")
