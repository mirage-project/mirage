"""BIT-EXACTNESS gate for the M4-I2 ferret port of the dense FP8 block-scale GEMM.

`linear_fp8_blockscale_sm100.cuh` now holds two implementations: the original
(`..._task_impl_golden`, body preserved byte-for-byte) and the ferret
`dense-fp8-blockscale` winner from workspace4 tag v011 (`..._task_impl_fast`),
which stages the whole-K activation tile and the fp32 scale panels in shared
memory, streams B through a deep per-warp cp.async ring, loads fragments with
`ldmatrix`, interleaves paired K tiles, and -- the structural change -- accepts a
per-task N slice FINER than the checkpoint's 128-row scale block.

Every one of those transformations is supposed to move the SAME bytes through the
SAME fp32 expressions in the SAME order. This test is the falsifier: for each
shipped Qwen3.5 dense projection and each decode batch size, compute the whole
projection twice from identical inputs -- once as N/128 golden tasks, once as
N/slice fast tasks -- and require the bf16 outputs to be byte-identical.

Byte-identical, not close: a tolerance test cannot distinguish "the promotion
order is preserved" from "the scales are being applied slightly differently",
which is the only failure mode that matters here.

Both nvcc lanes must pass, because the megakernel ships -use_fast_math and this
extension does not build with it by default:

    python setup.py build_ext --inplace                      # no-fast-math lane
    FP8BS_TEST_FAST_MATH=1 python setup.py build_ext --inplace   # shipped lane

Run:  python test_linear_fp8_blockscale_bitexact.py
"""

import sys

import torch

import runtime_kernel_blackwell_linear_fp8_blockscale as linear_kernel

BLOCK = 128
FP8_MAX = 448.0
EPS = 1e-10

# (N, K, WITH_RESIDUAL, per-shape N_SLICE, label). Must stay in step with
# builder.py's FP8_DENSE_N_SLICE and the wrapper's FOR_EACH_SHIPPED_SHAPE.
SHIPPED = [
    (8192, 2048, False, 64, "gdn_in_proj_qkv"),
    (4096, 2048, False, 32, "gdn_in_proj_z"),
    (9216, 2048, False, 64, "attn_qkvg_proj"),
    (2048, 4096, True, 16, "out_proj/o_proj"),
    (1024, 2048, False, 32, "shared_gate_up"),
    (2048, 512, False, 64, "shared_down"),
]
BATCH_SIZES = [1, 2, 4, 8, 16]


def quantize_activation(x_bf16):
    m, k = x_bf16.shape
    xf = x_bf16.float().reshape(m, k // BLOCK, BLOCK)
    absmax = xf.abs().amax(dim=-1).clamp(min=EPS)
    scale = absmax / FP8_MAX
    q = (xf / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return q.reshape(m, k).to(torch.float8_e4m3fn), scale.contiguous()


def quantize_weight_blocks(w_bf16):
    n, k = w_bf16.shape
    wf = w_bf16.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    absmax = wf.abs().amax(dim=(1, 3)).clamp(min=EPS)
    scale = absmax / FP8_MAX
    q = (wf / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    return q.reshape(n, k).to(torch.float8_e4m3fn), scale.contiguous()


def run_pair(batch_size, n, k, has_residual, generator):
    """Return (golden_bits, fast_bits) for one projection at one batch size."""
    x = torch.randn((batch_size, k), device="cuda", dtype=torch.bfloat16,
                    generator=generator)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16,
                    generator=generator)
    x_q, x_s = quantize_activation(x)
    w_q, w_s = quantize_weight_blocks(w)
    residual = (torch.randn((batch_size, n), device="cuda",
                            dtype=torch.bfloat16, generator=generator)
                if has_residual else None)

    outs = []
    for force_golden in (True, False):
        # 0xEE-filled so a task that never writes its slice cannot pass by
        # accidentally matching a zeroed buffer.
        out = torch.empty((batch_size, n), device="cuda", dtype=torch.bfloat16)
        out.view(torch.uint8).fill_(0xEE)
        linear_kernel.linear_fp8_blockscale_projection(
            x_q, x_s, w_q, w_s, residual, out, force_golden=force_golden)
        torch.cuda.synchronize()
        outs.append(out.view(torch.uint16).clone())
    return outs[0], outs[1]


def main():
    assert torch.cuda.is_available(), "needs a Blackwell GPU"
    fails = []
    print(f"{'shape':<18} {'N':>5} {'K':>5} {'res':>4} {'slice':>5} "
          f"{'bs':>3}  verdict")
    for n, k, has_residual, n_slice, label in SHIPPED:
        for bs in BATCH_SIZES:
            g = torch.Generator(device="cuda")
            g.manual_seed(0x5EED0000 + n + k + bs)
            golden, fast = run_pair(bs, n, k, has_residual, g)
            bad = int((golden != fast).sum().item())
            # Neither arm may leave the 0xEE poison behind.
            unwritten = int((golden == 0xEEEE).sum().item()) + \
                int((fast == 0xEEEE).sum().item())
            ok = bad == 0 and unwritten == 0
            print(f"{label:<18} {n:>5} {k:>5} {str(has_residual):>4} "
                  f"{n_slice:>5} {bs:>3}  "
                  f"{'BIT-EXACT' if ok else f'MISMATCH {bad} elems'}"
                  f"{'' if unwritten == 0 else f' UNWRITTEN {unwritten}'}")
            if not ok:
                fails.append((label, bs, bad, unwritten))
                idx = (golden != fast).nonzero()[:3]
                for i in idx:
                    r, c = int(i[0]), int(i[1])
                    print(f"    [{r},{c}] golden=0x{int(golden[r, c]):04x} "
                          f"fast=0x{int(fast[r, c]):04x}")

    if fails:
        print(f"\nFAIL: {len(fails)} shape/bs pairs are not bit-exact")
        return 1
    print(f"\nPASS: {len(SHIPPED) * len(BATCH_SIZES)} shape/bs pairs "
          f"bit-exact (ferret v011 fast path == golden path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
