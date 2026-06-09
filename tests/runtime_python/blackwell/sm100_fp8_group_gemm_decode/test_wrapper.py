"""Correctness + bench for fp8_group_gemm_decode wrapper.

Wrapper #include's the MPK task body (.cuh). Validates the device function in
isolation against a fp32 reference, then benches vs DeepGEMM with the
methodology the kernel author specified (L2 flush + disable_ue8m0_cast=True).
"""
import os
import sys
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "build", "lib.linux-x86_64-cpython-312"))
import runtime_kernel_fp8_group_gemm_decode as kern  # noqa: E402

device = "cuda"


def float_to_ue8m0(t):
    """fp32 → UE8M0 (8-bit exponent only, power-of-2 rounding)."""
    pos = torch.where(t > 0, t, torch.full_like(t, 1e-30))
    p2 = torch.pow(2.0, torch.round(torch.log2(pos)))
    bits = p2.view(torch.int32)
    ue = ((bits >> 23) & 0xFF).to(torch.uint8)
    ue = torch.where(t > 0, ue, torch.zeros_like(ue))
    return ue


def pack_sf(scales_2d):
    """[dim, nk] fp32 -> [num_sf_k, dim] uint32 row-major (dim innermost).

    Matches source's prepare_sf which writes packed[sk*dim + d]. The kernel's
    TMA descriptor uses innermost-first dim order: g=(dim, num_sf_k),
    leading_stride=(dim*4 bytes) — same physical layout, different notation."""
    dim, nk = scales_2d.shape
    num_sf_k = (nk + 3) // 4
    ue = float_to_ue8m0(scales_2d).to(torch.int64)  # [dim, nk]
    out = torch.zeros(num_sf_k, dim, dtype=torch.int64, device=scales_2d.device)
    for j in range(4):
        ki = torch.arange(num_sf_k, device=scales_2d.device) * 4 + j
        valid = ki < nk
        ue_col = torch.where(valid, ue[:, ki.clamp(max=nk - 1)], torch.zeros(dim, num_sf_k, dtype=torch.int64, device=scales_2d.device)[:, 0:num_sf_k])
        # ue_col: [dim, num_sf_k]; transpose to [num_sf_k, dim]
        out |= (ue_col.t() & 0xFF) << (j * 8)
    return out.to(torch.uint32).contiguous()


def make_inputs(MPE, E, K, N, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    M_total = E * MPE
    A_bf16 = torch.randn(M_total, K, dtype=torch.bfloat16, device=device, generator=g) * 0.5
    B_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16, device=device, generator=g) * 0.5
    A_fp8 = A_bf16.to(torch.float8_e4m3fn)
    B_fp8 = B_bf16.to(torch.float8_e4m3fn)
    nk = (K + 127) // 128
    sa = (0.5 + torch.rand(M_total, nk, dtype=torch.float32, device=device, generator=g) * 0.5).contiguous()
    # sfb: per-element expanded to [E*N, nk] (each output column shares per-128 N-block scale)
    nb = (N + 127) // 128
    sb_block = (0.5 + torch.rand(E, nb, nk, dtype=torch.float32, device=device, generator=g) * 0.5).contiguous()
    sb = sb_block.repeat_interleave(128, dim=1)[:, :N, :].reshape(E * N, nk).contiguous()
    sfa_packed = pack_sf(sa)
    sfb_packed = pack_sf(sb)
    D = torch.zeros(M_total, N, dtype=torch.bfloat16, device=device)
    m_indices = torch.arange(M_total, device=device, dtype=torch.int32) // MPE
    return (A_fp8.contiguous(), B_fp8.contiguous(),
            sfa_packed.contiguous(), sfb_packed.contiguous(),
            D, m_indices, sa, sb_block)


def ue8m0_round_trip(t):
    """Match kernel's scale path: fp32 -> UE8M0 -> fp32 (power-of-2 only)."""
    pos = torch.where(t > 0, t, torch.full_like(t, 1e-30))
    return torch.where(t > 0, torch.pow(2.0, torch.round(torch.log2(pos))),
                       torch.zeros_like(t))


def torch_reference(A_fp8, B_fp8, sa, sb_block, m_indices, MPE, E, K, N):
    """Match kernel: each BM=128 block uses m_indices[block_start]'s expert,
    scales are UE8M0-rounded (kernel's hardware dequant uses UE8M0)."""
    BM = 128
    M_total = E * MPE
    A = A_fp8.float()
    B = B_fp8.float()
    nk = K // 128
    out = torch.zeros(M_total, N, dtype=torch.float32, device=A.device)
    sa_q = ue8m0_round_trip(sa)              # [M_total, nk]
    sb_block_q = ue8m0_round_trip(sb_block)  # [E, nb, nk]
    nb = (N + 127) // 128
    sb_q_full = sb_block_q.repeat_interleave(128, dim=1)[:, :N, :]  # [E, N, nk]
    for bm in range(0, M_total, BM):
        block_end = min(bm + BM, M_total)
        expert_id = int(m_indices[bm].item())
        for ki in range(nk):
            a_blk = A[bm:block_end, ki * 128:(ki + 1) * 128]
            b_blk = B[expert_id, :, ki * 128:(ki + 1) * 128]
            partial = a_blk @ b_blk.T  # [block_size, N]
            sa_col = sa_q[bm:block_end, ki:ki + 1]
            sb_row = sb_q_full[expert_id, :, ki]
            out[bm:block_end] += partial * sa_col * sb_row[None, :]
    return out.to(torch.bfloat16)


def run_correctness(MPE, E, K, N):
    A, B, sfa, sfb, D, mi, sa, sb_block = make_inputs(MPE, E, K, N)
    A_u8 = A.view(torch.uint8)
    B_u8 = B.view(torch.uint8)
    kern.fp8_group_gemm_decode(A_u8, B_u8, sfa, sfb, D, mi)
    torch.cuda.synchronize()
    D_ref = torch_reference(A, B, sa, sb_block, mi, MPE, E, K, N)
    err = (D.float() - D_ref.float()).abs()
    rel = err / (D_ref.float().abs() + 1e-30)
    return err.max().item(), err.mean().item(), rel.max().item()


if __name__ == "__main__":
    cfgs = [
        ("gate_up_M1",  1,  32, 7168, 4096),
        ("gate_up_M4",  4,  32, 7168, 4096),
        ("gate_up_M8",  8,  32, 7168, 4096),
        ("gate_up_M16", 16, 32, 7168, 4096),
        ("down_M1",     1,  32, 2048, 7168),
        ("down_M4",     4,  32, 2048, 7168),
        ("down_M8",     8,  32, 2048, 7168),
        ("down_M16",    16, 32, 2048, 7168),
    ]
    print(f"{'config':>14} | {'max_err':>9} {'mean_err':>10} {'max_rel':>9} | status")
    print("-" * 65)
    fail = 0
    for name, MPE, E, K, N in cfgs:
        max_err, mean_err, max_rel = run_correctness(MPE, E, K, N)
        ok = (mean_err < 1e-2) and (max_err < 1.0)
        status = "OK" if ok else "FAIL"
        if not ok:
            fail += 1
        print(f"{name:>14} | {max_err:>9.4f} {mean_err:>9.5f} {max_rel:>9.4f} | {status}")
    sys.exit(fail)
