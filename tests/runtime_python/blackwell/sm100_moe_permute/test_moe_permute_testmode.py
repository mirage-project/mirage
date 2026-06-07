"""Standalone test for moe_permute_sm100 + moe_unpermute_sm100.

Builds the two-task graph (permute → unpermute) on a synthetic input and
compares outputs to a PyTorch reference. The actual fp8_group_gemm is
NOT exercised — we synthesize the "permuted_output" tensor by casting
permuted_fp8 (first HIDDEN bytes) to bf16. That isolates the two new
peripheral tasks from the upstream GEMM so the no-touch-kernel-core
constraint is preserved end-to-end.

meta layout (must match moe_permute_sm100.cuh):
  meta[0       : M_TOTAL]            = permuted_weights (float32 bits)
  meta[M_TOTAL : M_TOTAL + MBT*TOPK] = token_to_permuted (row + 1; 0 = not
                                       routed locally)
"""

import os

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

device = "cuda"


def make_routing(mbt: int, topk: int, e_local: int, seed: int = 0):
    """Build a synthetic topk routing: each token picks `topk` distinct
    experts uniformly from [0, e_local). Returns:
      topk_weights (mbt, topk) float32
      topk_expert_ids (mbt, topk) int32
      routing_indices (e_local, mbt) int32   # topk_sigmoid output format
                                              # value = topk_slot+1 if routed
                                              #         to that expert, else 0
    """
    g = torch.Generator(device=device).manual_seed(seed)
    topk_expert_ids = torch.zeros((mbt, topk), dtype=torch.int32, device=device)
    for t in range(mbt):
        perm = torch.randperm(e_local, generator=g, device=device)[:topk]
        topk_expert_ids[t] = perm.to(torch.int32)
    topk_weights = torch.rand((mbt, topk), generator=g, dtype=torch.float32,
                              device=device)
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

    routing_indices = torch.zeros((e_local, mbt), dtype=torch.int32,
                                  device=device)
    for t in range(mbt):
        for k in range(topk):
            e = int(topk_expert_ids[t, k].item())
            routing_indices[e, t] = k + 1
    return topk_weights, topk_expert_ids, routing_indices


def python_reference(input_fp8_bytes, input_scale_u32, topk_weights,
                     topk_expert_ids, mbt, topk, e_local, k, k_packed,
                     bm_padding, hidden, residual_bf16):
    """Compute reference outputs assuming the SAME deterministic per-expert
    scan order the kernel uses (single-thread scan over t = 0..mbt-1)."""
    m_total = e_local * bm_padding
    perm_fp8_ref = torch.zeros(m_total, k, dtype=torch.uint8, device=device)
    perm_scale_ref = torch.zeros(k_packed, m_total, dtype=torch.uint32,
                                 device=device)
    perm_weights_ref = torch.zeros(m_total, dtype=torch.float32, device=device)
    tok_to_perm_ref = torch.zeros((mbt, topk), dtype=torch.int32, device=device)

    expert_counts = [0] * e_local
    for e in range(e_local):
        # Same scan order as the kernel: sequential over t = 0..mbt-1.
        for t in range(mbt):
            for k_slot in range(topk):
                if int(topk_expert_ids[t, k_slot].item()) == e:
                    slot = expert_counts[e]
                    if slot >= bm_padding:
                        continue
                    row = e * bm_padding + slot
                    perm_fp8_ref[row] = input_fp8_bytes[t]
                    perm_scale_ref[:, row] = input_scale_u32[t]
                    perm_weights_ref[row] = topk_weights[t, k_slot]
                    tok_to_perm_ref[t, k_slot] = row + 1
                    expert_counts[e] += 1

    # Synthetic "GEMM": cast permuted_fp8[:, :HIDDEN] → bf16.
    perm_output_ref = perm_fp8_ref[:, :hidden].to(torch.float32).to(
        torch.bfloat16).contiguous()

    # Combine reference.
    out_ref = residual_bf16.to(torch.float32).clone()
    for t in range(mbt):
        for k_slot in range(topk):
            row_1idx = int(tok_to_perm_ref[t, k_slot].item())
            if row_1idx <= 0:
                continue
            row = row_1idx - 1
            w = float(perm_weights_ref[row].item())
            out_ref[t] += w * perm_output_ref[row].to(torch.float32)
    out_ref = out_ref.to(torch.bfloat16)

    return (perm_fp8_ref, perm_scale_ref, perm_weights_ref, tok_to_perm_ref,
            perm_output_ref, out_ref)


def test_moe_permute_unpermute_testmode():
    torch.manual_seed(42)
    MBT = 8
    TOPK = 2
    E_LOCAL = 4
    K = 2048
    nk = (K + 127) // 128
    K_PACKED = (nk + 3) // 4
    HIDDEN = 128
    BM_PADDING = 128
    M_TOTAL = E_LOCAL * BM_PADDING

    # Inputs.
    input_fp8 = torch.randint(0, 256, (MBT, K), dtype=torch.uint8,
                              device=device)
    input_scale = torch.randint(0, 0x7FFFFFFF, (MBT, K_PACKED),
                                dtype=torch.int32, device=device).view(
                                    torch.uint32)
    topk_weights, topk_expert_ids, routing_indices = make_routing(
        MBT, TOPK, E_LOCAL, seed=0)

    residual = torch.randn((MBT, HIDDEN), dtype=torch.bfloat16, device=device)

    # Reference.
    (perm_fp8_ref, perm_scale_ref, perm_weights_ref, tok_to_perm_ref,
     perm_output_ref, out_ref) = python_reference(
        input_fp8, input_scale, topk_weights, topk_expert_ids,
        MBT, TOPK, E_LOCAL, K, K_PACKED, BM_PADDING, HIDDEN, residual)

    # MPK output buffers.
    permuted_fp8 = torch.zeros((M_TOTAL, K), dtype=torch.uint8, device=device)
    permuted_scale = torch.zeros((K_PACKED, M_TOTAL), dtype=torch.uint32,
                                 device=device)
    # meta = M_TOTAL (weights as int32 bits) + MBT*TOPK (token_to_permuted).
    # MUST be zero-init before each iter so token_to_permuted's sentinel
    # value (0 = not routed locally) is correct.
    META_LEN = M_TOTAL + MBT * TOPK
    meta = torch.zeros((META_LEN,), dtype=torch.int32, device=device)

    permuted_output = perm_output_ref.clone()  # stand-in for real GEMM
    output = torch.zeros((MBT, HIDDEN), dtype=torch.bfloat16, device=device)

    nw, nsch = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=nw,
        num_local_schedulers=nsch,
        max_num_batched_tokens=MBT,
        max_num_batched_requests=MBT,
        max_seq_length=1,
    )
    pk = PersistentKernel(**params)

    in_fp8_dt = pk.attach_input(input_fp8, name="input_fp8")
    in_scale_dt = pk.attach_input(input_scale, name="input_scale")
    topk_w_dt = pk.attach_input(topk_weights, name="topk_weights")
    routing_dt = pk.attach_input(routing_indices, name="routing_indices")
    perm_fp8_dt = pk.attach_input(permuted_fp8, name="permuted_fp8")
    perm_scale_dt = pk.attach_input(permuted_scale, name="permuted_scale")
    meta_dt = pk.attach_input(meta, name="meta")
    perm_out_dt = pk.attach_input(permuted_output, name="permuted_output")
    res_dt = pk.attach_input(residual, name="residual")
    out_dt = pk.attach_input(output, name="output")

    pk.moe_permute_sm100_layer(
        input_fp8=in_fp8_dt,
        input_scale=in_scale_dt,
        topk_weights=topk_w_dt,
        routing_indices=routing_dt,
        permuted_fp8=perm_fp8_dt,
        permuted_scale=perm_scale_dt,
        meta=meta_dt,
        bm_padding=BM_PADDING,
    )

    pk.moe_unpermute_sm100_layer(
        permuted_output=perm_out_dt,
        meta=meta_dt,
        residual=res_dt,
        output=out_dt,
    )

    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    pk()
    torch.cuda.synchronize()

    # Decode meta.
    permuted_weights = meta[:M_TOTAL].view(torch.float32)
    token_to_permuted = meta[M_TOTAL:].view(MBT, TOPK)

    # 1) token_to_permuted matches reference.
    if not (token_to_permuted == tok_to_perm_ref).all():
        bad = (token_to_permuted != tok_to_perm_ref).nonzero()
        raise AssertionError(
            f"token_to_permuted mismatch at {bad[:5].tolist()}: "
            f"kernel={token_to_permuted}, ref={tok_to_perm_ref}")

    # 2) permuted_weights matches reference.
    if not torch.allclose(permuted_weights, perm_weights_ref, atol=1e-6):
        diff = (permuted_weights - perm_weights_ref).abs().max().item()
        raise AssertionError(f"permuted_weights mismatch: max_diff={diff}")

    # 3) permuted_fp8: only check routed rows (padding rows are undefined).
    for t in range(MBT):
        for k_slot in range(TOPK):
            row_1idx = int(tok_to_perm_ref[t, k_slot].item())
            if row_1idx <= 0:
                continue
            row = row_1idx - 1
            mismatch = (permuted_fp8[row] != input_fp8[t]).sum().item()
            assert mismatch == 0, (
                f"permuted_fp8 row {row} (from token {t}, slot {k_slot}) "
                f"differs in {mismatch} bytes")

    # 4) permuted_scale (transposed): same idea.
    for t in range(MBT):
        for k_slot in range(TOPK):
            row_1idx = int(tok_to_perm_ref[t, k_slot].item())
            if row_1idx <= 0:
                continue
            row = row_1idx - 1
            for sf in range(K_PACKED):
                assert int(permuted_scale[sf, row].item()) == \
                       int(input_scale[t, sf].item()), \
                       f"permuted_scale[{sf}, {row}] != input_scale[{t}, {sf}]"

    # 5) unpermute output.
    max_diff = (output.to(torch.float32) - out_ref.to(torch.float32)).abs().max().item()
    print(f"unpermute output max_diff: {max_diff}")
    torch.testing.assert_close(output, out_ref, rtol=1e-2, atol=1e-2)

    print("PASSED")
    pk.finalize()


if __name__ == "__main__":
    test_moe_permute_unpermute_testmode()
