"""MoE permute <-> unpermute round-trip test (DSV3 NEW-MoE path).

Validates the moe_permute_sm100_layer -> moe_unpermute_sm100_layer pair end to
end through the full MPK compilation pipeline. Both kernels are chained in ONE
task graph:

  1. moe_permute_sm100_layer scans a KNOWN routing (topk_weights +
     routing_indices) and packs the per-row metadata into the shared `meta`
     buffer:
       meta[0       : M_TOTAL]            = permuted_weights (f32 bits)
       meta[M_TOTAL : M_TOTAL + MBT*TOPK] = token_to_permuted (row + 1; 0 = not
                                            routed locally)
     (It also emits permuted_fp8 / permuted_scale — produced but irrelevant to
     the round-trip, which only checks the recombination.)

  2. moe_unpermute_sm100_layer reads that `meta` plus a KNOWN per-row
     permuted_output (a random bf16 marker, standing in for the W2 GEMM output)
     and a shared_residual, and writes
       output[t] = residual[t]
                 + sum_k(weights[t,k] * permuted_output[token_to_permuted[t,k]-1])

The PyTorch reference decodes the SAME routing identically to the permute
kernel's deterministic token-order scan (per expert, tokens gathered in
increasing token index -> consecutive permuted rows e*BM_PADDING + slot), then
recombines the SAME bf16 `permuted_output` rows. So the comparison is the exact
inverse the kernels implement.

Meta decode (matches moe_permute_sm100.cuh + moe_unpermute_sm100.cuh):
  - For local expert e, scan tokens t=0..num_active-1; if routing[e,t] > 0 the
    token claims permuted row e*BM_PADDING + (running slot count), with topk
    slot k = routing[e,t]-1.
  - permuted_weights[row] = topk_weights[t, k];  token_to_permuted[t,k] = row+1.
  - unpermute: output[t] = residual[t] + sum_k weights[t,k]*permuted_output[row].

EL reduction (LOGGED): real DSV3 ep=1 has E_LOCAL = num_local_experts = 256, so
M_TOTAL = 256*128 = 32768 permuted rows. The round-trip is correctness-only and
per-row independent, so E_LOCAL is reduced to a small value (>= TOPK so every
token's 8 experts fit in distinct local slots) to keep M_TOTAL / compile time
tractable. Sweep covers EL in {8, 16} and MBT (active tokens) in {1, 4, 8, 16}.
HIDDEN is the real DSV3 7168. Tolerance bf16 atol/rtol = 1e-2.

Run:
    python tests/runtime_python/blackwell/sm100_moe/test_permute_roundtrip_testmode.py
"""

import os

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from sm100_fp8_scale_layout import (  # noqa: E402
    quantize_to_fp8_deepgemm_style,
    packed_scale_k_for_reduction_size,
)

HIDDEN = 7168
TOPK = 8
BM_PADDING = 128

# Each entry: (E_LOCAL, MBT). E_LOCAL >= TOPK so a token's 8 experts get
# distinct local slots. MBT = number of (active) tokens.
MATRIX = [
    (8, 1),
    (8, 4),
    (16, 8),
    (16, 16),
]


def _build_known_routing(mbt, e_local, topk, seed):
    """Generate a known routing: each token picks `topk` distinct local experts.

    Returns:
      topk_weights   (mbt, topk)   f32  — per-slot routing weights.
      routing_indices(e_local, mbt) i32 — routing[e,t] = slot_1idx (1-indexed)
                                          if token t routed slot to expert e,
                                          else 0. Matches topk_sigmoid output.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    topk_weights = torch.zeros(mbt, topk, dtype=torch.float32)
    routing_indices = torch.zeros(e_local, mbt, dtype=torch.int32)
    for t in range(mbt):
        # topk distinct experts for this token, in arbitrary slot order.
        experts = torch.randperm(e_local, generator=g)[:topk]
        w = torch.rand(topk, generator=g) + 0.1
        w = (w / w.sum()) * 2.5  # DSV3 routed_scaling_factor.
        for k in range(topk):
            e = int(experts[k].item())
            topk_weights[t, k] = w[k]
            routing_indices[e, t] = k + 1  # 1-indexed slot.
    return topk_weights, routing_indices


def _decode_meta_reference(topk_weights, routing_indices, mbt, e_local,
                           num_active):
    """Decode routing exactly like moe_permute_sm100.cuh's token-order scan.

    Returns:
      tok_to_perm (mbt, topk) int  — permuted row+1 for each (token, slot),
                                     0 if that slot's expert is not routed.
      perm_weight (M_TOTAL,)  f32  — permuted_weights[row] (0 for padding).
    """
    m_total = e_local * BM_PADDING
    tok_to_perm = torch.zeros(mbt, TOPK, dtype=torch.int64)
    perm_weight = torch.zeros(m_total, dtype=torch.float32)
    for e in range(e_local):
        row_base = e * BM_PADDING
        slot = 0
        for t in range(min(num_active, mbt)):
            slot_1idx = int(routing_indices[e, t].item())
            if slot_1idx > 0 and slot < BM_PADDING:
                row = row_base + slot
                k = slot_1idx - 1
                tok_to_perm[t, k] = row + 1
                perm_weight[row] = float(topk_weights[t, k].item())
                slot += 1
    return tok_to_perm, perm_weight


def _run_case(e_local, mbt, seed=42):
    device = "cuda"
    m_total = e_local * BM_PADDING
    k_packed = packed_scale_k_for_reduction_size(HIDDEN)

    print(f"\n{'='*60}")
    print(f"permute<->unpermute roundtrip: E_LOCAL={e_local} MBT={mbt} "
          f"TOPK={TOPK} HIDDEN={HIDDEN} M_TOTAL={m_total} K_PACKED={k_packed}")
    print(f"{'='*60}")

    # --- Known routing + reference meta decode ---
    topk_weights_cpu, routing_indices_cpu = _build_known_routing(
        mbt, e_local, TOPK, seed)
    tok_to_perm, perm_weight = _decode_meta_reference(
        topk_weights_cpu, routing_indices_cpu, mbt, e_local, num_active=mbt)

    topk_weights = topk_weights_cpu.to(device)
    routing_indices = routing_indices_cpu.to(device)

    # --- Permute inputs: FP8 activation + UE8M0-packed scale ---
    # The activation values don't affect the round-trip (we feed a separate
    # known permuted_output into unpermute), but permute requires a valid FP8
    # input + packed scale in the producer (column-major) layout.
    act_bf16 = torch.randn(mbt, HIDDEN, dtype=torch.bfloat16, device=device)
    act_fp8, act_scale_dg = quantize_to_fp8_deepgemm_style(act_bf16)
    input_fp8 = act_fp8.view(torch.uint8).contiguous()
    # quantize_fp8 writes the packed scale column-major [K_PACKED, MBT_ALIGNED]
    # (out[sf*MBT_ALIGNED + t]); the deepgemm-style helper already produces a
    # (MBT, K_PACKED) tensor with stride (1, aligned_MBT), i.e. exactly that
    # column-major byte layout. moe_permute reads it column-major.
    input_scale = act_scale_dg  # (MBT, K_PACKED) u32, col-major stride.

    # --- Permute outputs ---
    permuted_fp8 = torch.zeros(m_total, HIDDEN, dtype=torch.uint8, device=device)
    permuted_scale = torch.zeros(k_packed, m_total, dtype=torch.uint32,
                                 device=device)
    meta_len = m_total + mbt * TOPK
    meta = torch.zeros(2, meta_len, dtype=torch.int32, device=device)

    # --- Unpermute inputs: KNOWN per-row permuted_output marker + residual ---
    # Random bf16 marker — the reference recombines the SAME bf16 rows, so any
    # values work and the comparison is exact-inverse.
    permuted_output = torch.randn(m_total, HIDDEN, dtype=torch.bfloat16,
                                  device=device)
    shared_residual = torch.randn(mbt, HIDDEN, dtype=torch.bfloat16, device=device)
    output = torch.zeros(mbt, HIDDEN, dtype=torch.bfloat16, device=device)

    # --- PyTorch reference for the recombination ---
    ref = shared_residual.float().clone()
    for t in range(mbt):
        for k in range(TOPK):
            row_1idx = int(tok_to_perm[t, k].item())
            if row_1idx > 0:
                w = perm_weight[row_1idx - 1].item()
                ref[t] += w * permuted_output[row_1idx - 1].float()
    ref = ref.to(torch.bfloat16)

    # --- Build PersistentKernel (chain permute -> unpermute) ---
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = mbt
    params["max_num_batched_requests"] = mbt
    params["max_seq_length"] = max(mbt, 1)
    params["max_num_pages"] = max(mbt, 1)
    # Single prefill request of length MBT -> num_active_rows (=
    # qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]) == MBT, so the permute
    # scan covers all tokens and unpermute writes all MBT rows.
    params["meta_tensors"] = {
        "prompt_lengths": torch.tensor([mbt], dtype=torch.int32, device=device),
    }
    pk = PersistentKernel(**params)

    input_fp8_dt = pk.attach_input(input_fp8, name="input_fp8")
    input_scale_dt = pk.attach_input(input_scale, name="input_scale")
    topk_weights_dt = pk.attach_input(topk_weights, name="topk_weights")
    routing_indices_dt = pk.attach_input(routing_indices, name="routing_indices")
    permuted_fp8_dt = pk.attach_input(permuted_fp8, name="permuted_fp8")
    permuted_scale_dt = pk.attach_input(permuted_scale, name="permuted_scale")
    meta_dt = pk.attach_input(meta, name="meta")
    permuted_output_dt = pk.attach_input(permuted_output, name="permuted_output")
    residual_dt = pk.attach_input(shared_residual, name="residual")
    output_dt = pk.attach_input(output, name="output")

    # Zero-init the meta buffer (tok_to_perm region MUST start at 0 — the
    # permute kernel only writes routed slots; mirrors builder L3242).
    pk.tensor_init_layer(
        target=meta_dt,
        dummy=input_fp8_dt,
        grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
        dummy_input_map=(-1, -1, -1),
        target_input_map=(-1, -1, -1),
    )

    pk.moe_permute_sm100_layer(
        input_fp8=input_fp8_dt,
        input_scale=input_scale_dt,
        topk_weights=topk_weights_dt,
        routing_indices=routing_indices_dt,
        permuted_fp8=permuted_fp8_dt,
        permuted_scale=permuted_scale_dt,
        meta=meta_dt,
        bm_padding=BM_PADDING,
        e_per_cta=1,
    )

    pk.moe_unpermute_sm100_layer(
        permuted_output=permuted_output_dt,
        meta=meta_dt,
        residual=residual_dt,
        output=output_dt,
        rows_per_cta=1,
        hidden_split=1,
    )

    folder_path = os.path.dirname(os.path.abspath(__file__))
    pk.compile(output_dir=folder_path)
    pk()
    torch.cuda.synchronize()

    # --- Verify the kernel-written meta matches the Python decode bit-for-bit
    #     (the heart of the round-trip: identical token<->row mapping). ---
    meta_host = meta.cpu()
    k_t2p = meta_host[0, m_total:m_total + mbt * TOPK].view(mbt, TOPK).to(torch.int64)
    k_weight = meta_host[0, 0:m_total].view(torch.float32)
    t2p_match = bool(torch.equal(k_t2p, tok_to_perm))
    w_match = bool(torch.allclose(k_weight, perm_weight, atol=0, rtol=0))
    print(f"  meta token_to_permuted match: {t2p_match}")
    print(f"  meta permuted_weights match:  {w_match}")

    max_diff = (output.float() - ref.float()).abs().max().item()
    try:
        torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
        out_ok = True
    except AssertionError as e:
        out_ok = False
        print(e)
    ok = out_ok and t2p_match and w_match
    print(f"  output max_diff={max_diff:.6g}  {'PASS' if ok else 'FAIL'}")

    pk.finalize()
    return ok, max_diff, t2p_match, w_match


def main():
    results = []
    for e_local, mbt in MATRIX:
        ok, md, t2p, w = _run_case(e_local, mbt)
        results.append((e_local, mbt, ok, md, t2p, w))

    n_pass = sum(1 for r in results if r[2])
    print(f"\n{'='*60}")
    print("permute<->unpermute roundtrip matrix summary")
    print(f"{'='*60}")
    for e_local, mbt, ok, md, t2p, w in results:
        print(f"  E_LOCAL={e_local:<3} MBT={mbt:<3} "
              f"t2p={t2p} w={w} out_max_diff={md:.6g}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"\n{n_pass}/{len(results)} PASS")
    if n_pass == len(results):
        print("ALL PASS")
    else:
        raise AssertionError(
            f"permute_roundtrip: {len(results)-n_pass} config(s) FAILED")


def test_permute_unpermute_roundtrip():
    main()


if __name__ == "__main__":
    main()
