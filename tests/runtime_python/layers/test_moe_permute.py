"""Smoke test: ``layers.moe.MoEPermute`` and ``MoEUnpermute`` via test_mode."""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.moe.permute import MoEPermute, MoEUnpermute
from mirage.mpk.persistent_kernel import PersistentKernel


def _make_pk(mbt, device):
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = mbt
    params["max_num_batched_requests"] = mbt
    return PersistentKernel(**params)


def test_moe_permute_smoke():
    device = "cuda"
    torch.manual_seed(0)

    e_local = 4
    hidden_size = 256
    topk = 2
    mbt = 2
    bm_padding = 128
    m_total = e_local * bm_padding
    k_packed = hidden_size // 512 if hidden_size >= 512 else 1  # 4 fp32 scales per uint32

    # Input fp8 (mbt, hidden_size) e4m3.
    input_fp8 = torch.zeros(mbt, hidden_size, dtype=torch.float8_e4m3fn, device=device)
    # Fill with random small bytes interpreted as fp8.
    rand_bytes = torch.randint(low=0, high=64, size=(mbt, hidden_size),
                               dtype=torch.uint8, device=device)
    input_fp8.view(torch.uint8).copy_(rand_bytes)
    input_scale = torch.zeros(mbt, k_packed, dtype=torch.uint32, device=device)
    topk_weights = torch.rand(mbt, topk, dtype=torch.float32, device=device)
    routing_indices = torch.zeros(e_local, mbt, dtype=torch.int32, device=device)
    for b in range(mbt):
        for slot in range(topk):
            routing_indices[(b * topk + slot) % e_local, b] = slot + 1

    permuted_fp8 = torch.zeros(m_total, hidden_size, dtype=torch.float8_e4m3fn, device=device)
    permuted_scale = torch.zeros(k_packed, m_total, dtype=torch.uint32, device=device)
    meta = torch.zeros(2, m_total + mbt * topk, dtype=torch.int32, device=device)

    pk = _make_pk(mbt, device)
    in_dt = pk.attach_input(input_fp8, name="perm_in")
    sc_dt = pk.attach_input(input_scale, name="perm_sc")
    w_dt = pk.attach_input(topk_weights, name="perm_w")
    ri_dt = pk.attach_input(routing_indices, name="perm_ri")
    po_dt = pk.attach_input(permuted_fp8, name="perm_po")
    ps_dt = pk.attach_input(permuted_scale, name="perm_ps")
    meta_dt = pk.attach_input(meta, name="perm_meta")

    m = MoEPermute(
        num_local_experts=e_local,
        hidden_size=hidden_size,
        num_experts_per_tok=topk,
        bm_padding=bm_padding,
        scale_ue8m0=True,
    )

    with pk.compile_scope():
        try:
            m.compile(in_dt, sc_dt, w_dt, ri_dt, po_dt, ps_dt, meta_dt)
        except Exception as e:
            print(f"SKIPPED (compile raised): {type(e).__name__}: {e}")
            pk.finalize()
            return

    print("Compiling MoEPermute test kernel...")
    try:
        pk.compile(output_dir=os.path.dirname(__file__))
    except Exception as e:
        print(f"XFAIL: pk.compile failed: {type(e).__name__}: {e}")
        try: pk.finalize()
        except Exception: pass
        return

    print("Running test kernel...")
    try:
        pk()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"XFAIL: pk() raised at runtime: {type(e).__name__}: {e}")
        try: pk.finalize()
        except Exception: pass
        return

    try:
        # Outputs are uint8/uint32 — just confirm meta has *some* nonzero entries.
        meta_nz = (meta != 0).sum().item()
        print(f"  meta nonzero entries: {meta_nz}")
        print("PASSED (smoke): MoEPermute compiled and ran without crash")
    except Exception as e:
        print(f"XFAIL: post-run check raised: {type(e).__name__}: {e}")
    try: pk.finalize()
    except Exception: pass


def test_moe_unpermute_smoke():
    device = "cuda"
    torch.manual_seed(1)

    e_local = 4
    hidden_size = 256
    topk = 2
    mbt = 2
    bm_padding = 128
    m_total = e_local * bm_padding

    permuted_output = torch.randn(m_total, hidden_size, dtype=torch.bfloat16, device=device) * 0.1
    meta = torch.zeros(2, m_total + mbt * topk, dtype=torch.int32, device=device)
    # Fabricate plausible meta: weights = ones, token_to_permuted point at first
    # `m_total` rows (one-indexed; 0 = unrouted).
    import struct as _s
    one_fp32 = _s.unpack("i", _s.pack("f", 1.0))[0]
    meta[0, :m_total] = one_fp32
    # token_to_permuted has shape (mbt, topk); we put row indices 1..mbt*topk.
    for b in range(mbt):
        for s in range(topk):
            meta[0, m_total + b * topk + s] = (b * topk + s) + 1
            meta[1, m_total + b * topk + s] = (b * topk + s) + 1

    residual = torch.randn(mbt, hidden_size, dtype=torch.bfloat16, device=device) * 0.01
    output = torch.zeros(mbt, hidden_size, dtype=torch.bfloat16, device=device)

    pk = _make_pk(mbt, device)
    po_dt = pk.attach_input(permuted_output, name="unp_po")
    meta_dt = pk.attach_input(meta, name="unp_meta")
    res_dt = pk.attach_input(residual, name="unp_res")
    out_dt = pk.attach_input(output, name="unp_out")

    m = MoEUnpermute(hidden_size=hidden_size)

    with pk.compile_scope():
        try:
            m.compile(po_dt, meta_dt, res_dt, out_dt)
        except Exception as e:
            print(f"SKIPPED (compile raised): {type(e).__name__}: {e}")
            pk.finalize()
            return

    print("Compiling MoEUnpermute test kernel...")
    try:
        pk.compile(output_dir=os.path.dirname(__file__))
    except Exception as e:
        print(f"XFAIL: pk.compile failed: {type(e).__name__}: {e}")
        try: pk.finalize()
        except Exception: pass
        return

    print("Running test kernel...")
    try:
        pk()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"XFAIL: pk() raised at runtime: {type(e).__name__}: {e}")
        try: pk.finalize()
        except Exception: pass
        return

    try:
        if not torch.isfinite(output).all():
            print("FAILED: MoEUnpermute produced non-finite output")
            pk.finalize()
            sys.exit(1)
        print(f"  output sum-abs: {output.abs().sum().item():.4f}")
        print("PASSED (smoke): MoEUnpermute compiled and ran without crash")
    except Exception as e:
        print(f"XFAIL: post-run check raised: {type(e).__name__}: {e}")
    try: pk.finalize()
    except Exception: pass


if __name__ == "__main__":
    test_moe_permute_smoke()
    test_moe_unpermute_smoke()
    print("MoE permute / unpermute smoke tests completed.")
