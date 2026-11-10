"""Kimi K3 MoE weights from SGLang's loaded parameters (this rank's shard, the checkpoint layout, before AITER's
process_weights_after_loading) to the layer kernel's (amd/tests/test_layer_one_gpu.py names and layouts):
  routed experts  w13 [E][2 IR][L/2] (gate rows, then up rows; two e2m1 per byte, the lower K in the low nibble), its E8M0 scales
                  [E][2 IR][L/32]; w2 [E][L][IR/2], scales [E][L][IR/32]
                  -> the gfx950 tile layout (amd/tools/shuffle_weights.py; W13 rows in item order, w13_rows)
  shared gate_up  [gate; up] rows -> our blocks of 128 (64 gate features, then their 64 up features)
The inverse of amd/bench/sglang_moe_layer.py's checkpoint_banks / shared_gate_up_rows (which matched SGLang's routing and output
on the same weights, 10-13); --self-test checks the round trip on the CPU.

python amd/serving/k3_moe_weights.py --self-test
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def unpack(b: torch.Tensor) -> torch.Tensor:
    """bytes [..][K/2] -> e2m1 codes [..][K] (the lower K from the low nibble)"""
    return torch.stack((b & 0xF, b >> 4), dim=-1).reshape(*b.shape[:-1], b.shape[-1] * 2)


def tiles_batched(codes: torch.Tensor) -> torch.Tensor:
    """e2m1 codes [E][N][K] -> tile bytes [E][N/16][K/128][64][16] (lane l: row 16 rt + l % 16, K 128 ks + 32 (l / 16) .. + 31,
    the lower K in the low nibble); the inverse of sglang_moe_layer.codes_batched"""
    e, n, k = codes.shape
    v = codes.reshape(e, n // 16, 16, k // 128, 4, 32).permute(0, 1, 3, 4, 2, 5)          # [e][rt][ks][g][r][32]
    v = v.reshape(e, n // 16, k // 128, 64, 16, 2)
    return (v[..., 0] | (v[..., 1] << 4)).to(torch.uint8).contiguous()


def scale_tiles_batched(s: torch.Tensor) -> torch.Tensor:
    """E8M0 scales [E][N][K/32] -> the gfx950 scale layout (bytes, same count); the inverse of sglang_moe_layer.scales_batched"""
    e, n, kb = s.shape
    ks = kb // 4
    if ks % 4 == 0:
        return s.reshape(e, n // 16, 16, ks // 4, 4, 4).permute(0, 1, 3, 5, 2, 4).contiguous().reshape(e, -1)
    return s.reshape(e, n // 64, 4, 16, ks, 4).permute(0, 1, 5, 3, 2, 4).contiguous().reshape(e, -1)


def w13_inverse_order(ir: int, device) -> torch.Tensor:
    """ours = checkpoint[inv]: checkpoint row i (gate rows, then up rows) goes to our item-order row order[i]"""
    f = torch.arange(ir, device=device)
    gate = 64 * (f // 32) + f % 32
    order = torch.cat((gate, gate + 32))
    inv = torch.empty_like(order)
    inv[order] = torch.arange(2 * ir, device=device)
    return inv


def shared_inverse_order(shr: int, device) -> torch.Tensor:
    """ours = sglang[inv] for the shared gate_up rows (ours: blocks of 128 = 64 gate features, then their 64 up features)"""
    f = torch.arange(shr, device=device)
    gate = 128 * (f // 64) + f % 64
    order = torch.cat((gate, gate + 64))
    inv = torch.empty_like(order)
    inv[order] = torch.arange(2 * shr, device=device)
    return inv


def routed_experts(w13: torch.Tensor, w13_s: torch.Tensor, w2: torch.Tensor, w2_s: torch.Tensor, chunk: int = 64) -> dict:
    """This rank's expert banks in the checkpoint layout -> our 4 tensors (w13_blocks, w13_scales, w2_blocks, w2_scales), with
    the shapes the kernel's graph declares (test_layer_one_gpu.random_weights): blocks [E][N][K/2], scales [E][N][K/32] uint8."""
    E, n13, half_l = w13.shape
    L, ir = 2 * half_l, n13 // 2
    inv = w13_inverse_order(ir, w13.device)
    out = {k: torch.empty_like(v) for k, v in (("w13_blocks", w13), ("w13_scales", w13_s), ("w2_blocks", w2), ("w2_scales", w2_s))}
    for e0 in range(0, E, chunk):
        sl = slice(e0, min(E, e0 + chunk))
        ne = sl.stop - sl.start
        out["w13_blocks"][sl] = tiles_batched(unpack(w13[sl][:, inv])).reshape(ne, n13, half_l)
        out["w13_scales"][sl] = scale_tiles_batched(w13_s[sl][:, inv]).reshape(ne, n13, L // 32)
        out["w2_blocks"][sl] = tiles_batched(unpack(w2[sl])).reshape(ne, L, ir // 2)
        out["w2_scales"][sl] = scale_tiles_batched(w2_s[sl]).reshape(ne, L, ir // 32)
    return out


def self_test():
    sys.path.insert(0, os.path.join(ROOT, "amd", "bench"))
    sys.path.insert(0, os.path.join(ROOT, "amd", "tests"))
    from types import SimpleNamespace
    import sglang_moe_layer as sml
    E, L, IR, SHR = 6, 3584, 384, 768
    g = torch.Generator().manual_seed(3)
    w = {"w13_blocks": torch.randint(0, 256, (E, 2 * IR, L // 2), generator=g, dtype=torch.uint8),
         "w13_scales": torch.randint(110, 130, (E, 2 * IR, L // 32), generator=g, dtype=torch.uint8),
         "w2_blocks": torch.randint(0, 256, (E, L, IR // 2), generator=g, dtype=torch.uint8),
         "w2_scales": torch.randint(110, 130, (E, L, IR // 32), generator=g, dtype=torch.uint8)}
    sml.D = SimpleNamespace(**{**vars(sml.D), "E": E})
    ck = sml.checkpoint_banks(w, chunk=4)                      # ours -> the checkpoint layout (the bench's, checked vs SGLang)
    back = routed_experts(ck["w13"], ck["w13_s"], ck["w2"], ck["w2_s"], chunk=4)
    for k in w:
        assert torch.equal(back[k], w[k]), k
    sg = torch.randn(2 * SHR, 64, generator=g)
    rows = sml.shared_gate_up_rows(SHR, "cpu")
    assert torch.equal(sg[rows][shared_inverse_order(SHR, "cpu")], sg)
    print("SELF-TEST PASSED: checkpoint layout -> ours is the inverse of the bench's ours -> checkpoint (routed experts, "
          "shared gate_up rows)")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
