"""MXFP4 expert weights -> the gfx950 layout moe_experts reads (amd/notes/moe_experts_gfx950.md).

Input, per expert, the usual MXFP4 storage: w [N][K / 2] uint8 (two e2m1 values per byte, the lower K in the low nibble) and
scales s [N][K / 32] uint8 (E8M0). Output:
  tiles   [N / 16][K / 128][64 lanes][16 B]: tile (rt, ks) = rows 16 rt .. +15, K 128 ks .. +127; lane l holds row 16 rt + l % 16,
          K 128 ks + 32 (l / 16) .. +31 (16 bytes), the A fragment of one v_mfma_scale_f32_16x16x128_f8f6f4 with FP4 A
          (ISA notes 2.1). A wave streams one row tile's K steps from consecutive 1 KB tiles.
  scales  K / 128 a multiple of 4 (W13): [N / 16][K / 512][64 lanes][4 B]: byte j of lane l = the scale of row
          16 rt + l % 16, K block 4 (4 kg + j) + l / 16, i.e. K step ks = 4 kg + j (one dword per lane carries 4 K steps).
          Otherwise (W2: 3 K steps): [N / 64][64 lanes][4 row tiles x K / 128] bytes: a lane's scales of 4 consecutive row
          tiles are contiguous (byte 3 (rt % 4) + ks with 3 K steps), so a 4-tile W2 item loads them as 3 dwords. Both have
          the input's byte count, so the graph's tensor shapes are the same as on B300.
  nibble  "lo" (the lower K in the low nibble, as the input) or "hi": amd/bench/mfma_layout_check.hip decides on day one.
W13 rows are reordered first (w13_item_order): item i (32 features 32 i .. +31) = gate tiles 2i, 2i + 1 then up tiles 2i, 2i + 1,
so one wave's 4 MFMA chains hold the gate and up values of the same features (SiTU in registers).

python amd/tools/shuffle_weights.py --self-test
"""
import argparse

import numpy as np

TILE_ROWS, TILE_K, LANES = 16, 128, 64


def unpack_fp4(w: np.ndarray) -> np.ndarray:
    """[N][K/2] bytes -> [N][K] nibbles (the lower K from the low nibble)."""
    lo, hi = w & 0xF, w >> 4
    out = np.empty((w.shape[0], w.shape[1] * 2), np.uint8)
    out[:, 0::2], out[:, 1::2] = lo, hi
    return out


def shuffle(w: np.ndarray, s: np.ndarray, nibble: str = "lo"):
    """(tiles, scales) of one weight matrix: w [N][K/2] uint8, s [N][K/32] uint8."""
    n, k = w.shape[0], w.shape[1] * 2
    assert n % TILE_ROWS == 0 and k % TILE_K == 0 and s.shape == (n, k // 32), (w.shape, s.shape)
    q = unpack_fp4(w)                                                  # [N][K]
    rt, ks = n // TILE_ROWS, k // TILE_K
    lane = np.arange(LANES)
    rows = (np.arange(rt)[:, None, None] * TILE_ROWS + (lane % 16)[None, None, :])               # [rt][1][64]
    kbase = (np.arange(ks)[None, :, None] * TILE_K + 32 * (lane // 16)[None, None, :])          # [1][ks][64]
    vals = q[rows[..., None], kbase[..., None] + np.arange(32)]                                  # [rt][ks][64][32]
    lo_k, hi_k = vals[..., 0::2], vals[..., 1::2]
    tiles = (lo_k | (hi_k << 4)) if nibble == "lo" else (hi_k | (lo_k << 4))
    if ks % 4 == 0:
        sc = np.empty((rt, ks // 4, LANES, 4), np.uint8)
        for j in range(4):
            for g in range(ks // 4):
                sc[:, g, :, j] = s[rows[:, 0, :], 4 * (4 * g + j) + lane // 16]
    else:
        assert rt % 4 == 0, "W2-style scales: whole groups of 4 row tiles"
        sc = np.empty((rt // 4, LANES, 4 * ks), np.uint8)
        for r in range(rt):
            for st in range(ks):
                sc[r // 4, :, ks * (r % 4) + st] = s[rows[r, 0, :], 4 * st + lane // 16]
    return tiles.astype(np.uint8), sc


def scale_of(sc, r, k, ks):
    """[64] scale bytes of row tile r, K step k (of ks), from either scale layout."""
    return sc[r, k // 4, :, k % 4] if ks % 4 == 0 else sc[r // 4, :, ks * (r % 4) + k]


def w13_item_order(ir: int) -> np.ndarray:
    """Row order of W13 [2 IR][K] (gate rows 0..IR-1, up rows IR..2IR-1) for items of 32 features: per item gate 32 rows, up 32."""
    order = []
    for i in range(ir // 32):
        order += list(range(32 * i, 32 * i + 32)) + list(range(ir + 32 * i, ir + 32 * i + 32))
    return np.array(order)


# ---- self-test: decode, emulate the MFMA from the shuffled layout, compare with a plain matmul ----
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def dequant(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    q = E2M1[unpack_fp4(w)]
    return q * np.repeat(np.exp2(s.astype(np.float64) - 127), 32, axis=1)


def mfma_emulate(tiles, sc, x, nibble="lo"):
    """y [N] x [T] from the shuffled layout, reading each lane's fragment the way the MFMA reads it (lane l: row l % 16,
    K 32 (l / 16) + 0..31 of the K step, scale byte (step % 4) of the lane's scale dword)."""
    rt, ks = tiles.shape[0], tiles.shape[1]
    y = np.zeros((rt * TILE_ROWS, x.shape[1]))
    for r in range(rt):
        for k in range(ks):
            frag = tiles[r, k]                                          # [64][16]
            lo, hi = (frag & 0xF, frag >> 4) if nibble == "lo" else (frag >> 4, frag & 0xF)
            vals = np.empty((LANES, 32), np.uint8)
            vals[:, 0::2], vals[:, 1::2] = lo, hi
            scale = np.exp2(scale_of(sc, r, k, ks).astype(np.float64) - 127)
            for l in range(LANES):
                kk = k * TILE_K + 32 * (l // 16) + np.arange(32)
                y[r * TILE_ROWS + l % 16] += (E2M1[vals[l]] * scale[l]) @ x[kk]
    return y


def self_test():
    rng = np.random.default_rng(1)
    for n, k in ((64, 384), (64, 512), (48, 3584)):
        w = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
        s = rng.integers(120, 134, (n, k // 32), dtype=np.uint8)
        x = rng.standard_normal((k, 8))
        ref = dequant(w, s) @ x
        for nibble in ("lo", "hi"):
            tiles, sc = shuffle(w, s, nibble)
            got = mfma_emulate(tiles, sc, x, nibble)
            err = np.abs(got - ref).max() / np.abs(ref).max()
            assert err < 1e-12, (n, k, nibble, err)
        print(f"N {n:3d} K {k:4d}: shuffled layout x tokens == dequantized matmul (both nibble orders)")
    order = w13_item_order(384)
    assert sorted(order) == list(range(768)) and list(order[:2]) == [0, 1] and order[32] == 384
    print("w13_item_order: a permutation; item 0 = gate 0..31, up 0..31")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    if p.parse_args().self_test:
        self_test()
