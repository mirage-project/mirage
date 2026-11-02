"""One GPU's expert weights of a K3 MoE layer: the B300 kernel's files -> the gfx950 layout (amd/tools/shuffle_weights.py).

The B300 files (scripts/b300_session/prep_weights_hf.py, one directory per GPU): w13_blk [E][6][28][128][64] (W13 rows in blocks of
128 = 64 gate rows then the matching 64 up rows; 4-bit values, 64-byte pieces of 128 rows), w13_sf [E][168 x 512] (scales in the
kernel's chunk order), w2_blk [E][28][3][128][64], w2_sf [E][84 x 512]. Written next to them, the same byte counts:
  w13_gfx950  W13 tiles in item order (item i: gate rows 32i..32i+31, then up rows 32i..32i+31), w13_sf_gfx950 its scales
  w2_gfx950   W2 tiles, w2_sf_gfx950 its scales
--nibble lo|hi: the fp4 nibble order amd/bench/mfma_layout_check.hip found (lo: the lower K in the low nibble).

python amd/tools/prep_gfx950_from_b300.py DIR [--nibble lo] [--experts 896]
python amd/tools/prep_gfx950_from_b300.py --self-test
"""
import argparse
import os

import numpy as np

from shuffle_weights import dequant, mfma_emulate, shuffle

IR, LAT = 384, 3584


def unblk(p: np.ndarray, rows: int, kbytes: int) -> np.ndarray:
    """[rows/128][kbytes/64][128][64] -> [rows][kbytes] (prep_weights_hf.blk inverted)."""
    return p.reshape(rows // 128, kbytes // 64, 128, 64).transpose(0, 2, 1, 3).reshape(rows, kbytes)


def blk(a: np.ndarray, rows: int, kbytes: int) -> np.ndarray:
    return a.reshape(rows // 128, 128, kbytes // 64, 64).transpose(0, 2, 1, 3).copy()


def chunk_index(rows: int, kblocks: int) -> np.ndarray:
    """idx[row][kb] = the byte of scale (row, kb) in the chunk order (prep_weights_hf.canonical_chunks): chunk (cm, ck) of
    512 bytes, cm = row / 128, ck = kb / 4, byte (row % 32) 16 + ((row % 128) / 32) 4 + kb % 4."""
    r, k = np.meshgrid(np.arange(rows), np.arange(kblocks), indexing="ij")
    cm, ck, rr = r // 128, k // 4, r % 128
    return (cm * (kblocks // 4) + ck) * 512 + (rr % 32) * 16 + (rr // 32) * 4 + k % 4


def unchunk(flat: np.ndarray, rows: int, kblocks: int) -> np.ndarray:
    return flat[chunk_index(rows, kblocks)]


def chunk(s: np.ndarray) -> np.ndarray:
    rows, kblocks = s.shape
    out = np.zeros((rows // 128) * (kblocks // 4) * 512, np.uint8)
    out[chunk_index(rows, kblocks)] = s
    return out


def w13_b300_to_item_rows() -> np.ndarray:
    """order[j] = the B300 row of item-order row j."""
    gate = lambda f: (f // 64) * 128 + f % 64
    up = lambda f: (f // 64) * 128 + 64 + f % 64
    order = []
    for i in range(IR // 32):
        order += [gate(f) for f in range(32 * i, 32 * i + 32)] + [up(f) for f in range(32 * i, 32 * i + 32)]
    return np.array(order)


def convert_expert(w13p, w13s, w2p, w2s, nibble="lo"):
    """One expert's B300 arrays -> (w13 tiles, w13 scales, w2 tiles, w2 scales) in the gfx950 layout."""
    w13 = unblk(w13p, 2 * IR, LAT // 2)[w13_b300_to_item_rows()]
    s13 = unchunk(w13s, 2 * IR, LAT // 32)[w13_b300_to_item_rows()]
    w2 = unblk(w2p, LAT, IR // 2)
    s2 = unchunk(w2s, LAT, IR // 32)
    t13, c13 = shuffle(w13, s13, nibble)
    t2, c2 = shuffle(w2, s2, nibble)
    return t13, c13, t2, c2


def convert_dir(d: str, nibble: str, experts: int) -> None:
    rd = lambda n, shape: np.fromfile(os.path.join(d, n + ".bin"), np.uint8).reshape(shape)
    w13p, w13s = rd("w13_blk", (experts, 6, 28, 128, 64)), rd("w13_sf", (experts, 168 * 512))
    w2p, w2s = rd("w2_blk", (experts, 28, 3, 128, 64)), rd("w2_sf", (experts, 84 * 512))
    outs = {n: open(os.path.join(d, n + ".bin"), "wb") for n in ("w13_gfx950", "w13_sf_gfx950", "w2_gfx950", "w2_sf_gfx950")}
    for e in range(experts):
        for f, a in zip(outs.values(), convert_expert(w13p[e], w13s[e], w2p[e], w2s[e], nibble)):
            f.write(a.tobytes())
    for n, f in outs.items():
        f.close()
        src = {"w13_gfx950": "w13_blk", "w13_sf_gfx950": "w13_sf", "w2_gfx950": "w2_blk", "w2_sf_gfx950": "w2_sf"}[n]
        a, b = os.path.getsize(os.path.join(d, n + ".bin")), os.path.getsize(os.path.join(d, src + ".bin"))
        assert a == b, f"{n}: {a} bytes, {src} has {b}"
    print(f"{d}: {experts} experts -> w13_gfx950, w13_sf_gfx950, w2_gfx950, w2_sf_gfx950 (nibble {nibble}; byte counts as B300)")


def self_test():
    rng = np.random.default_rng(5)
    # one expert, logical: gate [IR][LAT/2], up [IR][LAT/2], W2 [LAT][IR/2] and scales
    g, u = rng.integers(0, 256, (IR, LAT // 2), np.uint8), rng.integers(0, 256, (IR, LAT // 2), np.uint8)
    gs, us = rng.integers(120, 134, (IR, LAT // 32), np.uint8), rng.integers(120, 134, (IR, LAT // 32), np.uint8)
    w2, s2 = rng.integers(0, 256, (LAT, IR // 2), np.uint8), rng.integers(120, 134, (LAT, IR // 32), np.uint8)
    # the B300 files, as prep_weights_hf.py makes them
    rows_p = np.concatenate(sum([[g[64 * m:64 * m + 64], u[64 * m:64 * m + 64]] for m in range(IR // 64)], []))
    rows_s = np.concatenate(sum([[gs[64 * m:64 * m + 64], us[64 * m:64 * m + 64]] for m in range(IR // 64)], []))
    w13p, w13s, w2p, w2s = blk(rows_p, 2 * IR, LAT // 2), chunk(rows_s), blk(w2, LAT, IR // 2), chunk(s2)
    assert (unblk(w13p, 2 * IR, LAT // 2) == rows_p).all() and (unchunk(w13s, 2 * IR, LAT // 32) == rows_s).all()
    x1, x2 = rng.standard_normal((LAT, 8)), rng.standard_normal((IR, 8))
    for nibble in ("lo", "hi"):
        t13, c13, t2, c2 = convert_expert(w13p, w13s, w2p, w2s, nibble)
        # item order: item i = gate 32i.., up 32i..
        ref13 = np.concatenate(sum([[dequant(g, gs)[32 * i:32 * i + 32] @ x1, dequant(u, us)[32 * i:32 * i + 32] @ x1]
                                    for i in range(IR // 32)], []))
        got13 = mfma_emulate(t13, c13, x1, nibble)
        assert np.abs(got13 - ref13).max() / np.abs(ref13).max() < 1e-12, nibble
        ref2 = dequant(w2, s2) @ x2
        got2 = mfma_emulate(t2, c2, x2, nibble)
        assert np.abs(got2 - ref2).max() / np.abs(ref2).max() < 1e-12, nibble
        assert t13.nbytes == w13p.nbytes and c13.nbytes == w13s.nbytes and t2.nbytes == w2p.nbytes and c2.nbytes == w2s.nbytes
    print("B300 files of one expert -> gfx950 layout: W13 (item order) and W2 x tokens == the logical weights (both nibble "
          "orders), byte counts unchanged")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", nargs="?")
    p.add_argument("--nibble", default="lo", choices=("lo", "hi"))
    p.add_argument("--experts", type=int, default=896)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
    else:
        convert_dir(a.dir, a.nibble, a.experts)
