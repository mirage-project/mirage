"""RMSNorm on a hidden size that is not a whole number of thread-tiles.

HIDDEN_DIM used to have to divide both NUM_THREADS and the copy-async tile;
the tile loop now covers a short last tile.

Both widths run in one task graph: 4096 is the even-split no-regression
control, 2880 (2880 % 256 = 64) exercises the short tile.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

BATCH = 8
# (name, hidden, exercises the short-tile path)
CONFIGS = [("even_4096", 4096, False), ("uneven_2880", 2880, True)]
EPS = 1e-6  # the registration hardcodes this


def reference(x, w):
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + EPS) * w.float()).to(x.dtype)


def main():
    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=num_workers,
                  num_local_schedulers=num_schedulers,
                  max_num_batched_tokens=BATCH, max_num_batched_requests=1)
    pk = PersistentKernel(**params)

    cases = []
    for name, hidden, short_tile in CONFIGS:
        x = torch.randn(BATCH, hidden, dtype=dtype, device=device)
        w = torch.randn(hidden, dtype=dtype, device=device)
        out = torch.zeros(BATCH, hidden, dtype=dtype, device=device)
        pk.rmsnorm_layer(
            input=pk.attach_input(x, name=f"{name}_x"),
            weight=pk.attach_input(w, name=f"{name}_w"),
            output=pk.attach_input(out, name=f"{name}_out"),
            grid_dim=(BATCH, 1, 1), block_dim=(256, 1, 1))
        cases.append((name, hidden, short_tile, x, w, out))

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ok = True
    for name, hidden, short_tile, x, w, out in cases:
        ref = reference(x, w)
        diff = (out.float() - ref.float()).abs().max().item()
        tol = max(0.02, 0.01 * ref.abs().max().item())
        # 256 threads x 8 elements per copy = a 2048-element tile
        tiles = (hidden + 2047) // 2048
        print(f"[{name}] hidden {hidden} over {tiles} tile(s), last tile "
              f"{hidden - (tiles - 1) * 2048} wide: max diff {diff:.4f} "
              f"(tol {tol:.4f})")
        if (hidden % 2048 == 0) == short_tile:
            print(f"[{name}] FAILED: this case does not test what it claims")
            ok = False
        if diff >= tol:
            print(f"[{name}] FAILED: disagrees with the reference")
            ok = False
        # Check the scale directly: normalising over the wrong length still
        # looks smooth.
        got_rms = out.float().pow(2).mean(dim=-1).sqrt()
        ref_rms = ref.float().pow(2).mean(dim=-1).sqrt()
        if (got_rms - ref_rms).abs().max().item() >= 0.05:
            print(f"[{name}] FAILED: normalised over the wrong element count")
            ok = False

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("\nPASSED: RMSNorm matches the reference for both an even and a "
          "short last tile")


if __name__ == "__main__":
    main()
