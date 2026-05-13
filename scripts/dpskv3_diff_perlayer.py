#!/usr/bin/env python3
"""Diff MPK's per-layer hidden states against the reference's, layer by layer.

Inputs:
  --mpk    path to MPK dump dir (`--dump-hidden-dir <dir>` from demo.py).
           Contains `embed.pt` + `layer_<NN>_residual.pt`.
  --ref    path to reference dump dir (record_hidden=True; the runner
           writes `iter_0000/layer_<L>_residual.pt` for the prefill iter).
  --layers comma list or "a-b" of layer indices to compare. Default:
           all that exist on both sides.

Output: one line per layer with cosine + max-abs-diff between MPK's
residual and the reference's residual at that layer's output (= the
sum of attn_out and mlp_out for that block). When this prints
`cos >= 0.999` for every layer, the per-token argmax drift is purely
from FP8 quantization noise compounding — no kernel bug. A sharp
drop at layer K isolates the bug to that layer.

Example:

    python scripts/dpskv3_diff_perlayer.py \\
        --mpk outputs/<mpk_run>/ \\
        --ref outputs/<ref_run>/iter_0000

(Both `--mpk` and `--ref` only need to contain `layer_NN_residual.pt`
files at the top level for this script to work. Reference iter dirs
already match that pattern.)
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import torch
import torch.nn.functional as F


_LAYER_RE = re.compile(r"layer_0*(\d+)_residual\.pt$")


def _list_layers(d: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in d.glob("layer_*_residual.pt"):
        m = _LAYER_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def _parse_layer_filter(s: str | None) -> set[int] | None:
    if not s:
        return None
    if "-" in s:
        a, b = s.split("-", 1)
        return set(range(int(a), int(b) + 1))
    return {int(x) for x in s.split(",")}


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    if af.numel() == 0 or bf.numel() == 0:
        return float("nan")
    return F.cosine_similarity(af, bf, dim=0).item()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mpk", required=True, type=Path)
    p.add_argument("--ref", required=True, type=Path)
    p.add_argument("--layers", default=None)
    p.add_argument("--max-rows", type=int, default=None,
                   help="Only compare the first N rows of each tensor. "
                        "Useful when MPK pads rows past the real prompt.")
    args = p.parse_args()

    mpk_layers = _list_layers(args.mpk)
    ref_layers = _list_layers(args.ref)
    flt = _parse_layer_filter(args.layers)

    common = sorted(set(mpk_layers) & set(ref_layers))
    if flt is not None:
        common = [i for i in common if i in flt]
    if not common:
        print(f"No layer dumps in common.")
        print(f"  MPK layers: {sorted(mpk_layers)}")
        print(f"  REF layers: {sorted(ref_layers)}")
        return 1

    print(f"{'Layer':>5}  {'MPK shape':>16}  {'REF shape':>16}  {'cos':>8}  "
          f"{'mad':>10}  {'mean|ref|':>10}")
    print("-" * 80)
    for i in common:
        mpk = torch.load(mpk_layers[i], map_location="cpu", weights_only=True)
        ref = torch.load(ref_layers[i], map_location="cpu", weights_only=True)
        # Reference dumps [T, H]; MPK dumps [mbt, H] where mbt may pad past
        # the real prompt length. Truncate to the smaller of the two row
        # counts so the cosine isn't dominated by zero-padded rows.
        T = min(mpk.shape[0], ref.shape[0])
        if args.max_rows:
            T = min(T, args.max_rows)
        mpk_t = mpk[:T]
        ref_t = ref[:T]
        cos = _cosine(mpk_t, ref_t)
        mad = (mpk_t.float() - ref_t.float()).abs().max().item()
        mean_ref = ref_t.float().abs().mean().item()
        print(f"{i:>5d}  {str(tuple(mpk.shape)):>16}  {str(tuple(ref.shape)):>16}  "
              f"{cos:>8.4f}  {mad:>10.4f}  {mean_ref:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
