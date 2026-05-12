"""Compare a reference dump against an MPK dump.

Reference dumps (from `runner.run_reference`) live under
    <ref_dump_dir>/
        config.json
        tokens.json
        iter_<N>/
            input_ids.pt, positions.pt, embed.pt,
            layer_<L>_residual.pt for each L in `layer_indices`,
            final_norm.pt, logits.pt, argmax.pt

MPK dumps (from `demo/deepseek_v3/demo.py --dump-hidden-dir <out>`)
live under
    <mpk_dump_dir>/
        embed.pt
        layer_<L>_residual.pt for each L in `--layers`

The shape conventions differ slightly:
  - reference: [T, H]  where T = chunk len for iter
  - MPK: [mbt, H] where mbt = max_num_batched_tokens (rows beyond
    the real prompt are zero-padded; row 0 is overwritten by the
    decode iteration — see `feedback_row0_dump_artifact` memory).

So we compare rows [1 : prompt_len) by default, skipping row 0
(decode overwrite) and zero-padded rows beyond prompt_len.
"""

from __future__ import annotations
import argparse
import json
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


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    if af.numel() == 0 or bf.numel() == 0:
        return float("nan")
    return F.cosine_similarity(af, bf, dim=0).item()


def compare_dumps(
    ref_iter_dir: Path | str,
    mpk_dump_dir: Path | str,
    layer_indices: list[int] | None = None,
    skip_row_0: bool = True,
    max_rows: int | None = None,
) -> list[dict]:
    """Compare per-layer residual dumps.

    Args:
        ref_iter_dir: Path to a `iter_<N>` subdir of a reference dump
            (where layer_<L>_residual.pt files live).
        mpk_dump_dir: MPK's `--dump-hidden-dir` output.
        layer_indices: Layers to compare. None = intersection of dumps.
        skip_row_0: Skip row 0 (MPK's decode-overwrite artifact).
        max_rows: Truncate to first N rows after the optional row-0 skip.

    Returns:
        list of {layer, cos_full, mad, n_rows_compared, per_row_min_cos}.
    """
    ref_iter_dir = Path(ref_iter_dir)
    mpk_dump_dir = Path(mpk_dump_dir)
    ref_layers = _list_layers(ref_iter_dir)
    mpk_layers = _list_layers(mpk_dump_dir)
    common = sorted(set(ref_layers) & set(mpk_layers))
    if layer_indices is not None:
        common = [i for i in common if i in layer_indices]

    results = []
    for li in common:
        ref = torch.load(ref_layers[li], map_location="cpu", weights_only=True).float()
        mpk = torch.load(mpk_layers[li], map_location="cpu", weights_only=True).float()
        # Reference is [T, H], MPK is [mbt, H]. Use first min(T_ref, T_mpk) rows.
        T = min(ref.shape[0], mpk.shape[0])
        ref_t = ref[:T]
        mpk_t = mpk[:T]
        if skip_row_0 and T > 1:
            ref_t = ref_t[1:]
            mpk_t = mpk_t[1:]
        if max_rows:
            ref_t = ref_t[:max_rows]
            mpk_t = mpk_t[:max_rows]
        full_cos = _cosine(ref_t, mpk_t)
        per_row_cos = F.cosine_similarity(ref_t, mpk_t, dim=1)
        n_bad = int((per_row_cos < 0.99).sum().item())
        mad = float((ref_t - mpk_t).abs().max().item())
        results.append({
            "layer": li,
            "n_rows": ref_t.shape[0],
            "cos_full": full_cos,
            "per_row_min_cos": float(per_row_cos.min().item()),
            "n_rows_below_0.99": n_bad,
            "max_abs_diff": mad,
        })
    return results


def _parse_layer_arg(s: str | None) -> list[int] | None:
    if not s:
        return None
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="reference iter_<N> dir")
    p.add_argument("--mpk", required=True, help="MPK --dump-hidden-dir dir")
    p.add_argument("--layers", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--keep-row-0", action="store_true",
                   help="include row 0 in the comparison (default skip)")
    args = p.parse_args()
    rows = compare_dumps(
        ref_iter_dir=args.ref,
        mpk_dump_dir=args.mpk,
        layer_indices=_parse_layer_arg(args.layers),
        skip_row_0=not args.keep_row_0,
        max_rows=args.max_rows,
    )
    if not rows:
        print("no layers in common.")
        return 1
    print(f"{'layer':>5} {'n_rows':>7} {'cos_full':>10} {'min_row_cos':>12} "
          f"{'bad<0.99':>9} {'max_abs_diff':>12}")
    for r in rows:
        print(f"{r['layer']:>5} {r['n_rows']:>7} "
              f"{r['cos_full']:>10.4f} {r['per_row_min_cos']:>12.4f} "
              f"{r['n_rows_below_0.99']:>9} {r['max_abs_diff']:>12.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
