"""Compare a reference dump to an MPK dump.

Usage:
    python -m tests.dpskv3_reference.comparator \
        --reference outputs/dpskv3_reference_dump_<ts> \
        --mpk       outputs/<mpk_run>/<workload>_dump

Reports per-tensor cosine similarity + max-abs-diff for every tensor
that exists in both dumps. Tensors only in one side get flagged.

The MPK side currently does not produce a per-layer hidden state dump
— that's task #19 in the plan and lives in `python/mirage/mpk/...`
on the MPK side. Until that lands, this comparator is a no-op for
hidden states (it can still compare token IDs from `tokens.json`).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    if af.numel() == 0 or bf.numel() == 0:
        return float("nan")
    return F.cosine_similarity(af, bf, dim=0).item()


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def compare_dumps(
    reference_dir: Path,
    mpk_dir: Path,
    cosine_threshold: float = 0.999,
    abs_threshold: Optional[float] = None,
) -> dict:
    """Diff two dump directories, return a structured report."""
    report = {
        "reference_dir": str(reference_dir),
        "mpk_dir": str(mpk_dir),
        "iterations": [],
        "tokens_match": None,
        "errors": [],
    }
    # Compare token sequences if both exist.
    ref_tokens = reference_dir / "tokens.json"
    mpk_tokens = mpk_dir / "tokens.json"
    if ref_tokens.exists() and mpk_tokens.exists():
        with open(ref_tokens) as f:
            r = json.load(f)
        with open(mpk_tokens) as f:
            m = json.load(f)
        ref_ids = r.get("decoded_suffix_ids", r.get("token_ids", []))
        mpk_ids = m.get("decoded_suffix_ids", m.get("token_ids", []))
        n = min(len(ref_ids), len(mpk_ids))
        report["tokens_match"] = ref_ids[:n] == mpk_ids[:n]
        report["tokens_n_compared"] = n
        report["tokens_first_mismatch"] = next(
            (i for i in range(n) if ref_ids[i] != mpk_ids[i]), None
        )
        if n > 0:
            report["tokens_ref_head"] = ref_ids[:8]
            report["tokens_mpk_head"] = mpk_ids[:8]

    # Compare iter dumps tensor-by-tensor.
    ref_iter_dirs = sorted(reference_dir.glob("iter_*"))
    mpk_iter_dirs = sorted(mpk_dir.glob("iter_*"))
    for ref_d, mpk_d in zip(ref_iter_dirs, mpk_iter_dirs):
        per_iter = {"name": ref_d.name, "tensors": {}}
        ref_tensors = {p.stem: p for p in ref_d.glob("*.pt")}
        mpk_tensors = {p.stem: p for p in mpk_d.glob("*.pt")}
        common = sorted(set(ref_tensors) & set(mpk_tensors))
        only_ref = sorted(set(ref_tensors) - set(mpk_tensors))
        only_mpk = sorted(set(mpk_tensors) - set(ref_tensors))
        per_iter["only_in_reference"] = only_ref
        per_iter["only_in_mpk"] = only_mpk
        for name in common:
            try:
                a = torch.load(ref_tensors[name], map_location="cpu")
                b = torch.load(mpk_tensors[name], map_location="cpu")
            except Exception as e:
                per_iter["tensors"][name] = {"error": str(e)}
                continue
            if a.shape != b.shape:
                per_iter["tensors"][name] = {
                    "error": "shape mismatch",
                    "ref_shape": list(a.shape),
                    "mpk_shape": list(b.shape),
                }
                continue
            cos = _cosine(a, b)
            mad = _max_abs_diff(a, b)
            ok = cos >= cosine_threshold
            if abs_threshold is not None and mad > abs_threshold:
                ok = False
            per_iter["tensors"][name] = {
                "cosine": cos, "max_abs_diff": mad, "pass": ok,
            }
        report["iterations"].append(per_iter)
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--mpk", required=True)
    p.add_argument("--cosine-threshold", type=float, default=0.999)
    p.add_argument("--abs-threshold", type=float, default=None)
    p.add_argument("--out", default=None,
                   help="Write report JSON to this path (else print).")
    args = p.parse_args()
    rep = compare_dumps(
        Path(args.reference), Path(args.mpk),
        cosine_threshold=args.cosine_threshold,
        abs_threshold=args.abs_threshold,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
