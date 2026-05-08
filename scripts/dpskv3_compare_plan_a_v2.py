"""Compare reference Plan A v2 sweep dir to MPK Plan A v2 sweep dir.

For each <tag>_mtp<N>/ subdir present in BOTH sides, read tokens.json
and report:
  - tokens_match (bool)
  - tokens_n_compared (int)
  - first_mismatch (idx or None)
  - ref_decoded_head, mpk_decoded_head (first 8 ids)
  - prefill_ms_ref, decode_tpot_ms_ref (latencies)
  - latency_ms_per_token_mpk

Also writes:
  - summary.json: structured machine-readable
  - summary.md: pretty markdown table for the user

Usage:
    python scripts/dpskv3_compare_plan_a_v2.py \
        --ref outputs/dpskv3_ref_plan_a_v2_<ts> \
        --mpk outputs/dpskv3_mpk_plan_a_v2_<ts> \
        --out outputs/plan_a_v2_compare_<ts>
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def _read_tokens_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def compare_one(ref_sub: Path, mpk_sub: Path) -> dict:
    ref = _read_tokens_json(ref_sub / "tokens.json")
    mpk = _read_tokens_json(mpk_sub / "tokens.json")
    out: dict = {
        "tag": ref_sub.name,
        "ref_present": ref is not None,
        "mpk_present": mpk is not None,
    }
    if ref is None or mpk is None:
        return out
    ref_ids = ref.get("decoded_suffix_ids") or ref.get("token_ids") or []
    mpk_ids = mpk.get("decoded_suffix_ids") or mpk.get("token_ids") or []
    n = min(len(ref_ids), len(mpk_ids))
    first_mm = next(
        (i for i in range(n) if ref_ids[i] != mpk_ids[i]), None
    )
    out.update({
        "tokens_match": ref_ids[:n] == mpk_ids[:n],
        "tokens_n_compared": n,
        "tokens_n_ref": len(ref_ids),
        "tokens_n_mpk": len(mpk_ids),
        "first_mismatch_idx": first_mm,
        "ref_head": ref_ids[:8],
        "mpk_head": mpk_ids[:8],
        "prefill_ms_ref": ref.get("prefill_ms"),
        "decode_tpot_ms_ref": ref.get("decode_tpot_ms"),
        "latency_ms_per_token_mpk": (
            mpk.get("latency_ms_per_token") if isinstance(mpk, dict) else None
        ),
    })
    return out


def collate(ref_root: Path, mpk_root: Path) -> list[dict]:
    rows: list[dict] = []
    # Find subdirs with tokens.json on either side.
    ref_subs = {p.name: p for p in ref_root.iterdir() if p.is_dir()}
    mpk_subs = {p.name: p for p in mpk_root.iterdir() if p.is_dir()}
    keys = sorted(set(ref_subs) | set(mpk_subs))
    for k in keys:
        ref_sub = ref_subs.get(k, ref_root / k)
        mpk_sub = mpk_subs.get(k, mpk_root / k)
        rows.append(compare_one(ref_sub, mpk_sub))
    return rows


def _is_mtp_workload(tag: str) -> bool:
    """Subdir naming convention: '<tag>_mtp<N>'."""
    return tag.endswith(("_mtp1", "_mtp2", "_mtp3"))


def to_markdown(rows: list[dict]) -> str:
    canonical_pass = sum(
        1 for r in rows
        if not _is_mtp_workload(r["tag"])
        and r.get("ref_present") and r.get("mpk_present")
        and r.get("tokens_match")
    )
    canonical_total = sum(
        1 for r in rows
        if not _is_mtp_workload(r["tag"])
        and r.get("ref_present") and r.get("mpk_present")
    )
    mtp_match = sum(
        1 for r in rows
        if _is_mtp_workload(r["tag"])
        and r.get("ref_present") and r.get("mpk_present")
        and r.get("tokens_match")
    )
    mtp_total = sum(
        1 for r in rows
        if _is_mtp_workload(r["tag"])
        and r.get("ref_present") and r.get("mpk_present")
    )

    lines = [
        "# Plan A v2 — Reference ↔ MPK comparison",
        "",
        f"**Canonical (mtp=0) tokens match**: {canonical_pass} / {canonical_total}",
        "",
        f"**MTP-on workload tokens match**: {mtp_match} / {mtp_total}  "
        "(expected to differ: ref always uses main argmax for next token, "
        "MPK uses MTP-draft + verify path so accepted drafts replace main).",
        "",
        "| Tag | Match | nRef/nMPK | First mismatch | Ref decode head | MPK decode head | prefill (ref) | TPOT (ref) | TPOT (mpk) |",
        "|-----|-------|-----------|----------------|-----------------|-----------------|---------------|------------|------------|",
    ]
    for r in rows:
        if not r.get("ref_present") or not r.get("mpk_present"):
            stat = "MISSING"
            extra = (
                f"ref={r['ref_present']}, mpk={r['mpk_present']}"
            )
            lines.append(
                f"| {r['tag']} | {stat} | — | — | {extra} | — | — | — | — |"
            )
            continue
        match = "PASS" if r["tokens_match"] else "FAIL"
        ratio = f"{r['tokens_n_ref']}/{r['tokens_n_mpk']}"
        first = r.get("first_mismatch_idx")
        first_s = "—" if first is None else str(first)
        ref_h = r["ref_head"]
        mpk_h = r["mpk_head"]
        prefill = r.get("prefill_ms_ref")
        prefill_s = "—" if prefill is None else f"{prefill:.1f}ms"
        tpot_r = r.get("decode_tpot_ms_ref")
        tpot_r_s = "—" if tpot_r is None else f"{tpot_r:.1f}ms"
        tpot_m = r.get("latency_ms_per_token_mpk")
        tpot_m_s = "—" if tpot_m is None else f"{tpot_m:.1f}ms"
        lines.append(
            f"| {r['tag']} | {match} | {ratio} | {first_s} | "
            f"{ref_h} | {mpk_h} | {prefill_s} | {tpot_r_s} | {tpot_m_s} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--mpk", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ref_root = Path(args.ref)
    mpk_root = Path(args.mpk)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = collate(ref_root, mpk_root)
    with open(out_root / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(out_root / "summary.md", "w") as f:
        f.write(to_markdown(rows))

    canonical_rows = [r for r in rows if not _is_mtp_workload(r["tag"])]
    canonical_present = [
        r for r in canonical_rows
        if r.get("ref_present") and r.get("mpk_present")
    ]
    canonical_pass = [r for r in canonical_present if r.get("tokens_match")]
    print(
        f"Canonical (mtp=0): {len(canonical_pass)}/{len(canonical_present)} match  "
        f"({len(canonical_rows)} total)"
    )
    print(f"summary: {out_root}/summary.md")
    # Exit non-zero if any canonical workload mismatched.
    return 0 if (
        canonical_present
        and len(canonical_pass) == len(canonical_present)
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
