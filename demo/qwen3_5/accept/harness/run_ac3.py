#!/usr/bin/env python3
"""AC-3 correctness harness — CLI entry point.

Compares an engine's greedy-decode token ids against the committed HF `transformers`
reference (`accept/reference/reference_outputs.json`) for the pinned prompt set, at every
batch size in `.pm/goal.md`'s AC-3 protocol. Gate: all token ids equal, no tolerance. Emits
per-position margin instrumentation (ref top-2 ids + margin, engine argmax, first divergence
per prompt) on every run, and a waiver-request evidence record (never an auto-waiver) for any
first-divergent position — see `ac3_types.WaiverRequest` and `tie_classifier`.

Usage (real gate, once an engine can dump token ids per batch size):
    python run_ac3.py --engine-dump-dir /path/to/dumps --output-json run_report.json
    # expects /path/to/dumps/bs1.json, bs2.json, bs4.json, bs8.json, bs16.json — see
    # engine_adapter.py's module docstring for the exact per-file JSON shape.

Usage (partial smoke — NOT an AC-3 verdict, e.g. against the committed vLLM smoke artifact):
    python run_ac3.py --vllm-smoke ../reference/vllm_smoke/vllm_smoke_result.json

Exit codes: 0 = AC-3 PASS. 1 = AC-3 FAIL (real mismatch — see the waiver-request records
before assuming it's a bug). 2 = usage/integrity error (bad args, malformed reference, wrong
prompt count). 3 = not applicable as the real gate (no engine data at all, or an intentional
--allow-partial / --vllm-smoke smoke run — see the printed status either way).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

# Flat sibling-module imports (matches this directory's other scripts, e.g. bench_vllm.py) —
# make this directory importable regardless of the caller's cwd before importing siblings.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ac3_types import GateReport  # noqa: E402
from ac3_runner import run_ac3  # noqa: E402
from engine_adapter import JSONDumpAdapter, StaticMappingAdapter, load_vllm_smoke  # noqa: E402
from reference_loader import load_reference  # noqa: E402
from tie_classifier import DEFAULT_TIE_MARGIN_THRESHOLD  # noqa: E402


def _default_reference_path() -> str:
    # .../accept/harness/run_ac3.py -> .../accept/reference/reference_outputs.json
    return str(Path(__file__).resolve().parent.parent / "reference" / "reference_outputs.json")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default=_default_reference_path(),
                     help="Path to the committed reference_outputs.json.")
    ap.add_argument("--engine-dump-dir", default=None,
                     help="Directory with bs<N>.json per requested batch size (see engine_adapter.py).")
    ap.add_argument("--vllm-smoke", default=None,
                     help="Path to vllm_smoke_result.json - convenience partial-smoke mode (implies "
                          "--allow-partial, restricts the sweep to bs=1 / that one prompt).")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16",
                     help="Comma-separated batch sizes (default: the AC-3/AC-4 pinned set).")
    ap.add_argument("--allow-partial", action="store_true",
                     help="Score only the (prompt, batch_size) pairs the engine actually returned, "
                          "instead of hard-failing on anything missing. NOT an AC-3 verdict.")
    ap.add_argument("--tie-margin-threshold", type=float, default=DEFAULT_TIE_MARGIN_THRESHOLD,
                     help="Provisional noise-floor cutoff for the tie-flip classifier (see tie_classifier.py).")
    ap.add_argument("--expect-num-prompts", type=int, default=10,
                     help="Integrity check: the reference must contain exactly this many prompts.")
    ap.add_argument("--output-json", default=None,
                     help="If given, write the full machine-readable run report here.")
    return ap


def _print_summary(report: GateReport) -> None:
    print(f"AC-3 HARNESS RUN — status={report.status} overall_pass={report.overall_pass}")
    print(f"  batch_sizes={report.batch_sizes}")
    print(f"  prompts={report.prompt_ids}")
    print(f"  tie_margin_threshold={report.tie_margin_threshold}")

    me = report.margin_evidence
    if me.get("available"):
        print(
            f"  margin evidence: {me['margin_data_available_positions']}/{me['total_positions']} "
            f"positions (min={me['min']:.4f} max={me['max']:.4f} mean={me['mean']:.4f})"
        )
    else:
        print(f"  margin evidence: UNAVAILABLE — {me.get('reason')}")

    for r in report.prompt_results:
        status_str = "PASS" if r.passed else f"FAIL (first divergence @ position {r.first_divergent_position})"
        print(f"  [{r.prompt_id} bs={r.batch_size}] {status_str} — {len(r.positions)} position(s) scored")

    if report.waiver_requests:
        print(f"  {len(report.waiver_requests)} waiver-request record(s) — evidence only, never auto-waived:")
        for w in report.waiver_requests:
            ev = w.evidence
            print(
                f"    - {w.prompt_id} bs={w.batch_size} pos={w.first_divergent_position} "
                f"verdict={w.classifier_verdict} ref_top1={ev.ref_top1_id} "
                f"engine_argmax={ev.engine_argmax_id} margin={ev.margin}"
            )

    for n in report.notes:
        print(f"  NOTE: {n}")


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"ACCEPT FAIL (integrity, exit 2): reference file not found: {ref_path}")
        return 2
    try:
        references = load_reference(ref_path)
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"ACCEPT FAIL (integrity, exit 2): malformed reference {ref_path}: {e}")
        return 2

    if len(references) != args.expect_num_prompts:
        print(
            f"ACCEPT FAIL (integrity, exit 2): expected {args.expect_num_prompts} prompts, "
            f"found {len(references)}: {sorted(references)}"
        )
        return 2

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    allow_partial = args.allow_partial or bool(args.vllm_smoke)
    prompt_ids = None

    if args.vllm_smoke:
        smoke_path = Path(args.vllm_smoke)
        if not smoke_path.exists():
            print(f"ACCEPT NOT-APPLICABLE (exit 3): vLLM smoke artifact not found: {smoke_path}")
            return 3
        engine_map = load_vllm_smoke(smoke_path)
        adapter = StaticMappingAdapter({1: engine_map})
        if batch_sizes != [1]:
            print("NOTE: --vllm-smoke only carries bs=1 data; restricting this run to bs=1.")
        batch_sizes = [1]
        prompt_ids = list(engine_map.keys())
    elif args.engine_dump_dir:
        dump_dir = Path(args.engine_dump_dir)
        candidate_paths = {bs: dump_dir / f"bs{bs}.json" for bs in batch_sizes}
        present = {bs: p for bs, p in candidate_paths.items() if p.exists()}
        missing = [str(p) for bs, p in candidate_paths.items() if bs not in present]
        if missing and not allow_partial:
            print(f"ACCEPT NOT-APPLICABLE (exit 3): engine dump file(s) missing: {missing}")
            return 3
        if not present:
            print(f"ACCEPT NOT-APPLICABLE (exit 3): no engine dump files found under {dump_dir}")
            return 3
        adapter = JSONDumpAdapter(present)
        batch_sizes = list(present.keys())
    else:
        print("ACCEPT NOT-APPLICABLE (exit 3): no --engine-dump-dir or --vllm-smoke given - nothing to test yet.")
        return 3

    report = run_ac3(
        adapter=adapter,
        references=references,
        batch_sizes=batch_sizes,
        prompt_ids=prompt_ids,
        tie_margin_threshold=args.tie_margin_threshold,
        allow_partial=allow_partial,
    )

    _print_summary(report)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(dataclasses.asdict(report), f, indent=2)
        print(f"Full run report written to {out_path}")

    if report.status == "partial_smoke_only":
        return 3
    return 0 if report.overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
