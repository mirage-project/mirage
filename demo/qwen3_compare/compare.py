#!/usr/bin/env python3
"""Qwen3 hand-written against Qwen3 compiler-generated, side by side."""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO, "tests", "ci-tests", "run_batch_perf.py")

THROUGHPUT = re.compile(r"throughput:\s+([0-9.]+)\s+tokens/s")
LATENCY = re.compile(r"per-token latency:\s+([0-9.]+)\s+ms")
SAMPLE = re.compile(r"Sample output[^:]*:\s*(.*)")

VARIANTS = {
    "handwritten": {},
    "graph": {"MPK_MODEL_SOURCE": "mugraph"},
}

DEFAULT = ("handwritten", "graph")


def run_variant(name, args, tokens_path):
    """One build + decode, in its own process. A dict, or None if it failed."""
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "MPK_DUMP_TOKENS": tokens_path, **VARIANTS[name]}
    if args.gpu:
        env.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    cmd = [sys.executable, "-u", BENCH,
           "--model", args.model,
           "--max-num-batched-tokens", str(args.tokens),
           "--max-num-batched-requests", str(args.requests),
           "--max-seq-length", str(args.seq_len),
           "--ignore-eos"]
    if args.verbose:
        print(f"  $ {' '.join(cmd)}", flush=True)

    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True,
                              text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {args.timeout}s", flush=True)
        return None

    out = proc.stdout + proc.stderr
    throughput = THROUGHPUT.search(out)
    if not throughput:
        print(f"  FAILED (exit {proc.returncode})", flush=True)
        print("\n".join("    " + l for l in out.splitlines()[-12:]), flush=True)
        return None

    latency = LATENCY.search(out)
    sample = SAMPLE.search(out)
    tokens = None
    if os.path.exists(tokens_path):
        with open(tokens_path) as f:
            tokens = json.load(f).get("tokens")

    return {
        "variant": name,
        "env": VARIANTS[name],
        "throughput_tokens_per_s": float(throughput.group(1)),
        "latency_ms_per_token": float(latency.group(1)) if latency else None,
        "sample": sample.group(1).strip() if sample else "",
        "tokens": tokens,
    }


def tokens_column(result, base):
    if result is base:
        return "baseline"
    if result["tokens"] is None or base["tokens"] is None:
        return "?"
    return "same" if result["tokens"] == base["tokens"] else "DIFFER"


def report(results, baseline):
    """The table, everything measured against the `baseline` variant."""
    done = [r for r in results if r]
    if not done:
        print("\nno variant completed.")
        return

    base = next((r for r in done if r["variant"] == baseline), done[0])
    base_tps = base["throughput_tokens_per_s"]
    w = max(len(r["variant"]) for r in done)
    rule = "=" * (w + 46)

    print(f"\n{rule}")
    print(f"  Qwen3 build paths, side by side (baseline: {base['variant']})")
    print(rule)
    print(f"  {'variant':<{w}}  {'tok/s':>9}  {'ms/tok':>7}  {'vs base':>8}"
          f"  tokens")
    for r in done:
        tps = r["throughput_tokens_per_s"]
        lat = r["latency_ms_per_token"]
        print(f"  {r['variant']:<{w}}  {tps:>9.1f}"
              f"  {(f'{lat:.3f}' if lat else '--'):>7}"
              f"  {(f'{tps / base_tps:.3f}x' if base_tps else '--'):>8}"
              f"  {tokens_column(r, base)}")
    print(rule)

    for r in done:
        print(f"  {r['variant']:<{w}}  sample: {r['sample'][:60]}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", default=",".join(DEFAULT),
                    help=f"comma-separated; default {','.join(DEFAULT)}")
    ap.add_argument("--list", action="store_true",
                    help="print the available variants and exit")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokens", type=int, default=8,
                    help="max_num_batched_tokens")
    ap.add_argument("--requests", type=int, default=8,
                    help="max_num_batched_requests")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES for each child")
    ap.add_argument("--timeout", type=int, default=5400,
                    help="per-variant seconds; a build can take minutes")
    ap.add_argument("--baseline", help="default: the first variant run")
    ap.add_argument("--json", help="write the full results here")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.list:
        w = max(len(k) for k in VARIANTS)
        for name, env in VARIANTS.items():
            print(f"  {name:<{w}}  {env or '(no overrides)'}")
        return 0

    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        ap.error(f"unknown variant(s) {unknown}; --list shows them all")

    print(f"Qwen3 side-by-side: {', '.join(names)}")
    print(f"  model {args.model}, {args.requests} requests x {args.tokens} "
          f"tokens, seq {args.seq_len}")
    print("  one megakernel build per variant -- this takes minutes each\n")

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            print(f"[{name}] building and decoding...", flush=True)
            result = run_variant(name, args, os.path.join(tmp, f"{name}.json"))
            if result:
                print(f"  {result['throughput_tokens_per_s']:.1f} tokens/s",
                      flush=True)
            results.append(result)

    report(results, args.baseline or names[0])

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"config": vars(args),
                       "results": [r for r in results if r]}, f, indent=2)
        print(f"Wrote {args.json}")

    return 0 if any(results) else 1


if __name__ == "__main__":
    sys.exit(main())
