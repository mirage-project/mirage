#!/usr/bin/env python3
"""Probe P3 runner (v1-architecture.md §14) — attention smem instantiation sweep.

Compiles `p3_attn_smem.cu` once per MAX_TOKENS value against a given mirage tree
and records whether the SM100 attention kernel's own
`static_assert(S_TOTAL_OFFSET <= MAX_DYNAMIC_SHARED_MEMORY_SIZE)` fires.

Run it twice — against a PRE-cherry-pick tree and against the POST-cherry-pick
tree — so the artifact carries the counterfactual, not just the outcome we want.

  python3 run_p3.py --tree /path/to/mirage --label post_pick --out p3_post.json

Nothing here is GPU-bound: `nvcc -c -arch=sm_100a` is a host-side compile.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

MAX_TOKENS_SWEEP = [1, 2, 4, 8, 16]

# Mirrors python/mirage/mpk/persistent_kernel.py:200-241 (the flags the real
# megakernel build uses) restricted to what a bare TU needs.
BASE_FLAGS = [
    "-std=c++17",
    "-O3",
    "--expt-relaxed-constexpr",
    "-arch=sm_100a",
    "-DMPK_TARGET_CC=100",
    "-DMIRAGE_GRACE_BLACKWELL",
    "-DMIRAGE_BACKEND_USE_CUDA",
    "-DMODE_OFFLINE",
]


def include_flags(tree):
    return [
        f"-I{os.path.join(tree, 'include')}",
        f"-I{os.path.join(tree, 'include/mirage/persistent_kernel')}",
        f"-I{os.path.join(tree, 'include/mirage/persistent_kernel/tasks')}",
        f"-I{os.path.join(tree, 'deps/cutlass/include')}",
        f"-I{os.path.join(tree, 'deps/cutlass/tools/util/include')}",
    ]


def run_nvcc(nvcc, tree, src, extra, timeout=900):
    cmd = [nvcc] + BASE_FLAGS + include_flags(tree) + extra + ["-c", src, "-o", "/dev/null"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return {
        "cmd": " ".join(cmd),
        "returncode": p.returncode,
        "stderr": p.stderr,
        "seconds": round(time.time() - t0, 1),
    }


def classify(res):
    """COMPILES | STATIC_ASSERT | OTHER_ERROR — never guess from returncode alone."""
    if res["returncode"] == 0:
        return "COMPILES"
    err = res["stderr"]
    # nvcc surfaces the kernel's own budget assertion; require that we are
    # looking at THAT assert, not some unrelated compile break.
    if "static assertion failed" in err or "static_assert" in err:
        if "S_TOTAL_OFFSET" in err or "MAX_DYNAMIC_SHARED_MEMORY_SIZE" in err or "attention_sm100.cuh" in err:
            return "STATIC_ASSERT"
        return "STATIC_ASSERT_OTHER"
    return "OTHER_ERROR"


def emit_sizes(nvcc, tree, src, max_tokens):
    """Read the arena sizes out of the compiler's own diagnostic."""
    res = run_nvcc(nvcc, tree, src, [f"-DP3_MAX_TOKENS={max_tokens}", "-DP3_EMIT_SIZES"])
    found = {}
    for name, sym in [
        ("total_post_pick_model", "p3_emit_post_pick"),
        ("total_pre_pick_model", "p3_emit_pre_pick"),
        ("budget_bytes", "p3_emit_budget"),
        ("mma_iters_m", "p3_emit_mma_iters_m"),
    ]:
        # e.g.  incomplete type "P3_ARENA<174128UL>" ... p3_emit_post_pick
        m = re.findall(r"P3_ARENA<\s*(\d+)[uUlL]*\s*>[^\n]*" + sym, res["stderr"])
        if not m:
            # some nvcc versions put the symbol on the preceding line
            for line_i, line in enumerate(res["stderr"].splitlines()):
                if sym in line:
                    ctx = "\n".join(res["stderr"].splitlines()[max(0, line_i - 3):line_i + 4])
                    mm = re.findall(r"P3_ARENA<\s*(\d+)", ctx)
                    if mm:
                        m = mm
                        break
        found[name] = int(m[0]) if m else None
    return found, res["stderr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="mirage repo root to compile against")
    ap.add_argument("--label", required=True, help="pre_pick | post_pick")
    ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "p3_attn_smem.cu"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--nvcc", default=None)
    args = ap.parse_args()

    nvcc = args.nvcc or shutil.which("nvcc")
    if nvcc is None:
        sys.exit("nvcc not found on PATH (export PATH=/usr/local/cuda-12.8/bin:$PATH)")

    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip()
    head = subprocess.run(["git", "-C", args.tree, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", args.tree, "status", "--porcelain",
                            "include/mirage/persistent_kernel/tasks/blackwell/attention_sm100.cuh"],
                           capture_output=True, text=True).stdout.strip()
    kern = os.path.join(args.tree, "include/mirage/persistent_kernel/tasks/blackwell/attention_sm100.cuh")
    sha = subprocess.run(["sha256sum", kern], capture_output=True, text=True).stdout.split()[0]

    out = {
        "probe": "P3",
        "spec": "docs/qwen35/v1-architecture.md §14 P3 (validates §4.3 MAX_TOKENS choice)",
        "label": args.label,
        "tree": args.tree,
        "tree_head": head,
        "attention_sm100_cuh_sha256": sha,
        "attention_sm100_cuh_worktree_status": dirty or "clean",
        "nvcc_version": ver,
        "flags": BASE_FLAGS,
        "shape": {
            "num_q_heads": 16, "num_kv_heads": 2, "num_qo_per_kv": 8,
            "head_dim": 256, "kv_cache_stride": 512, "o_stride": 4096,
            "qkv_stride_qkvg": 9216, "page_size": 64, "max_seq_len": 2048,
        },
        "sweep": {},
    }

    for mt in MAX_TOKENS_SWEEP:
        res = run_nvcc(nvcc, args.tree, args.src, [f"-DP3_MAX_TOKENS={mt}"])
        verdict = classify(res)
        sizes, raw = emit_sizes(nvcc, args.tree, args.src, mt)
        out["sweep"][str(mt)] = {
            "verdict": verdict,
            "returncode": res["returncode"],
            "seconds": res["seconds"],
            "nvcc_cmd": res["cmd"],
            "stderr_head": res["stderr"][:1500],
            "arena_model": sizes,
        }
        print(f"MT={mt:<3} {verdict:<16} rc={res['returncode']} "
              f"post_pick_model={sizes.get('total_post_pick_model')} "
              f"pre_pick_model={sizes.get('total_pre_pick_model')} "
              f"budget={sizes.get('budget_bytes')} ({res['seconds']}s)", flush=True)

    # Falsifiable cross-check: does the mirrored arena model predict the SAME
    # boundary the real kernel's static_assert produced?
    budget = None
    agree = True
    mismatches = []
    key = "total_post_pick_model" if args.label == "post_pick" else "total_pre_pick_model"
    for mt, rec in out["sweep"].items():
        b = rec["arena_model"].get("budget_bytes")
        budget = b if b is not None else budget
        modeled = rec["arena_model"].get(key)
        if modeled is None or b is None:
            agree = False
            mismatches.append({"max_tokens": mt, "reason": "size extraction failed"})
            continue
        predicted = "COMPILES" if modeled <= b else "STATIC_ASSERT"
        if predicted != rec["verdict"]:
            agree = False
            mismatches.append({"max_tokens": mt, "predicted": predicted, "observed": rec["verdict"],
                               "modeled_bytes": modeled, "budget": b})
    out["model_vs_kernel_agreement"] = {"agree": agree, "budget_bytes": budget,
                                        "model_key": key, "mismatches": mismatches}

    admissible = [int(mt) for mt, r in out["sweep"].items() if r["verdict"] == "COMPILES"]
    out["max_admissible_max_tokens"] = max(admissible) if admissible else None

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nadmissible MAX_TOKENS: {sorted(admissible)}  -> max={out['max_admissible_max_tokens']}")
    print(f"model/kernel boundary agreement: {agree} (mismatches={mismatches})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
