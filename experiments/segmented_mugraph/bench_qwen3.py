"""Stage 2: three-mode Qwen3-0.6B end-to-end experiment.

Modes
-----
``torch``            unmodified Hugging Face model execution.
``hybrid-mugraph``   the same execution with each layer's MLP replaced by
                     :class:`HybridQwen3MLP` (segmented muGraph regions).
``mpk``              the existing MPK task-graph / custom-kernel path, driven by
                     ``demo/qwen3/demo.py --use-mirage`` completely unmodified.

Fair-comparison caveat
----------------------
The Stage-1 microbenchmark is an apples-to-apples *kernel* comparison.  This
end-to-end comparison is **not**: ``torch`` and ``hybrid-mugraph`` are driven by
PyTorch/HF Python orchestration (per-op launches, HF KV cache, Python sampling
loop), whereas ``mpk`` runs the whole decode inside one persistent megakernel
with its own scheduler and its own attention/sampling kernels.  Differences
therefore reflect two different runtimes, not only the MLP.

Example
-------
    PYTHONPATH=. python -m experiments.segmented_mugraph.bench_qwen3 \
        --model Qwen/Qwen3-0.6B --gen-tokens 32 \
        --out experiments/outputs/stage2_qwen3.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import torch

from . import common
from .common import env_info, fmt_table, num, peak_memory, write_json

MODES = ("torch", "hybrid-mugraph", "mpk")
DEFAULT_PROMPT = "Explain what a persistent GPU megakernel is, in two sentences."
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _percentile(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = q * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


# ==========================================================================
# PyTorch / hybrid worker
# ==========================================================================


def _load_model(args):
    # A stale HF_TOKEN in the environment makes the hub reject anonymous reads.
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    from transformers import AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    src = args.model_path or args.model
    tokenizer = AutoTokenizer.from_pretrained(src, token=False)
    model = Qwen3ForCausalLM.from_pretrained(
        src, dtype=torch.bfloat16, token=False
    ).to("cuda").eval()
    return model, tokenizer


@torch.inference_mode()
def _generate(model, tokenizer, args, stats=None) -> Dict[str, Any]:
    """Greedy decode with the HF KV cache, timing prefill and each decode step."""
    from transformers import DynamicCache

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer([text], return_tensors="pt").to(model.device)
    input_ids = enc.input_ids
    prompt_len = input_ids.shape[-1]

    cache = DynamicCache()
    ev = lambda: torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    p0, p1 = ev(), ev()
    p0.record()
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    p1.record()
    torch.cuda.synchronize()
    prefill_ms = p0.elapsed_time(p1)

    first_logits = out.logits[:, -1, :].float().clone()
    next_tok = first_logits.argmax(dim=-1)

    generated = [int(next_tok.item())]
    step_ms: List[float] = []
    cur = next_tok.view(1, 1)
    pos = prompt_len

    for _ in range(args.gen_tokens - 1):
        s0, s1 = ev(), ev()
        torch.cuda.synchronize()
        s0.record()
        out = model(
            input_ids=cur,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([pos], device=model.device),
        )
        nxt = out.logits[:, -1, :].argmax(dim=-1)
        s1.record()
        torch.cuda.synchronize()
        step_ms.append(s0.elapsed_time(s1))
        generated.append(int(nxt.item()))
        cur = nxt.view(1, 1)
        pos += 1
        if tokenizer.eos_token_id is not None and int(nxt.item()) == tokenizer.eos_token_id:
            break

    ttft_ms = prefill_ms + (step_ms[0] if step_ms else 0.0)
    mean_itl = sum(step_ms) / len(step_ms) if step_ms else float("nan")
    return {
        "prompt_length": prompt_len,
        "token_ids": generated,
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "first_decode_logits": first_logits,
        "step_ms": step_ms,
        "prefill_ms": prefill_ms,
        "time_to_first_token_ms": ttft_ms,
        "mean_inter_token_ms": mean_itl,
        "p50_inter_token_ms": _percentile(step_ms, 0.50),
        "p95_inter_token_ms": _percentile(step_ms, 0.95),
        "decode_tokens_per_s": (1e3 / mean_itl) if mean_itl and mean_itl == mean_itl else float("nan"),
        "num_decode_steps": len(step_ms),
    }


def _aggregate(reps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool decode-step samples across repetitions into one set of statistics.

    Percentiles are computed over every measured step from every repetition
    rather than over per-repetition means, so tail latency stays visible.
    """
    first = reps[0]
    steps = [ms for r in reps for ms in r["step_ms"]]
    prefills = [r["prefill_ms"] for r in reps]
    ttfts = [r["time_to_first_token_ms"] for r in reps]
    mean_itl = (sum(steps) / len(steps)) if steps else float("nan")
    return {
        "prompt_length": first["prompt_length"],
        # token ids/text come from the first repetition; greedy decoding is
        # deterministic, so every repetition produces the same sequence.
        "token_ids": first["token_ids"],
        "text": first["text"],
        "first_decode_logits": first["first_decode_logits"],
        "reps": len(reps),
        "prefill_ms": sum(prefills) / len(prefills),
        "time_to_first_token_ms": sum(ttfts) / len(ttfts),
        "mean_inter_token_ms": mean_itl,
        "p50_inter_token_ms": _percentile(steps, 0.50),
        "p95_inter_token_ms": _percentile(steps, 0.95),
        "decode_tokens_per_s": (1e3 / mean_itl) if mean_itl == mean_itl else float("nan"),
        "num_decode_steps": len(steps),
    }


def run_torchlike_worker(args) -> Dict[str, Any]:
    from .runner import SegmentedMuGraphRunner, no_task_graph_guard
    from .hybrid_mlp import patch_qwen3_mlps, precompile_buckets

    torch.cuda.reset_peak_memory_stats()
    cold0 = time.perf_counter()
    model, tokenizer = _load_model(args)
    load_s = time.perf_counter() - cold0

    handle: Dict[str, Any] = {}
    runner = None
    compile_s = 0.0
    guard = None

    if args.mode == "hybrid-mugraph":
        buckets = sorted({1} | {int(b) for b in args.extra_buckets.split(",") if b.strip()})
        guard = no_task_graph_guard(REPO_ROOT)
        guard.__enter__()
        runner = SegmentedMuGraphRunner(
            device="cuda",
            try_superoptimize=not args.no_superoptimize,
            verbose=True,
        )
        handle = patch_qwen3_mlps(model, runner, allowed_tokens=buckets)
        c0 = time.perf_counter()
        precompile_buckets(model, runner, buckets)
        compile_s = time.perf_counter() - c0

    # warmups, then `--reps` measured generations aggregated over all steps
    for _ in range(args.warmups):
        _generate(model, tokenizer, args)
    reps = [_generate(model, tokenizer, args) for _ in range(max(1, args.reps))]
    res = _aggregate(reps)
    cold_total = time.perf_counter() - cold0

    logits = res.pop("first_decode_logits")
    payload = {
        "mode": args.mode,
        "model": args.model_path or args.model,
        "cold": {
            "model_load_s": load_s,
            "region_compile_s": compile_s,
            "total_s": cold_total,
        },
        "memory": peak_memory(),
        **res,
    }
    if runner is not None:
        report = runner.report()
        payload["mugraph"] = {
            "num_region_variants": runner.num_variants,
            "cache_hits": runner.cache_hits,
            "cache_misses": runner.cache_misses,
            "mugraph_calls": handle["stats"]["mugraph_calls"],
            "fallback_calls": handle["stats"]["fallback_calls"],
            "allowed_token_buckets": handle["allowed_tokens"],
            "patched_layers": len(handle["patched_layers"]),
            "search_time_s": sum(r["search_time_s"] for r in report),
            "cuda_compile_time_s": sum(r["compile_time_s"] for r in report),
            "regions": report,
        }
    if guard is not None:
        guard.__exit__(None, None, None)

    if args.logits_out:
        torch.save(logits.cpu(), args.logits_out)
    return payload


# ==========================================================================
# MPK worker -- shells out to the unmodified demo
# ==========================================================================


def run_mpk(args) -> Dict[str, Any]:
    demo = os.path.join(REPO_ROOT, "demo", "qwen3", "demo.py")
    if not os.path.exists(demo):
        return {"mode": "mpk", "status": "skipped", "reason": f"missing {demo}"}
    with tempfile.TemporaryDirectory() as td:
        tokens_path = os.path.join(td, "mpk_output.json")
        cmd = [
            sys.executable, demo, "--use-mirage",
            "--model", args.model,
            "--prompt", args.prompt,
            "--max-new-tokens", str(args.gen_tokens),
            "--save-tokens", tokens_path,
        ]
        if args.model_path:
            cmd += ["--model-path", args.model_path]
        base_env = dict(os.environ)
        base_env["PYTHONPATH"] = REPO_ROOT + os.pathsep + base_env.get("PYTHONPATH", "")
        base_env.pop("HF_TOKEN", None)
        base_env.pop("HUGGING_FACE_HUB_TOKEN", None)

        # demo.py loads the tokenizer without ``token=False``; a stale on-disk HF
        # credential then makes even a fully cached repo 401.  Retry offline.
        attempts = [base_env, {**base_env, "HF_HUB_OFFLINE": "1"}]
        proc = None
        wall = 0.0
        for env in attempts:
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                                  capture_output=True, text=True, timeout=args.mpk_timeout)
            wall = time.perf_counter() - t0
            if proc.returncode == 0 and os.path.exists(tokens_path):
                break
        if proc is None or proc.returncode != 0 or not os.path.exists(tokens_path):
            tail = (proc.stderr or proc.stdout or "")[-1200:] if proc else ""
            return {
                "mode": "mpk", "status": "skipped",
                "reason": f"demo/qwen3/demo.py --use-mirage failed (rc={proc.returncode if proc else '?'})",
                "stderr_tail": tail,
            }
        with open(tokens_path) as f:
            got = json.load(f)
    per_tok = got.get("latency_ms_per_token", float("nan"))
    return {
        "mode": "mpk",
        "status": "ok",
        "model": args.model_path or args.model,
        "cold": {"model_load_s": None, "region_compile_s": None, "total_s": wall},
        "prompt_length": got.get("prompt_length"),
        "token_ids": got.get("token_ids", []),
        "text": got.get("text", ""),
        "prefill_ms": None,
        "time_to_first_token_ms": None,
        "mean_inter_token_ms": per_tok,
        "p50_inter_token_ms": None,
        "p95_inter_token_ms": None,
        "decode_tokens_per_s": (1e3 / per_tok) if per_tok and per_tok == per_tok else None,
        "num_decode_steps": got.get("generate_length"),
        "memory": {},
        "note": (
            "MPK runs the whole generation inside one persistent megakernel launch, so "
            "prefill, TTFT and per-step p50/p95 are not separable; demo.py reports a "
            "single amortized per-token latency. demo.py's --max-new-tokens only caps "
            "the PyTorch branch, so the MPK run decodes until EOS/max_seq_length; "
            "token agreement is compared over the overlapping prefix."
        ),
    }


# ==========================================================================
# orchestration
# ==========================================================================


def _spawn_worker(args, mode: str, logits_out: Optional[str]) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "res.json")
        cmd = [
            sys.executable, "-m", "experiments.segmented_mugraph.bench_qwen3",
            "--worker", "--mode", mode,
            "--model", args.model,
            "--prompt", args.prompt,
            "--gen-tokens", str(args.gen_tokens),
            "--warmups", str(args.warmups),
            "--reps", str(args.reps),
            "--extra-buckets", args.extra_buckets,
            "--device", str(args.device),
            "--mpk-timeout", str(args.mpk_timeout),
            "--worker-out", out_path,
        ]
        if args.model_path:
            cmd += ["--model-path", args.model_path]
        if args.no_superoptimize:
            cmd.append("--no-superoptimize")
        if logits_out:
            cmd += ["--logits-out", logits_out]
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        env["CUDA_VISIBLE_DEVICES"] = str(args.device)
        print(f"\n=== mode: {mode} ===", flush=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                              stdout=None if args.verbose else subprocess.DEVNULL,
                              stderr=None if args.verbose else subprocess.DEVNULL)
        if proc.returncode != 0 or not os.path.exists(out_path):
            return {"mode": mode, "status": "failed", "returncode": proc.returncode}
        with open(out_path) as f:
            res = json.load(f)
        res.setdefault("status", "ok")
        return res


def _compare_logits(path_a: str, path_b: str) -> Dict[str, Any]:
    a, b = torch.load(path_a), torch.load(path_b)
    af, bf = a.flatten().float(), b.flatten().float()
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=0).item()
    return {
        "cosine_sim": cos,
        "max_abs_err": (af - bf).abs().max().item(),
        "all_finite": bool(torch.isfinite(bf).all().item()),
        "top1_torch": int(af.argmax().item()),
        "top1_hybrid": int(bf.argmax().item()),
        "top1_match": int(af.argmax().item()) == int(bf.argmax().item()),
        "cosine_threshold": 0.99,
        "passed": bool(
            torch.isfinite(bf).all().item()
            and cos >= 0.99
            and int(af.argmax().item()) == int(bf.argmax().item())
        ),
    }


def _token_agreement(a: List[int], b: List[int]) -> Dict[str, Any]:
    n = min(len(a), len(b))
    matches = [i for i in range(n) if a[i] == b[i]]
    first_div = next((i for i in range(n) if a[i] != b[i]), None)
    return {
        "compared": n,
        "num_matching": len(matches),
        "agreement_fraction": (len(matches) / n) if n else float("nan"),
        "first_divergence_index": first_div,
        "note": (
            "Greedy decoding amplifies small numerical differences; once two runs "
            "diverge at one step the suffixes are unrelated, so agreement is "
            "reported rather than asserted exact."
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--model-path", default=None)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--gen-tokens", type=int, default=32)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--extra-buckets", default="",
                   help="Extra fixed token-count buckets to compile, e.g. '2,4'")
    p.add_argument("--no-superoptimize", action="store_true")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--mpk-timeout", type=int, default=3600)
    p.add_argument("--out", default="experiments/outputs/stage2_qwen3.json")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--quiet", dest="verbose", action="store_false")
    # worker-only
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)
    p.add_argument("--worker-out", help=argparse.SUPPRESS)
    p.add_argument("--logits-out", help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available; the Qwen3 experiment needs a GPU.")
        return 0

    if args.worker:
        res = run_mpk(args) if args.mode == "mpk" else run_torchlike_worker(args)
        write_json(args.worker_out, res)
        return 0

    modes = [m for m in args.modes.split(",") if m]
    logits_dir = tempfile.mkdtemp(prefix="segmug_logits_")
    results: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        lp = os.path.join(logits_dir, f"{mode}.pt") if mode != "mpk" else None
        results[mode] = _spawn_worker(args, mode, lp)

    correctness: Dict[str, Any] = {}
    ta, hb = os.path.join(logits_dir, "torch.pt"), os.path.join(logits_dir, "hybrid-mugraph.pt")
    if os.path.exists(ta) and os.path.exists(hb):
        correctness["first_decode_logits_torch_vs_hybrid"] = _compare_logits(ta, hb)
    for other in ("hybrid-mugraph", "mpk"):
        if "torch" in results and other in results:
            a = results["torch"].get("token_ids") or []
            b = results[other].get("token_ids") or []
            if a and b:
                correctness[f"token_agreement_torch_vs_{other}"] = _token_agreement(a, b)

    payload = {
        "benchmark": "stage2_qwen3_hybrid",
        "environment": env_info(args.device),
        "config": {
            "model": args.model_path or args.model,
            "prompt": args.prompt,
            "gen_tokens": args.gen_tokens,
            "warmups": args.warmups,
            "reps": args.reps,
            "dtype": "torch.bfloat16",
            "batch_size": 1,
            "decoding": "greedy",
            "extra_buckets": args.extra_buckets,
        },
        "results": results,
        "correctness": correctness,
        "fair_comparison_note": (
            "Stage 1 is an apples-to-apples kernel comparison. This Stage-2 table is "
            "NOT: torch and hybrid-mugraph run under PyTorch/HF Python orchestration, "
            "while mpk runs the entire decode inside one persistent megakernel with "
            "its own scheduler, attention and sampling kernels."
        ),
    }
    write_json(args.out, payload)
    print("\n" + _render(payload))
    print(f"\nJSON written to {args.out}")
    return 0


def _render(payload: Dict[str, Any]) -> str:
    rows = []
    for mode, r in payload["results"].items():
        if r.get("status") != "ok":
            rows.append([mode, r.get("status", "?"), "-", "-", "-", "-", "-", "-", "-"])
            continue
        mg = r.get("mugraph", {})
        rows.append([
            mode,
            "ok",
            num(r.get("cold", {}).get("total_s"), 1),
            num(r.get("prefill_ms"), 3),
            num(r.get("time_to_first_token_ms"), 3),
            num(r.get("mean_inter_token_ms"), 3),
            num(r.get("p50_inter_token_ms"), 3),
            num(r.get("p95_inter_token_ms"), 3),
            num(r.get("decode_tokens_per_s"), 1),
        ])
    headers = ["mode", "status", "cold s", "prefill ms", "TTFT ms",
               "mean ITL ms", "p50 ITL", "p95 ITL", "dec tok/s"]
    out = ["Stage 2 -- Qwen3 hybrid  [" + str(payload["config"]["model"]) + "]",
           fmt_table(rows, headers)]

    hy = payload["results"].get("hybrid-mugraph", {}).get("mugraph")
    if hy:
        out += ["", (f"muGraph region variants compiled: {hy['num_region_variants']}  |  "
                     f"muGraph MLP calls: {hy['mugraph_calls']}  |  "
                     f"PyTorch fallback calls: {hy['fallback_calls']}  |  "
                     f"cache hits/misses: {hy['cache_hits']}/{hy['cache_misses']}")]
    c = payload.get("correctness", {})
    if "first_decode_logits_torch_vs_hybrid" in c:
        d = c["first_decode_logits_torch_vs_hybrid"]
        out += ["", (f"first-decode logits torch vs hybrid: cos={d['cosine_sim']:.6f} "
                     f"top1_match={d['top1_match']} -> "
                     f"{'PASS' if d['passed'] else 'FAIL'}")]
    for k, v in c.items():
        if k.startswith("token_agreement"):
            out += [f"{k}: {v['num_matching']}/{v['compared']} tokens "
                    f"({v['agreement_fraction']*100:.1f}%), first divergence at "
                    f"{v['first_divergence_index']}"]
    out += ["", payload["fair_comparison_note"]]
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
