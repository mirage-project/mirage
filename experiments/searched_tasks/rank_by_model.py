"""Rank searched task schedules by WHOLE-MODEL throughput.

search() enumerates schedules and checks they compute the same thing; it
never measures speed. superoptimize() measures, but times each candidate as a
standalone kernel -- the wrong number for a body that will run inside a
megakernel on a persistent worker.

Per-task in-MPK latency is closer but still wrong as an objective. Measured
here on Qwen3-0.6B: a silu_mul schedule 1.20x faster per task (8800 -> 7360
ns) left the model 2.5% slower end to end, because silu_mul is the cheapest
of the three MLP tasks and the matmuls that dominate were untouched. The only
number that cannot mislead is the one you actually care about, so each
candidate is ranked by building the entire model with it and measuring the
model.

That costs one megakernel compile plus one decode run per candidate, and it
has to be a fresh process each time (one megakernel, one CUDA context). The
cost is paid once per task TYPE, not per layer -- all 28 Qwen3 layers use the
same silu_mul schedule -- and the winner is cached to JSON.

    python experiments/searched_tasks/rank_by_model.py --repeats 2
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(REPO, "tests", "ci-tests", "run_batch_perf.py")

_THROUGHPUT = re.compile(r"throughput:\s+([0-9.]+)\s+tokens/s")
_SAMPLE = re.compile(r"Sample output \(first 32 tokens of request 0\): (.*)")


# task name -> (mlp_impl to select it, python source for its TaskSpec).
# forloop_range is deliberately NOT pinned: for a matmul the K-loop split is
# the main free choice, and it is what gives this a real schedule space
# (silu_mul has none -- it deduplicates to a single candidate).
TASKS = {
    "silu_mul": (
        "searched_silu_mul",
        'TaskSpec("silu_mul",\n'
        '         lambda kn, t: kn.mul(kn.silu(t[0]), t[1]),\n'
        '         [TensorSpec(({tokens}, {intermediate})),\n'
        '          TensorSpec(({tokens}, {intermediate}))])',
    ),
    "linear": (
        "searched_linear",
        'TaskSpec("linear",\n'
        '         lambda kn, t: kn.matmul(t[0], t[1]),\n'
        '         [TensorSpec(({tokens}, {hidden})),\n'
        '          TensorSpec(({hidden}, {intermediate}))])',
    ),
}


def enumerate_schedules(task, tokens, hidden, intermediate, out_path):
    """Search once, in its own process, and write every valid candidate."""
    spec_src = TASKS[task][1].format(tokens=tokens, hidden=hidden,
                                      intermediate=intermediate)
    src = f"""
import json, sys
from mirage.mpk.lowering.task_search import TaskSpec, TensorSpec, search_task_schedules
scheds = search_task_schedules(
    {spec_src},
    grid_dim=({intermediate} // 64, 1, 1))
with open({out_path!r}, "w") as f:
    json.dump([s.to_dict() for s in scheds], f)
for i, s in enumerate(scheds):
    print(f"CANDIDATE {{i}} {{s.describe()}}", flush=True)
"""
    proc = subprocess.run([sys.executable, "-c", src], cwd=REPO,
                           capture_output=True, text=True, timeout=3600)
    for line in proc.stdout.splitlines():
        if line.startswith("CANDIDATE"):
            print("  " + line, flush=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
        raise SystemExit(f"schedule enumeration failed:\n{tail}")
    with open(out_path) as f:
        return json.load(f)


def run_model(env_extra, args, label):
    """Build + run the whole model once. -> (tokens/s, sample text) or None."""
    env = dict(os.environ)
    env.update(env_extra)
    env.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    cmd = [sys.executable, "-u", BENCH,
           "--model", args.model,
           "--max-num-batched-tokens", str(args.tokens),
           "--max-num-batched-requests", str(args.requests),
           "--max-seq-length", str(args.seq_len), "--ignore-eos"]
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True,
                           text=True, timeout=args.timeout)
    m = _THROUGHPUT.search(proc.stdout)
    if not m:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-12:])
        print(f"  {label}: FAILED\n{tail}", flush=True)
        return None
    s = _SAMPLE.search(proc.stdout)
    return float(m.group(1)), (s.group(1)[:60] if s else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--task", choices=sorted(TASKS), default="silu_mul",
                     help="which MPK task type's schedule to search and rank")
    ap.add_argument("--intermediate", type=int, default=3072,
                     help="Qwen3-0.6B MLP intermediate size")
    ap.add_argument("--hidden", type=int, default=1024,
                     help="Qwen3-0.6B hidden size")
    ap.add_argument("--repeats", type=int, default=1,
                     help="model runs per candidate; >1 puts an error bar on "
                          "differences that are only a few percent")
    ap.add_argument("--gpu", default="7")
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "best_schedule.json"))
    args = ap.parse_args()

    print(f"== enumerating {args.task} schedules (search runs once) ==",
          flush=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        cand_path = f.name
    cands = enumerate_schedules(args.task, args.tokens, args.hidden,
                                args.intermediate, cand_path)
    print(f"  {len(cands)} distinct valid schedule(s)", flush=True)

    print("\n== baselines ==", flush=True)
    results = []
    for label, env in (
        ("handwritten .cuh tasks", {"MPK_COMPILED_MLP": "0"}),
        ("generated, hand-written bgraph",
         {"MPK_COMPILED_MLP": "1", "MPK_COMPILED_MLP_IMPL": "separate"}),
    ):
        runs = [run_model(env, args, label) for _ in range(args.repeats)]
        runs = [r for r in runs if r]
        if runs:
            best = max(r[0] for r in runs)
            results.append((label, best, runs[0][1]))
            print(f"  {label:<34} {best:8.1f} tok/s "
                  f"(n={len(runs)}, all={[f'{r[0]:.0f}' for r in runs]})", flush=True)

    print("\n== candidates, ranked by whole-model throughput ==", flush=True)
    scored = []
    for i, c in enumerate(cands):
        env = {"MPK_COMPILED_MLP": "1",
               "MPK_COMPILED_MLP_IMPL": TASKS[args.task][0],
               "MPK_SEARCHED_SCHEDULE_JSON": cand_path,
               "MPK_SEARCHED_SCHEDULE_INDEX": str(i)}
        label = f"candidate {i}"
        runs = [run_model(env, args, label) for _ in range(args.repeats)]
        runs = [r for r in runs if r]
        if not runs:
            continue
        best = max(r[0] for r in runs)
        scored.append((best, i, c, runs[0][1]))
        kinds = [o["op_type"].replace("tb_", "").replace("_op", "")
                 for o in c["ops"]]
        print(f"  {label:<34} {best:8.1f} tok/s  fl={c['forloop_range']} "
              f"ops={kinds}\n{' ' * 38}sample={runs[0][1]!r}", flush=True)

    if not scored:
        raise SystemExit("no candidate produced a throughput number")
    scored.sort(reverse=True)
    best_tp, best_i, best_c, _ = scored[0]
    with open(args.out, "w") as f:
        json.dump(best_c, f, indent=2)

    print("\n== summary ==", flush=True)
    for label, tp, _ in results:
        print(f"  {label:<34} {tp:8.1f} tok/s", flush=True)
    print(f"  {'best searched (candidate ' + str(best_i) + ')':<34} "
          f"{best_tp:8.1f} tok/s", flush=True)
    if len(scored) > 1:
        print(f"  spread across candidates: "
              f"{scored[-1][0]:.1f} .. {scored[0][0]:.1f} tok/s", flush=True)
    print(f"\nwrote winner to {args.out}", flush=True)
    print("RANKDONE", flush=True)


if __name__ == "__main__":
    main()
