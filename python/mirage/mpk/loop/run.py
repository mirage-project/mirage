"""The generic harness run of one part (part.py) and one plan or candidate: demo.py's steps for any part.

  1. the part's tensors and graph       hooks.build(input dir of the bucket, GPUs, tokens, graph options)
  2. the plan                           --plan FILE | random:SEED (explore.random_plan) | --candidate cand.json;
                                        --search COSTS: the solver's plan from it (search.search_plan)
  3. schedules, layer.cu, built         compiler.compile_plan, StaticMegakernel.compile_static (--record: the timing build)
  4. check                              hooks.check: one launch, the part's output checks; failed: no timing (exit 1)
  5. timed launches                     --reps, L2 emptied before each (part.yaml flush_bytes); the watched tensors bit by bit
  6. results                            <out>/result.json: spans, grids, cluster size, code tag, the layer's md5, the machine;
                                        --record FILE: cost records appended to FILE, the raw stamps of the first 5 launches
                                        (<out>/task_times.npy, timing.npy: triage.py, model.py read them)
                                        --predict COSTS: the model's timeline of the plan (<out>/plan_view.json)

python -m mirage.mpk.loop.run --part demo/kimi_k3_moe --plan demo/kimi_k3_moe/solver_plan.json [--bucket 8] [--input-set NAME]
       [--search COSTS ...] [--record FILE] [--predict COSTS ...] [--reps 20] [--out DIR]
"""
import argparse
import hashlib
import json
import os
import socket
import statistics
import subprocess
import sys
from collections import Counter
from typing import Optional

from . import candidate as cand_mod
from . import part as part_mod


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def machine_info(torch) -> dict:
    try:
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    except (OSError, IndexError):
        nvcc = None
    return {"gpu": torch.cuda.get_device_name(0), "cuda": nvcc, "host": socket.gethostname()}


def get_plan(args, part, mpk, ngpu: int, num_sms: int):
    """(plan, candidate or None, the plan's name)."""
    from .. import compiler
    if args.candidate:
        cand = cand_mod.load(args.candidate)
        if cand["part"] != part.name:
            raise ValueError(f"{args.candidate} is a candidate of part {cand['part']}, not {part.name}")
        return cand_mod.plan_from_doc(cand["plan"]), cand, f"candidate {cand['id']}"
    if args.plan == "start":   # the part's own start plan (hooks.start_plan)
        return part.hooks.start_plan(mpk, ngpu, num_sms), None, "start plan"
    if args.plan.startswith("random:"):
        from .explore import random_plan
        return random_plan(mpk, ngpu, num_sms, int(args.plan.split(":", 1)[1])), None, args.plan
    return compiler.plan_from_file(args.plan), None, args.plan


def costs_of(args, paths, torch):
    from .. import search, static_schedule
    code = {"current": static_schedule.code_tag(), "any": None}.get(args.costs_code, args.costs_code)
    return search.Costs(paths, code=code, gpu=torch.cuda.get_device_name(0))


def main(argv: Optional[list] = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", required=True, help="the part's directory (its part.yaml)")
    p.add_argument("--bucket", type=int, default=None, help="tokens per step (default: the part's first bucket)")
    p.add_argument("--input-set", default=None, help="the bucket's input set (default: its first)")
    p.add_argument("--input", default=None, help="an input directory instead of the bucket's input set")
    p.add_argument("--plan", default=None, help="a plan file (compiler.write_plan_file), random:SEED, or start (the part's "
                   "hooks.start_plan)")
    p.add_argument("--candidate", default=None, help="a candidate.json (candidate.py) instead of --plan")
    p.add_argument("--graph-option", action="append", default=[], metavar="NAME=VALUE", help="a part graph option (repeatable)")
    p.add_argument("--search", nargs="+", default=None, metavar="COSTS", help="cost record files: the solver's plan, from the plan")
    p.add_argument("--predict", nargs="+", default=None, metavar="COSTS", help="cost record files: the loop model's timeline of the "
                   "plan (model.Model) -> prediction.json, compared with the measured stamps by model.run_error")
    p.add_argument("--model-stat", default="p75", help="with --predict: the records' value the model uses: p75 or median")
    p.add_argument("--costs-code", default="current", help="the records of these kernel templates: current, any, or a tag")
    p.add_argument("--time", type=float, default=300, help="with --search: the solver's time limit (s)")
    p.add_argument("--record", default=None, metavar="FILE", help="the timing build; cost records appended to FILE")
    p.add_argument("--run-name", default=None, help="with --record: the run's name in the records (default: the --out name)")
    p.add_argument("--nvcc-flag", action="append", default=[])
    p.add_argument("--gpus", type=int, default=None, help="default: the part's tp")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--wait", type=float, default=60, help="seconds to wait for one launch before reporting it as not finished")
    p.add_argument("--out", default="part_run")
    args = p.parse_args(argv)
    if bool(args.plan) == bool(args.candidate):
        p.error("one of --plan, --candidate")

    import torch
    from .. import compiler
    from ..static_harness import Harness
    from ..static_schedule import code_tag

    part = part_mod.load(args.part)
    os.makedirs(args.out, exist_ok=True)
    ngpu = args.gpus or part.tp
    tokens = args.bucket or part.buckets()[0]
    input_dir = os.path.expanduser(args.input) if args.input else part.input_dir(tokens, args.input_set)
    overrides = dict(kv.split("=", 1) for kv in args.graph_option)
    overrides = {k: json.loads(v) if v[:1] in "0123456789-[{tf" else v for k, v in overrides.items()}
    cand = cand_mod.load(args.candidate) if args.candidate else None
    if cand:
        overrides = {**cand.get("graph_options", {}), **overrides}
        tokens = cand["bucket"]
        input_dir = os.path.expanduser(args.input) if args.input else part.input_dir(tokens, args.input_set)
    options = part.graph_options(overrides)

    # 1. tensors and graph
    ws, mpk = part.hooks.build(input_dir, ngpu, tokens, options)
    num_sms = mpk._num_sms()
    # 2. the plan
    plan, cand, plan_name = get_plan(args, part, mpk, ngpu, num_sms)
    nvcc_flags = list(args.nvcc_flag) + (list(cand.get("nvcc_flags", [])) if cand else [])
    if args.search:
        from .. import search
        plan, summary = search.search_plan(mpk, costs_of(args, args.search, torch), time_limit_s=args.time, start_from=plan)
        compiler.write_plan_file(os.path.join(args.out, "plan.json"), plan)
        plan_name = f"searched from {plan_name}"
    elif args.predict:
        from .model import Model, predict
        # the loop's model: "current" = each task type's records of its current code (static_schedule.task_code_tag)
        model = Model(args.predict, code={"any": None}.get(args.costs_code, args.costs_code), gpu=torch.cuda.get_device_name(0),
                      stat=args.model_stat)
        try:
            summary = predict(mpk, plan, model)
        except (ValueError, RuntimeError) as e:   # no records yet for some node (the first timing builds of a part make them)
            print(f"prediction skipped: {e}", flush=True)
            summary = None
        if summary is not None:
            with open(os.path.join(args.out, "prediction.json"), "w") as f:
                json.dump({"predicted_us": summary["predicted_us"], "timeline": summary["timeline"],
                           "fallbacks": summary["fallbacks"], "stat": args.model_stat}, f)
    if not args.search:
        compiler.write_plan_file(os.path.join(args.out, "plan.json"), plan)
    # 3. schedules, layer.cu
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(args.out, "schedules"))
    sk = mpk.compile_static(paths, gpu_tensors=ws, out_dir=os.path.join(args.out, "build"), extra_flags=nvcc_flags,
                            profile=args.record is not None)
    nodes = compiler.graph_nodes(mpk)
    result = {"part": part.name, "bucket": tokens, "input": input_dir, "plan": plan_name,
              # what triage.py and model.run_error need without the graph: each node's first output name (part.yaml phases),
              # the nodes whose output is in the exchange region (cross-GPU waits), the one-task-per-SM node (the pool)
              "node_names": {n.graph_idx: mpk._tensor_names[n.outputs[0].guid] for n in nodes},
              "exchange_nodes": [n.graph_idx for n in nodes if n.info["exchange"]],
              "pool_nodes": [n.graph_idx for n in nodes if n.info["one_task_per_sm"]],
              "candidate": cand["id"] if cand else None, "grids": info["grids"], "cluster_size": info["cluster_size"],
              "graph_options": options, "nvcc_flags": nvcc_flags, "code": code_tag(), "layer_md5": md5_of(sk.so_path),
              "profile": args.record is not None, "machine": machine_info(torch), "reps": args.reps,
              "flush_bytes": part.flush_bytes}
    h = Harness(sk)
    # 4. check
    ok = part.hooks.check(h, ws, input_dir, args.out)
    result["check"] = bool(ok)
    if not ok:
        print("CHECK FAILED: no timing", flush=True)
        _write(args.out, result)
        sk.finalize()
        sys.exit(1)
    # 5. timed launches
    watch = part.hooks.watch(ws) if hasattr(part.hooks, "watch") else {}
    before = getattr(part.hooks, "before_launch", None)
    launches = []

    def after_launch(i):
        if args.record:
            launches.append((h.task_times(), h.timing()))
        if before is not None and i + 1 < args.reps:
            before(ws, i + 1)
    if before is not None:
        before(ws, 0)
    spans, changed = h.repeat(args.reps, watch, part.flush_bytes, args.wait, after_launch)
    result["spans_us"] = spans
    result["e2e_gpus_us"] = h.e2e_gpus                       # per launch, per GPU (launched in GPU order by one host thread)
    last = [e[-1] for e in h.e2e_gpus]                        # the last-launched GPU: no wait for later launches
    worst = [max(e) for e in h.e2e_gpus]
    result["unstable"] = {name: Counter(int(c[name].sum()) for c in changed).most_common() for name in watch}
    if any(n for name in watch for n, _ in result["unstable"][name] if n):
        # the same inputs gave different outputs: a failed check (KDA k57-k66: a task wrote past a tensor; the check launch
        # itself passed), no records and no verdict from this run
        print(f"values that changed between timed launches (count -> launches): {result['unstable']}", flush=True)
        print("CHECK FAILED: the output changed between timed launches", flush=True)
        result["check"] = False
        _write(args.out, result)
        sk.finalize()
        sys.exit(1)
    # 6. records, results
    if args.record:
        import numpy as np
        from .. import cost_records
        run = args.run_name or os.path.basename(os.path.normpath(args.out))
        records = cost_records.records_of_run(mpk, paths, launches, run, {k: result["machine"][k] for k in ("gpu", "cuda")})
        for r in records:
            r.setdefault("context", {})
            r["context"].update({"bucket": tokens, "cluster_size": info["cluster_size"], "candidate": result["candidate"]})
        cost_records.append_records(args.record, records)
        np.save(os.path.join(args.out, "task_times.npy"), np.array([t for t, _ in launches[:5]], dtype=np.int64))
        np.save(os.path.join(args.out, "timing.npy"), np.array([t for _, t in launches[:5]], dtype=np.int64))
        result["records_added"] = len(records)
        print(f"cost records: {len(records)} added to {args.record} (code {result['code']})", flush=True)
    print(f"machine {result['machine']['host']} {result['machine']['gpu']} x {ngpu}, part {part.name}, bucket {tokens}, "
          f"plan {plan_name}, grids {info['grids']}, {args.reps} launches (L2 emptied before each): e2e last-launched GPU "
          f"min {min(last):.1f} median {statistics.median(last):.1f} max {max(last):.1f} us (max over GPUs median "
          f"{statistics.median(worst):.1f}) | span min {min(spans):.1f} "
          f"median {statistics.median(spans):.1f} max {max(spans):.1f} us", flush=True)
    _write(args.out, result)
    sk.finalize()


def _write(out: str, result: dict) -> None:
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
