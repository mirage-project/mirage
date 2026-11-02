"""The whole static compiler path for gfx950 on the real K3 MoE graph, without a GPU: the demo's graph (zero-filled CPU tensors of
the weights' shapes), a plan for 256 CUs (amd/tools/plan_for_cus.py), compile_plan -> the schedules, generate_code -> layer.cu
(MK_BACKEND=hip: the HIP templates), hipcc for gfx950 (compile only: the extension is not loaded). Then the kernel's registers and
scratch from the assembly.

MK_BACKEND=hip python amd/tools/codegen_check.py [--plan demo/kimi_k3_moe/solver_plan_256cu.json] [--out DIR] [--profile]
    [--flag -DSTATIC_GEMM_LDSW=3 ...] [--gpus 1]
--gpus 1, 2, 4: the layer for that many GPUs (amd/tests/test_layer_one_gpu.py's build), the plan made by
amd/tools/plan_one_gpu.py.
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys

os.environ.setdefault("MK_BACKEND", "hip")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", default=os.path.join(ROOT, "demo", "kimi_k3_moe", "solver_plan_256cu.json"))
    p.add_argument("--out", default="codegen_check")
    p.add_argument("--cus", type=int, default=256)
    p.add_argument("--arch", default="gfx950")
    p.add_argument("--profile", action="store_true", help="the timing build")
    p.add_argument("--flag", action="append", default=[])
    p.add_argument("--gpus", type=int, default=8, choices=(1, 2, 4, 8))
    p.add_argument("--search", nargs="+", default=None, metavar="COSTS", help="the plan: search.search_plan's from these cost "
                   "records (of this code, recorded on --gpu-name), starting from --plan; written to <out>/plan.json")
    p.add_argument("--gpu-name", default="AMD Instinct MI355X VF", help="--search: the records' machine")
    p.add_argument("--time", type=float, default=120, help="--search: the solver's time limit (s)")
    p.add_argument("--grids", default=None, help='JSON {graph_idx: grid}: a simple plan with these grids (plan_one_gpu.simple_plan)')
    p.add_argument("--pack", action="append", default=[],
                   help="NODE:PER_CU: the node's tasks PER_CU in a row on fewer CUs (plan_one_gpu.pack_plan); repeatable")
    p.add_argument("--regrid", action="append", default=[], help="NODE:GX,GY,GZ:AFTER,...: plan_one_gpu.regrid_plan")
    p.add_argument("--front", default=None, help="GX,GY,GZ: plan_one_gpu.front_plan with the router at this grid")
    p.add_argument("--shared-early", action="store_true", help="plan_one_gpu.shared_early_plan (after --fit)")
    p.add_argument("--shared-late", action="store_true", help="plan_one_gpu.shared_late_plan (after --fit)")
    p.add_argument("--fit", action="store_true", help="fit the plan to the graph's tasks per GPU (plan_one_gpu.fewer_gpus_plan, "
                   "as test_layer_one_gpu.py does for --gpus 8), after --fold / --lup-finish")
    p.add_argument("--fold", action="store_true", help="RMSNorm folded into latent_up (MPK_RMSNORM_FOLD)")
    p.add_argument("--zq-finish", action="store_true", help="latent_down quantizes and sends z_q itself (MPK_ZQ_FINISH)")
    p.add_argument("--lup-finish", action="store_true", help="latent_up's last K part per row block finishes (MPK_LATENT_UP_FINISH)")
    p.add_argument("--lup-push", default=None, help="GX,GY,GZ: latent_up pushing its rows (MPK_LATENT_UP_PUSH), at this grid")
    p.add_argument("--options", action="store_true", help="print every node's candidate grids")
    p.add_argument("--spread", default=None, help="comma-separated graph_idx: plan_one_gpu.spread_plan with --grids")
    p.add_argument("--phases", default=None, help="JSON [[graph_idx, ...], ...]: plan_one_gpu.phase_plan with --grids")
    a = p.parse_args()
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        # no GPU here: PersistentKernel allocates its meta tensors on the device; this check only builds the graph and the code,
        # so they go to the CPU (on the AMD box ROCm's torch has the device)
        for fn in ("empty", "zeros", "ones", "full", "arange", "tensor", "empty_like", "zeros_like"):
            def on_cpu(*args, _orig=getattr(torch, fn), **kw):
                if str(kw.get("device", "")).startswith("cuda"):
                    kw["device"] = "cpu"
                return _orig(*args, **kw)
            setattr(torch, fn, on_cpu)

        class Props:  # what ROCm's torch reports for an MI355X
            major, minor, multi_processor_count, gcnArchName = 9, 5, a.cus, a.arch
            name, total_memory = "no GPU (codegen check)", 288 << 30

        torch.cuda.get_device_properties = lambda *_: Props
        torch.cuda.current_device = lambda: 0
        torch.cuda.device_count = lambda: 8
    from mirage.mpk import compiler, gpu_backend, static_schedule
    from mirage.mpk.static_megakernel import StaticMegakernel
    assert gpu_backend.backend() == "hip", "MK_BACKEND=hip"
    spec = importlib.util.spec_from_file_location("k3_moe_demo", os.path.join(ROOT, "demo", "kimi_k3_moe", "demo.py"))
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    # 1. the graph on zero-filled CPU tensors of the weights' shapes
    torch_dtype = {"bf16": torch.bfloat16, np.float32: torch.float32, np.uint8: torch.uint8}
    w = {name: torch.zeros(shape, dtype=torch_dtype[dtype]) for name, _, shape, dtype, _ in demo.WEIGHT_FILES}
    w["moe_out"] = torch.zeros(demo.T, demo.H, dtype=torch.bfloat16)
    w["routed_sum_partial"] = torch.zeros(demo.T * demo.TOPK, demo.L, dtype=torch.float32)
    w["shared_down_partial"] = torch.zeros(demo.T, demo.H, dtype=torch.float32)
    mpk = StaticMegakernel(num_gpus=a.gpus, num_sms=a.cus)
    if a.lup_push:
        os.environ["MPK_LATENT_UP_PUSH"] = "1"
    if a.lup_finish:
        os.environ["MPK_LATENT_UP_FINISH"] = "1"
    if a.fold:
        os.environ["MPK_RMSNORM_FOLD"] = "1"
    if a.zq_finish:
        os.environ["MPK_ZQ_FINISH"] = "1"
    demo.build_graph(mpk, w)
    nodes = compiler.graph_nodes(mpk)
    print(f"graph: {len(nodes)} nodes: {', '.join(f'{n.name}#{n.graph_idx}' for n in nodes)}", flush=True)
    if a.options:
        for n in nodes:
            if n.info["changeable_grid_dims"]:
                print(f"  options {n.name}#{n.graph_idx}: {compiler.candidate_grids(n)}", flush=True)
    # 2. the plan -> schedules
    os.makedirs(a.out, exist_ok=True)
    plan = compiler.plan_from_file(a.plan)
    if a.search:
        from mirage.mpk import plan_viewer, search
        costs = search.Costs(a.search, code=static_schedule.code_tag(), gpu=a.gpu_name)
        print(f"search: {len(costs.records)} records of code {static_schedule.code_tag()} on {a.gpu_name}", flush=True)
        os.makedirs(a.out, exist_ok=True)
        plan, summary = search.search_plan(mpk, costs, time_limit_s=a.time, start_from=plan)
        compiler.write_plan_file(os.path.join(a.out, "plan.json"), plan)
        plan_viewer.save(os.path.join(a.out, "plan_view.json"), plan, summary, f"searched from {a.plan}")
        print(f"search: {summary}", flush=True)
    if a.grids or a.phases or a.spread:
        sys.path.insert(0, HERE)
        from plan_one_gpu import phase_plan, simple_plan
        grids = dict(plan["grids"])
        grids.update({int(k): tuple(v) for k, v in json.loads(a.grids or "{}").items()})
        if a.spread:
            from plan_one_gpu import spread_plan
            plan = spread_plan(mpk, plan, json.loads(a.grids or "{}"), [int(i) for i in a.spread.split(",")])
        else:
            plan = phase_plan(mpk, grids, json.loads(a.phases)) if a.phases else simple_plan(mpk, grids)
        print(f"plan: tasks per SM list (GPU 0): max {max(len(l) for l in plan['lists'][0])}, "
              f"first entries of SM 0: {plan['lists'][0][0][:6]}", flush=True)
    if a.fold or a.zq_finish:
        sys.path.insert(0, HERE)
        from plan_one_gpu import drop_plan
        plan = drop_plan(plan, ({38, 41} if a.fold else set()) | ({29} if a.zq_finish else set()))
        print(f"rmsnorm fold: GPU 0 CU 0 list {[t[0] for t in plan['lists'][0][0]]}", flush=True)
    if a.lup_finish:
        sys.path.insert(0, HERE)
        from plan_one_gpu import lup_finish_plan
        plan = lup_finish_plan(plan)
        print(f"latent_up finish: GPU 0 CU 0 list {[t[0] for t in plan['lists'][0][0]]}", flush=True)
    if a.fit:
        sys.path.insert(0, HERE)
        from plan_one_gpu import fewer_gpus_plan
        plan = fewer_gpus_plan(mpk, plan)
        print(f"fitted plan: GPU 0 CU 0 list {[t[0] for t in plan['lists'][0][0]]}", flush=True)
    if a.lup_push:
        sys.path.insert(0, HERE)
        from plan_one_gpu import lup_push_plan
        plan = lup_push_plan(mpk, plan, tuple(int(v) for v in a.lup_push.split(",")))
        print(f"latent_up push: grid {plan['grids'][40]}; GPU 0 CU 0 list {[t[0] for t in plan['lists'][0][0]]}", flush=True)
    if a.front:
        sys.path.insert(0, HERE)
        from plan_one_gpu import front_plan
        plan = front_plan(mpk, plan, 26, tuple(int(v) for v in a.front.split(",")), 28, 30)
        print(f"front plan: GPU 0 CU lists 0, 100, 200: {[[t[0] for t in plan['lists'][0][i]] for i in (0, 100, 200)]}",
              flush=True)
    if a.shared_early:
        sys.path.insert(0, HERE)
        from plan_one_gpu import shared_early_plan
        plan = shared_early_plan(mpk, plan)
        from collections import Counter
        heads = Counter(tuple(t[0] for t in cu[:3]) for cu in plan["lists"][0])
        print(f"shared early: GPU 0 list heads {heads.most_common(8)}", flush=True)
    if a.shared_late:
        sys.path.insert(0, HERE)
        from plan_one_gpu import shared_late_plan
        plan = shared_late_plan(mpk, plan)
        from collections import Counter
        seqs = Counter(tuple(t[0] for t in cu) for cu in plan["lists"][0])
        print(f"shared late: GPU 0 lists {seqs.most_common(10)}", flush=True)
    for pc in a.pack:   # NODE:PER_CU (plan_one_gpu.pack_plan)
        sys.path.insert(0, HERE)
        from plan_one_gpu import pack_plan
        f = pc.split(":")
        plan = pack_plan(plan, int(f[0]), int(f[1]))
        print(f"pack: node {f[0]}, {f[1]} tasks per CU", flush=True)
        from collections import Counter
        seqs = Counter(tuple(t[0] for t in cu) for cu in plan["lists"][0])
        print(f"pack: GPU 0 lists {seqs.most_common(10)}", flush=True)
    for rg in a.regrid:   # NODE:GX,GY,GZ:AFTER1,... (plan_one_gpu.regrid_plan)
        sys.path.insert(0, HERE)
        from plan_one_gpu import regrid_plan
        f = rg.split(":")
        node = int(f[0])
        plan = regrid_plan(mpk, plan, node, tuple(int(v) for v in f[1].split(",")),
                           {int(v) for v in f[2].split(",") if v} if len(f) > 2 else set())
        for g, lists in enumerate(plan["lists"]):
            n_task = sum(t[0] == node for cu in lists for t in cu)
            first = [[t[0] for t in cu].index(node) for cu in lists if any(t[0] == node for t in cu)]
            cus = len(first)
            if g == 0:
                print(f"regrid: node {node} at {plan['grids'][node]}: GPU 0 {n_task} tasks on {cus} CUs; CU 0 list "
                      f"{[t[0] for t in plan['lists'][0][0]]}", flush=True)
    if a.gpus < 8:
        sys.path.insert(0, HERE)
        from plan_one_gpu import fewer_gpus_plan
        plan = fewer_gpus_plan(mpk, plan)
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(a.out, "schedules"))
    print(f"compile_plan: {len(paths)} schedules, grids {info['grids']}, cluster size {info['cluster_size']}", flush=True)
    # 3. layer.cu
    schedules = [json.load(open(path)) for path in paths]
    code = static_schedule.generate_code(mpk, schedules, a.profile)
    cu = os.path.join(a.out, "layer.cu")
    with open(cu, "w") as f:
        f.write(code)
    print(f"layer.cu: {len(code.splitlines())} lines", flush=True)
    # 4. hipcc
    cmd = static_schedule.compile_command(cu, os.path.join(a.out, "layer.so"), a.flag + ["-Wall", "--save-temps"], arch=a.arch)
    r = subprocess.run(cmd, cwd=a.out, capture_output=True, text=True)
    warnings = r.stderr.count("warning:")
    print(f"hipcc exit {r.returncode}, {warnings} warnings", flush=True)
    if r.returncode or warnings:
        print("\n".join(l for l in r.stderr.splitlines() if "error" in l or "warning:" in l)[:3000])
    if r.returncode:
        sys.exit(1)
    s = open(os.path.join(a.out, f"layer-hip-amdgcn-amd-amdhsa-{a.arch}.s")).read()
    regs = {k: re.search(rf"; {k}: (\d+)", s).group(1) for k in ("NumVgprs", "NumAgprs", "ScratchSize", "Occupancy")}
    print(f"layer_kernel: {regs}")
    print(json.dumps({"check": "codegen", "nodes": len(nodes), "hipcc": r.returncode, "warnings": warnings, **regs}))


if __name__ == "__main__":
    main()
