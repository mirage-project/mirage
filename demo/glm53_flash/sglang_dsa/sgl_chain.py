"""Run the SGLang runs of the GLM-5.3-Flash static-layer comparison one after another on the same 4 GPUs (CUDA_VISIBLE_DEVICES,
picked once by the caller): each run is its own process with its own environment (the static runs: the hooks through
boot/sitecustomize.py and the venv's kda_static_boot.pth, mirage's libz3), its output in <cwd>/<name>.log; this prints each
run's result lines. A run whose name starts with prof_ runs under nsys (<name>.nsys-rep, then <name>.sqlite).

  check_*    sgl_check.py in check mode (eager): every static layer against SGLang's at each step
  prof_*     the end-to-end comparison: nsys step periods (nsys_e2e.py) at 256- and 4000-token prompts, one run per length:
             SGLang with the static layers' KV cap (base) against every static layer with the L2 prefetch (all3ppf)
  base_* / all3ppf_*   the same pairs as wall-clock benches (sgl_spec_bench.py, 4000-token prompts)
  gsm8k_*    the accuracy of SGLang alone and with the static layers (sgl_gsm8k.py)

python sgl_chain.py [name ...]   (in the given order, a name again: <name>_2, ...; default: every run, in this order)
Env: MIRAGE_Z3_LIB (mirage's libz3 directory for the static runs, when SGLang's venv has another z3), GSM8K_TEST_JSONL,
GLM53F_MODEL (the checkpoint directory, default /raid/catalyst/models/GLM-5.3-Flash), GLM53F_NCCL_PORT (SGLang's
torch.distributed port for every run; default: per run a free port of 1000 by the first visible GPU, 22000 + 1000 x GPU
index, e.g. 22000-22999 for CUDA_VISIBLE_DEVICES=0,1,2,3 and 26000-26999 for 4,5,6,7). Two chains at once on two halves
of a box: each in its own directory (logs, nsys reports, build directories are under the current directory); their ports
come from disjoint ranges (SGLang's own default draws from 22000-29999 for both and binds later); the build cache is
shared (one lock file per layer.cu).
"""
import os
import random
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("DSA_STATIC_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(HERE))))   # this checkout
Z3 = os.environ.get("MIRAGE_Z3_LIB", "")   # the z3/lib of mirage's venv (z3 4.16; SGLang ships 4.15), if not the SGLang one
MODEL = os.environ.setdefault("GLM53F_MODEL", "/raid/catalyst/models/GLM-5.3-Flash")   # every run reads GLM53F_MODEL
GSM = os.environ.get("GSM8K_TEST_JSONL", "gsm8k_test.jsonl")   # GSM8K's test.jsonl (sgl_gsm8k.py --data-path)
BOOT = {"DSA_STATIC_ROOT": ROOT, "MIRAGE_ROOT": ROOT, "PYTHONPATH": os.path.join(HERE, "boot"),
        **({"LD_LIBRARY_PATH": Z3} if Z3 else {})}
# the static layers: every DSA and KDA layer to the FFN's input (dsa_static_hook, kda_static_hook); every MoE layer
# (moe_static_hook) with its end in the kernel (the all-reduce and the FFN's mHC post: mhc; at T = 16 / 32 also the next
# layer's attention mHC pre: mhc_pre); the MoE layers' own expert copies need the small KV pool (MTT)
ATTN = {**BOOT, "DSA_STATIC_LAYERS": "all", "DSA_STATIC_CHECK": "0", "KDA_STATIC_LAYERS": "all", "KDA_STATIC_ROOT": ROOT}
MOE = {"MOE_STATIC_LAYERS": "all", "MOE_STATIC_ROOT": ROOT, "MOE_STATIC_CHECK": "0"}
MOE_ONLY = {**BOOT, **MOE}
ALL3P = {**ATTN, **MOE}
PF = {"STATIC_L2_PREFETCH_MB": "16"}   # each layer's first 16 MB of GEMM weights into L2 at its start (core.cuh kernel_begin)
CHECK = {"DSA_STATIC_CHECK": "1", "KDA_STATIC_CHECK": "1"}
# the layers' T: the hooks' default, every T (1, 2, 4, 8, 16, 32, 64); a decode step of n rows runs the smallest T >= n
TOKENS = (1, 2, 4, 8, 16, 32, 64)
# the KV pool with the static MoE layers (their expert banks take the memory SGLang would give the pool): 8 requests of
# 8000 + 256 tokens need 66048; bs 16 x (8000 + 256) = 132096 in one wave (~13 KB a token); bs 64 x (4000 + 256) = 272384
MTT, MTT16, MTT64 = ["--max-total-tokens", "81920"], ["--max-total-tokens", "163840"], ["--max-total-tokens", "294912"]

RUNS = [("check_all3p", {**ALL3P, **CHECK, "MOE_STATIC_CHECK": "1"}, ["sgl_check.py", "8", "1500", "65536"])]
# bs 1 / 2 / 4 decode in check mode: the attention layers, then 4 MoE layers (the MoE in check mode keeps its weights)
for bs in (1, 2, 4):
    RUNS += [(f"check_bs{bs}_attn", {**ATTN, **CHECK}, ["sgl_check.py", "24", "1500", "0", str(bs)]),
             (f"check_bs{bs}_moe4l", {**MOE_ONLY, "MOE_STATIC_CHECK": "1", "MOE_STATIC_LAYERS": "3,4,5,6"},
              ["sgl_check.py", "24", "1500", "81920", str(bs)])]
# the T = 16 / 32 layers in check mode (enough decode steps that every request has joined the batch before they end): the
# attention layers, then 4 MoE layers; a decode step of T rows
for t in (16, 32):
    RUNS += [(f"check_t{t}_bs{t}_attn", {**ATTN, **CHECK}, ["sgl_check.py", "40", "300", "0", str(t)]),
             (f"check_t{t}_bs{t}_moe4l", {**MOE_ONLY, "MOE_STATIC_CHECK": "1", "MOE_STATIC_LAYERS": "3,4,5,6"},
              ["sgl_check.py", "40", "300", "81920", str(t)])]
# T = 64: the decode layers (bs 64)
RUNS += [("check_t64_bs64_attn", {**ATTN, **CHECK}, ["sgl_check.py", "72", "300", "0", "64"]),
         ("check_t64_bs64_moe4l", {**MOE_ONLY, "MOE_STATIC_CHECK": "1", "MOE_STATIC_LAYERS": "3,4,5,6"},
          ["sgl_check.py", "72", "300", "81920", "64"])]
# the scenarios: plain decode at every batch size
E2E_SCEN = {"off_bs1": (1, MTT), "off_bs2": (2, MTT), "off_bs4": (4, MTT), "off_bs8": (8, MTT), "off_bs16": (16, MTT16),
            "off_bs32": (32, MTT16), "off_bs64": (64, MTT64)}
# nsys: 64 profiled steps of up to 256 new tokens (the steps after the prefill chunks), whole graphs traced
# (--cuda-graph-trace=graph: per-node records perturb the timeline: they cost the static kernels' PDL overlap)
GRAPH_TRACE = set()
for scen, (bs, mtt) in E2E_SCEN.items():
    for p in (256, 4000):
        a = ["sgl_spec_bench.py", "--bs", str(bs), "--prompt-tokens", str(p), *mtt, "--reps", "1",
             "--profile", "--profile-steps", "64", "--profile-tokens", "256"]
        RUNS += [(f"prof_base_{scen}_p{p}", {}, a), (f"prof_all3ppf_{scen}_p{p}", {**ALL3P, **PF}, a)]
        GRAPH_TRACE |= {f"prof_base_{scen}_p{p}", f"prof_all3ppf_{scen}_p{p}"}
    a = ["sgl_spec_bench.py", "--bs", str(bs), *mtt]
    RUNS += [(f"base_{scen}_mtt", {}, a), (f"all3ppf_{scen}", {**ALL3P, **PF}, a)]
# node-traced variants at 4000 tokens (every kernel its own record: decode_compare.py's sublayer spans; they perturb the
# period, so not for the end-to-end table)
for scen in ("off_bs32", "off_bs64"):
    bs, mtt = E2E_SCEN[scen]
    a = ["sgl_spec_bench.py", "--bs", str(bs), "--prompt-tokens", "4000", *mtt, "--reps", "1", "--profile",
         "--profile-steps", "64", "--profile-tokens", "256"]
    RUNS += [(f"prof_base_{scen}_p4000_node", {}, a), (f"prof_all3ppf_{scen}_p4000_node", {**ALL3P, **PF}, a)]
# GSM8K (5 shots, 200 questions, greedy, plain decode): SGLang alone 4 at a time; the static layers 8 and 4 at a time (4: the
# batch shrinks to 2 and 1 as questions end, the layers of those T too)
RUNS += [("gsm8k_off", {}, ["sgl_gsm8k.py", "--data-path", GSM, "--bs", "4"])]
RUNS += [("all3ppf_gsm8k_off", {**ALL3P, **PF}, ["sgl_gsm8k.py", "--data-path", GSM, "--bs", "8", *MTT]),
         ("all3ppf_gsm8k_off_bs4", {**ALL3P, **PF}, ["sgl_gsm8k.py", "--data-path", GSM, "--bs", "4", *MTT])]
NSYS = ["nsys", "profile", "--force-overwrite", "true", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
        "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--trace-fork-before-exec=true"]
KEYS = ("decode step", "output ", "Error", "error:", "Traceback", "replaces SGLang", "KERNEL", "launch ", "graph replay",
        "PASS", "FAIL")

# every layer kind on the best plan measured (one GPU): the DSA layers on the searched plans of their graph by context
# (dsa_fused_plan_long.json: contexts 3346-3600, 64.1 vs round robin 71.8 us; _short: 20-275, 60.8 vs 62.9), the MoE on
# the searched plan of its T's end (MOE_END) where one exists (T = 8 ends at mhc: round robin); the KDA layers' round robin
# (30.3; the searched 31.9). Every T alike: its own graph's plan file where one exists (a graph whose searched plan did not
# beat round robin, and the T = 1 / 2 / 4 graphs, not searched: none, round robin). A run that names a plan keeps it.
DSA_PLAN = {8: ("dsa_fused_plan_long.json", "dsa_fused_plan_short.json"),
            16: ("dsa_fused_plan16_long.json", "dsa_fused_plan16_short.json"),
            32: ("dsa_fused_plan32_long.json", "dsa_fused_plan32_short.json"),
            64: (None, "dsa_fused_plan64_short.json")}   # (T = 64, post-only layers: contexts 256-512, 78.4 vs round
                                                         # robin 79.8 us; long contexts not searched)
KDA_PLAN = {16: os.path.join(os.path.dirname(HERE), "sglang_kda", "kda_plan16.json"),
            32: os.path.join(os.path.dirname(HERE), "sglang_kda", "kda_plan32.json"),
            64: os.path.join(os.path.dirname(HERE), "sglang_kda", "kda_plan64.json")}
MOE_PLAN = {(t, end): os.path.join(os.path.dirname(HERE), "sglang_moe", f"moe_plan{'' if t == 8 else t}_{end}.json")
            for t in TOKENS for end in ("mhc", "mhc_pre")}
MOE_END = {t: "mhc_pre" if t in (16, 32) else "mhc" for t in TOKENS}   # the MoE layers' end by T (moe_static_hook
                                                                         # END_PRE_TOKENS)


def _best_plans(name, env, args):
    env = dict(env)
    # the short-context DSA plans (searched at contexts 20-275): sgl_spec_bench.py at --prompt-tokens <= 1000
    pt = args[args.index("--prompt-tokens") + 1:] if "--prompt-tokens" in args else []
    pt = [int(x) for x in pt[:next((i for i, x in enumerate(pt) if x.startswith("--")), len(pt))]]
    short = bool(pt) and max(pt) <= 1000
    has = lambda kind, t: str(t) in env.get(f"{kind}_STATIC_TOKENS", ",".join(map(str, TOKENS))).split(",")
    for t in TOKENS:
        key = "" if t == 8 else str(t)
        dsa = DSA_PLAN.get(t, (None, None))[short]
        dsa = dsa and os.path.join(HERE, dsa)
        for kind, var, plan in (("DSA", "DSA_STATIC_FUSED_PLAN", dsa), ("KDA", "KDA_STATIC_PLAN", KDA_PLAN.get(t)),
                                ("MOE", "MOE_STATIC_PLAN", MOE_PLAN[t, MOE_END[t]])):
            if plan and env.get(f"{kind}_STATIC_LAYERS") and has(kind, t) and var + key not in env and os.path.exists(plan):
                env[var + key] = plan
    return name, env, args


def _foreign(gpus: str) -> list:
    """The other processes (pid, MiB) holding memory on these GPUs (nvidia-smi indices, PCI bus order), and the GPUs whose
    used memory is above 1000 MiB with none listed (a process of another container)."""
    q = lambda *a: subprocess.run(["nvidia-smi", *a, "--format=csv,noheader,nounits"], capture_output=True,
                                  text=True).stdout.strip().splitlines()
    want = {i.strip() for i in gpus.split(",")}
    bus = {b.strip().lower(): (i.strip(), int(m)) for i, b, m in
           (l.split(",") for l in q("--query-gpu=index,pci.bus_id,memory.used")) if i.strip() in want}
    apps = [(b.strip().lower(), p.strip(), m.strip()) for b, p, m in
            (l.split(",") for l in q("--query-compute-apps=gpu_bus_id,pid,used_memory") if l.count(",") == 2)]
    out = [f"GPU {bus[b][0]}: pid {p} {m} MiB" for b, p, m in apps if b in bus]
    listed = {b for b, _, _ in apps}
    out += [f"GPU {i}: {m} MiB used" for b, (i, m) in bus.items() if m > 1000 and b not in listed]
    return out


def _wait_free(gpus: str, name: str, limit_s: float = 3 * 3600) -> bool:
    """Before a run: wait (20 s checks, up to limit_s) until no other process holds memory on the chain's GPUs (other
    users' jobs land on them between two runs: SGLang then fails at load, "memory capacity is unbalanced" / OOM)."""
    import time
    t0, said = time.time(), False
    while True:
        busy = _foreign(gpus)
        if not busy:
            return True
        if not said:
            print(f"--- {name}: waiting for GPUs {gpus} ({'; '.join(busy)[:200]})", flush=True)
            said = True
        if time.time() - t0 > limit_s:
            return False
        time.sleep(20)


def _nccl_port(gpus):
    """A free port (bind test, IPv4 as SGLang's TCPStore) in 22000 + 1000 x the first visible GPU's index .. + 999: two
    chains on disjoint GPUs draw from disjoint ranges, and each run a new port (a port of the previous engine may still be
    held while its processes exit)."""
    head = (gpus or "0").split(",")[0].strip()
    low = 22000 + 1000 * (min(int(head), 7) if head.isdigit() else 0)   # (GPU UUIDs: 22000)
    for port in random.sample(range(low, low + 1000), 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port in {low}-{low + 999}")


LOAD_FAIL = ("memory capacity is unbalanced", "CUDA out of memory",   # a run that died at load (retried once)
             "leave no GPU memory for the KV cache")

if __name__ == "__main__":
    runs = {r[0]: r for r in map(lambda r: _best_plans(*r), RUNS)}
    unknown = [w for w in sys.argv[1:] if w not in runs]
    if unknown:   # before the first run, not after hours of the earlier ones
        sys.exit(f"unknown run names: {' '.join(unknown)}")
    if not os.path.isdir(MODEL):
        sys.exit(f"no model directory {MODEL} (GLM53F_MODEL)")
    seen = {}
    for want in sys.argv[1:] or list(runs):   # in the given order; a name given again runs again as <name>_2, ...
        _, extra, cmd = runs[want]
        seen[want] = seen.get(want, 0) + 1
        name = want if seen[want] == 1 else f"{want}_{seen[want]}"
        env = {k: v for k, v in os.environ.items() if not k.startswith(("DSA_STATIC", "KDA_STATIC", "MOE_STATIC", "STATIC_L2")) and
               k not in ("PYTHONPATH", "LD_LIBRARY_PATH", "MIRAGE_ROOT")}
        env.update(extra, DSA_STATIC_OUT=os.path.join(os.getcwd(), name, "static"),
                   KDA_STATIC_OUT=os.path.join(os.getcwd(), name, "static_kda"),
                   MOE_STATIC_OUT=os.path.join(os.getcwd(), name, "static_moe"))
        log = os.path.join(os.getcwd(), f"{name}.log")
        run = [sys.executable, os.path.join(HERE, cmd[0]), *cmd[1:]]
        if name.startswith("prof_"):
            nsys = [("--cuda-graph-trace=graph" if a == "--cuda-graph-trace=node" and want in GRAPH_TRACE else a) for a in NSYS]
            run = nsys + ["-o", os.path.join(os.getcwd(), name)] + run
        gpus = os.environ.get("CUDA_VISIBLE_DEVICES")   # (nvidia-smi indices, PCI bus order)
        for attempt in (1, 2):   # a run that died at load because of another user's job: once more after the GPUs clear
            if not os.environ.get("GLM53F_NCCL_PORT"):   # the run's scripts read it (nccl_port): a new free port per try
                env["GLM53F_NCCL_PORT"] = str(_nccl_port(gpus))
            limit = float(os.environ.get("SGL_FREE_WAIT_S", 3 * 3600))
            if gpus and os.environ.get("SGL_WAIT_FREE", "1") != "0" and not _wait_free(gpus, name, limit):
                print(f"=== {name}: GPUs {gpus} never cleared, skipped", flush=True)
                if "SGL_FREE_WAIT_S" in os.environ:
                    sys.exit(4)   # the caller starts the remaining runs again on 4 other idle GPUs
                code = None
                break
            with open(log, "w") as f:
                code = subprocess.run(run, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
                if name.startswith("prof_") and code == 0:
                    subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o",
                                    os.path.join(os.getcwd(), f"{name}.sqlite"), os.path.join(os.getcwd(), f"{name}.nsys-rep")],
                                   env=env, stdout=f, stderr=subprocess.STDOUT)
            text = open(log).read()
            if code == 0 or attempt == 2 or not any(k in text for k in LOAD_FAIL):
                break
            os.replace(log, log.replace(".log", ".try1.log"))
            print(f"--- {name}: died at load (another job on the GPUs), retrying", flush=True)
        if code is None:
            continue
        lines = open(log).read().splitlines()
        checks = [l for l in lines if "check layer" in l]
        print(f"=== {name}: exit {code} ({log}){f', {len(checks)} check lines' if checks else ''}", flush=True)
        for l in [l for l in lines if any(k in l for k in KEYS) and "replaces SGLang" not in l][:12]:
            print("   ", l[:300], flush=True)
