"""compare_bs8.py MODEL [--rounds 2] [--out bs8_runs] [--dry-run]: DeepSeek-V4.1-Flash, TP4, bs 8, 4000-token prompts:
SGLang against the static layers, one run of each in turn per round, on the 4 GPUs of CUDA_VISIBLE_DEVICES (default
0,1,2,3), with PR #819's method. Each run is dsv41_spec_bench.py --profile under nsys with the CUDA graph trace; its decode
step is the period of the forward graph (period.py). Prints each run's period, the speedup of each round, and the speedup
of the medians = SGLang / static - 1. The runs (.nsys-rep, .sqlite, .log) stay in --out.

Before each run it stops when a GPU holds another process or has less than 250000 MiB (the static layers need about
213 GiB a GPU: a B300). It also stops when a run fails, or when a static run's log does not show the static layers.
Env: MIRAGE_Z3_LIB (the static runs: mirage's libz3 directory, when mirage was built in another venv than SGLang's).
"""
import argparse
import os
import re
import shlex
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)                     # demo/deepseek_v4_1_sglang
ROOT = os.path.dirname(os.path.dirname(D))    # this checkout
ARGS = ["--bs", "8", "--spec", "off", "--prompt-tokens", "4000", "--max-total-tokens", "81920", "--reps", "1",
        "--profile", "--profile-steps", "64", "--profile-tokens", "256"]
NSYS = ["nsys", "profile", "--force-overwrite", "true", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
        "--trace=cuda,nvtx", "--cuda-graph-trace=graph", "--trace-fork-before-exec=true"]
# the static layers: every attention and MoE layer, mHC and the all-reduce in the kernels, and the switches of
# README_bs8.md (the KV layout v41_new is theirs: SGLang alone keeps its default)
STATIC = {"DSV41_STATIC_ROOT": ROOT, "DSV41_STATIC_ATTN": "all", "DSV41_STATIC_MOE": "all", "DSV41_STATIC_MAX_POS": "4607",
          "SGLANG_DSV4_KV_LAYOUT": "v41_new", "DSV41_STATIC_PDL": "1", "DSV41_STATIC_MHC": "1", "SA_DRY": "0",
          "HC_DRY_MOE": "1", "HC_DRY_ATTN": "0",
          "DSV41_STATIC_FLAGS": "-DHCP_LOOP -DHCP_SSQ=2 -DHCP_VPOLL -DSTATIC_NO_START_BARRIER -DSTATIC_RESET_PARITY "
                                "-DI1_BF16 -DI1B_FQ -DMOE_ROWS_POLL",
          "DSV41_STATIC_IS_TCGEN05": "1", "DSV41_S2_ARGMAX": "1", "DSV41_X_FLAG": "1", "DSV41_X_FLAG_ATTN": "1",
          "DSV41_X_FLAG_ATTN_BIND": "0", "DSV41_I1C": "2", "DSV41_S2_DEFER": "1"}
STATIC_RAN = "first static decode step (L0): in a CUDA graph capture"   # (dsv41_static_hook.py's log)
PERIOD = re.compile(r"^forward graph \S+ on GPU \d+: \d+ periods; period median ([\d.]+) us")
NEED_MIB = 250000


def smi(gpus: str, *query) -> list:
    r = subprocess.run(["nvidia-smi", "-i", gpus, *query, "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"nvidia-smi -i {gpus} {' '.join(query)}: exit {r.returncode} {(r.stdout + r.stderr).strip()[:300]}")
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def check_gpus(gpus: str, wait_s: int = 120):
    small = [g for g in smi(gpus, "--query-gpu=index,memory.total") if int(g.split(",")[1]) < NEED_MIB]
    if small:
        sys.exit(f"GPUs with less than {NEED_MIB} MiB (index, MiB): {small}; the static layers need a B300")
    for _ in range(wait_s // 5):   # (the last run's processes can take some seconds to exit)
        busy = smi(gpus, "--query-compute-apps=pid,used_memory")
        if not busy:
            return
        time.sleep(5)
    sys.exit(f"GPUs {gpus} still hold other processes after {wait_s} s (pid, MiB): {busy}")


def _prepend(path: str, var: str) -> str:
    return os.pathsep.join(p for p in (path, os.environ.get(var)) if p)


def period_of(db: str) -> float:
    out = subprocess.run([sys.executable, os.path.join(HERE, "period.py"), db], capture_output=True, text=True,
                         check=True).stdout
    m = next((PERIOD.match(line) for line in out.splitlines() if PERIOD.match(line)), None)
    if m is None:
        sys.exit(f"no forward graph in {db}:\n{out}")
    return float(m.group(1))


def run(name: str, static: bool, a, gpus: str) -> float:
    env = {"DSV41_MODEL": a.model, "DSV41_TP": "4", "DSV41_PRINT_ALL": "1"}
    if static:   # the hook enters SGLang's processes through boot/sitecustomize.py
        env.update(STATIC, PYTHONPATH=_prepend(os.path.join(D, "boot"), "PYTHONPATH"))
        if os.environ.get("MIRAGE_Z3_LIB"):   # mirage's libz3, when it was built in another venv than SGLang's (as #819)
            env["LD_LIBRARY_PATH"] = _prepend(os.environ["MIRAGE_Z3_LIB"], "LD_LIBRARY_PATH")
    out = os.path.join(a.out, name)
    cmd = [*NSYS, "-o", out, sys.executable, os.path.join(HERE, "dsv41_spec_bench.py"), *ARGS]
    if a.dry_run:
        print(" ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()), shlex.join(cmd), f"> {out}.log")
        return float("nan")
    check_gpus(gpus)
    t0 = time.time()
    with open(out + ".log", "w") as f:
        code = subprocess.run(cmd, env={**os.environ, **env}, stdout=f, stderr=subprocess.STDOUT).returncode
    if code:
        sys.exit(f"{name}: exit {code}, see {out}.log")
    if static and STATIC_RAN not in open(out + ".log", errors="replace").read():
        sys.exit(f"{name}: the static layers did not run ('{STATIC_RAN}' is not in {out}.log)")
    subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", out + ".sqlite",
                    out + ".nsys-rep"], check=True, stdout=subprocess.DEVNULL)
    p = period_of(out + ".sqlite")
    print(f"{name}: period median {p:.1f} us a step ({(time.time() - t0) / 60:.0f} min)", flush=True)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="the official 40-layer DeepSeek-V4.1-Flash checkout")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default="bs8_runs")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args()
    gpus = ",".join(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")[:4])
    os.makedirs(a.out, exist_ok=True)
    sgl, sta = [], []
    for r in range(1, a.rounds + 1):
        sgl.append(run(f"sglang_r{r}", False, a, gpus))
        sta.append(run(f"static_r{r}", True, a, gpus))
        if not a.dry_run:
            print(f"round {r}: speedup {100 * (sgl[-1] / sta[-1] - 1):+.2f} %", flush=True)
    if not a.dry_run:
        s, t = statistics.median(sgl), statistics.median(sta)
        print(f"median over {a.rounds} rounds: SGLang {s:.1f} us, static {t:.1f} us a step; "
              f"speedup = SGLang / static - 1 = {100 * (s / t - 1):+.2f} %")


if __name__ == "__main__":
    main()
