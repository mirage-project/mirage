#!/usr/bin/env python3
"""M3-I8 CPU-side gate. No GPU, no B200, no compiled kernel required.

Four groups of checks, all falsifiable:

  S1  SOURCE     the change is present, defaults OFF, and gates only the two
                 marking writes -- not the row read, not the weight write
  S2  CONSUMERS  `moe_routing_indices` / `moe_mask` reach exactly the two
                 grouped-GEMM call sites, so nothing else can see the change
  S3  GRAPH      the compiled task graph really does dispatch the MoE grouped
                 GEMMs as 80 sites x 256 tasks on a (128, 2, 1) grid with
                 y-inner expert_offset order, which is what makes the live
                 tasks a contiguous prefix and the worker mapping computable
  S4  MODEL      the cost model reproduces M3-I1's ten measured wall spans, and
                 the pre/post activated-group and wave table it implies

S3 is skipped with a loud note when no compiled graph is reachable (they live
on the B200 under ~/mpk-qwen35/m3i1/kernel_bs*_prof/); S1/S2/S4 always run.

    python3 static_checks.py
    python3 static_checks.py --graph /path/to/task_graph_rank0.json [...]
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *([os.pardir] * 5)))
FAILS = []
BS = [1, 2, 4, 8, 16]
# M3-I1 measured activated groups per layer (nlong / (40 * moe_n_splits))
MEASURED = {1: 56.4, 2: 59.4, 4: 60.2, 8: 70.1, 16: 86.7}


def check(name, cond, detail=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)
    return cond


def read(p):
    with open(os.path.join(REPO, p)) as f:
        return f.read()


def s1_source():
    print("\nS1 source")
    k = read("include/mirage/persistent_kernel/tasks/blackwell/topk_softmax_sm100.cuh")
    check("kernel takes num_active_rows, default -1 (no gating)",
          "int const num_active_rows = -1" in k)
    check("out-of-range live count falls back to num_rows",
          re.search(r"num_active_rows > 0 && num_active_rows < num_rows", k) is not None)
    check("row_is_active gates on the live-row bound",
          "(thread_row < live_rows)" in k)
    # The three clauses that must stay UNGATED.
    body = k[k.index("if (thread_row < num_rows) {"):]
    zero = body.index("reset input buffer to 0 for split-k gate linear")
    guard = body.index("if (should_process_row && mpk_routing_indices")
    check("input-buffer zeroing stays outside the gate", zero < guard,
          "gating it would let padding-row logits accumulate across iterations")
    wwrite = body.index("output[out_idx] = max_val;")
    check("top-k weight write stays outside the gate", wwrite < guard)
    live_lines = [l for l in body.splitlines()
                  if "should_process_row" in l and not l.strip().startswith("//")]
    check("only the marking writes are guarded", len(live_lines) == 2,
          "; ".join(l.strip()[:48] for l in live_lines))

    r = read("src/kernel/task_register.cc")
    fn = r[r.index("int TaskRegister::register_moe_topk_softmax_sm100_task"):]
    fn = fn[:fn.index("int TaskRegister::register_moe_topk_sigmoid")]
    check("registration accepts params[1] and defaults it off",
          "gate_padding_rows = params.size() > 1 && params[1] == 1" in fn)
    check("gated path emits the live-token scalar",
          "runtime_config.qo_indptr_buffer[" in fn
          and "MPK_MAX_NUM_BATCHED_REQUESTS]);" in fn)
    emit = fn[fn.index("if (gate_padding_rows) {"):]
    off = emit[emit.index("} else {"):]
    check("ungated path emits the pre-M3-I8 argument list unchanged",
          off.count("code.e") == 1
          and 'code.e("    $);", round_weights ? "true" : "false");' in off)

    p = read("python/mirage/mpk/persistent_kernel.py")
    lay = p[p.index("def moe_topk_softmax_routing_layer"):]
    lay = lay[:lay.index("def moe_topk_sigmoid_routing_layer")]
    check("layer API param defaults off", "gate_padding_rows: bool = False" in lay)
    check("params tail unchanged when the gate is off",
          "params = [1] if round_weights_to_input_dtype else []" in lay)

    b = read("python/mirage/mpk/models/qwen3_5/builder.py")
    check("builder exposes MOE_GATE_PADDING_ROWS",
          re.search(r"^MOE_GATE_PADDING_ROWS = (True|False)$", b, re.M) is not None)
    check("router call site passes it",
          "gate_padding_rows=MOE_GATE_PADDING_ROWS" in b)
    m = re.search(r"^MOE_GATE_PADDING_ROWS = (True|False)$", b, re.M)
    print(f"        tree is currently built with MOE_GATE_PADDING_ROWS = "
          f"{m.group(1) if m else '?'}")


def s2_consumers():
    print("\nS2 consumers of routing / mask")
    b = read("python/mirage/mpk/models/qwen3_5/builder.py")
    moe = b[b.index("def _build_moe"):]
    moe = moe[:moe.index("def encode")]
    n_routing = moe.count("moe_routing_indices=routing")
    n_mask = moe.count("moe_mask=mask")
    check("routing/mask reach exactly the two grouped GEMMs",
          n_routing == 2 and n_mask == 2, f"routing={n_routing} mask={n_mask}")
    check("both are moe_fp8_blockscale_layer call sites",
          moe.count("pk.moe_fp8_blockscale_layer(") == 2)
    # the combine reads topk_w, which the change does NOT touch
    check("the combine still reads the (ungated) top-k weights",
          "weight=topk_w" in moe)


def s3_graph(paths):
    print("\nS3 compiled task graph")
    if not paths:
        print("  [skip] no compiled task graph given; pass --graph "
              "<task_graph_rank0.json> (they live on the B200 under "
              "~/mpk-qwen35/m3i1/kernel_bs*_prof/)")
        return None
    sys.path.insert(0, HERE)
    import taskgraph_moe as tg
    out = {}
    for p in paths:
        d = json.load(open(p))
        tasks = d["all_tasks"]
        for i, t in enumerate(tasks):
            t["_idx"] = i
        launch = [e for e in d["all_events"] if e["event_type"] == 903]
        check(f"{os.path.basename(p)}: one dependent launch event covers the graph",
              len(launch) == 1 and launch[0]["last_task_id"] >= len(tasks) - 1)
        first = launch[0]["first_task_id"]
        by = {}
        for t in tasks:
            by.setdefault(t["task_type"], []).append(t)
        for tt in (241, 242):
            sites = tg.sites_of(by[tt])
            descs = [tg.describe_site(s) for s in sites]
            check(f"  type {tt}: 40 sites of 256 tasks", len(sites) == 40
                  and all(x["n_tasks"] == 256 for x in descs))
            check(f"  type {tt}: grid (128, 2, 1)",
                  all(x["grid_x"] == 128 and x["splits"] == 2 for x in descs))
            check(f"  type {tt}: expert_offset is y-inner (live tasks are a "
                  f"contiguous prefix)", all(x["interleaved"] for x in descs))
            out[(p, tt)] = (descs[0], first)
    return out


def s4_model(graph_info):
    print("\nS4 cost model")
    rc = subprocess.call([sys.executable, os.path.join(HERE, "model_moe_wall.py"),
                          "--check"], stdout=subprocess.DEVNULL)
    check("model_moe_wall.py --check (C1 flat per-task time, C2 tile law, "
          "C3 wave model within 12%)", rc == 0,
          "run it directly for the tables")

    sys.path.insert(0, HERE)
    import model_moe_wall as mm
    meas = mm.load_i1()
    _, _, est_a, est_b = mm.activated_after(meas)
    print("\n  pre/post activated expert groups per layer, and the worker "
          "depth they imply\n")
    print(f"  {'bs':>3} {'live':>5} {'cap':>5} {'now':>7} {'gated':>7} "
          f"{'tasks now':>10} {'tasks gated':>12} {'waves now':>10} "
          f"{'waves gated':>12}")
    for bs in BS:
        a_now = meas[bs][241]["activated"]
        a_new = 0.5 * (est_a[bs] + est_b[bs])
        cap = min(256, 8 * bs)
        t_now, t_new = a_now * 2, a_new * 2
        w_now, w_new = mm.waves(t_now), mm.waves(t_new)
        print(f"  {bs:>3} {bs:>5} {cap:>5} {a_now:>7.1f} {a_new:>7.1f} "
              f"{t_now:>10.1f} {t_new:>12.1f} {w_now:>10d} {w_new:>12d}")
        check(f"  bs{bs}: gated estimate respects the hard cap",
              a_new <= cap + 1e-6)
        check(f"  bs{bs}: measured pre-change count matches M3-I1's table",
              abs(a_now - MEASURED[bs]) < 0.15, f"{a_now:.1f} vs {MEASURED[bs]}")
    # the one structural claim that does not depend on the collision estimate
    check("bs8 drops to exactly one wave for ANY collision rate "
          "(8 tokens x top-8 <= 64 groups x 2 splits = 128 tasks)",
          mm.waves(min(256, 8 * 8) * 2) == 1)
    check("bs16 cannot improve (every row is already live)",
          abs(0.5 * (est_a[16] + est_b[16]) - meas[16][241]["activated"]) < 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", action="append", default=None)
    a = ap.parse_args()
    print("M3-I8 static checks (CPU only)")
    s1_source()
    s2_consumers()
    gi = s3_graph(a.graph)
    s4_model(gi)
    print("\n" + ("FAILED: " + ", ".join(FAILS) if FAILS else "ALL STATIC CHECKS PASS"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
