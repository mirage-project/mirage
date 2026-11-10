"""The per-rank server library of a generated layer (one tensor-parallel rank per process, as SGLang runs it) from the
compiler's layer.cu (a test's --codegen output): the kernel and its task tables kept, the harness's host part (all GPUs from one
process) replaced by include/mirage/static_megakernel_hip/rank_host.cuh, the slot args written as its tables.

  python amd/tools/gen_rank_lib.py --layer-cu kda_layer/layer.cu --kind kda --out rank_kda [--flag -DSTATIC_...] [--no-build]

--kind names the graph tensors a layer binds once (weights) and per run (KINDS below); every tensor the slot args name must be
one of them or a node output. Writes <out>/layer_rank.cu, <out>/rank_tables.h and (hipcc, gfx950) <out>/lib.so.
The build flags: the layer's (--flag), as its harness build had them.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MIRAGE = os.path.dirname(os.path.dirname(HERE))
INCLUDE = os.path.join(MIRAGE, "include")
HOST_MARK = "// ---- host: all GPUs of this process ----"

# per layer kind: (weights, run tensors, hidden width)
KINDS = {
    "kda": (["wproj_k", "conv_w", "wfb", "prm", "oproj_k", "q", "gain", "ow"],
            ["x", "bank", "prefix", "new_prefix", "hidden", "info", "slot", "icw_row", "conv_pool", "state_pool", "icw",
             "ring_k", "ring_v", "ring_g", "ring_beta"], 7168),
    "moe": (["router_weight", "score_correction_bias", "latent_down_weight", "latent_up_weight", "routed_expert_norm_weight",
             "shared_gate_up_weight", "shared_down_weight", "w13_blocks", "w13_scales", "w2_blocks", "w2_scales"],
            ["moe_in", "moe_prefix", "moe_out"], 7168),
    "mla": (["wa_k", "wq_norm", "wkv_norm", "wqb_k", "wg_k", "wkc_k", "wvc_k", "wo_k", "q", "gain", "ow"],
            ["x", "bank", "prefix", "new_prefix", "hidden", "info", "cache", "req_to_token", "seq_lens", "req_idx", "out_loc"],
            7168),
}
# plain decode (amd/tests/test_kda_decode.py: tasks/short_conv_decode.cuh, delta_rule_decode.cuh): the pools read and written at
# per-request slots; the decode MLA and MoE libraries (test_mla_decode.py, test_layer_one_gpu.py with MPK_TOKENS) bind what "mla"
# and "moe" bind
KINDS["kdad"] = (KINDS["kda"][0], ["x", "bank", "prefix", "new_prefix", "hidden", "info", "slots", "conv_pool", "state_pool"], 7168)
# the MoE ending with the next layer's aggregation 1 (amd/tests/test_layer_one_gpu.py --agg NB,WRITE,2: tasks/moe_agg1.cuh)
KINDS["moeagg"] = (KINDS["moe"][0] + ["agg_q", "agg_gain", "agg_ow"],
                   KINDS["moe"][1] + ["agg_bank", "agg_info", "agg_out"], 7168)
# an attention part and its MoE in one kernel (amd/tests/test_kda_moe_layer.py): the MoE reads the part's hidden and new prefix
for _a in ("kda", "mla"):
    KINDS[_a + "moe"] = (KINDS[_a][0] + KINDS["moe"][0], KINDS[_a][1] + ["moe_out"], 7168)
    # ... and the MoE ending with the next layer's aggregation 1 (test_kda_moe_layer.py --agg NB,WRITE,2)
    KINDS[_a + "moeagg"] = (KINDS[_a][0] + KINDS["moeagg"][0],
                            KINDS[_a][1] + ["moe_out", "agg_bank", "agg_info", "agg_out"], 7168)
# a MoE layer with the next layer's aggregation 1, then the NEXT layer's attention part in the same kernel (test_kda_moe_layer.py
# --order moe): the KDA / MLA part reads aggregation 1's output (x) and the MoE's y (prefix) inside the kernel
for _a in ("kda", "mla", "kdad"):   # (kdad: the plain-decode KDA part, test_kda_moe_layer.py --decode --order moe)
    KINDS["moeagg" + _a] = (KINDS["moeagg"][0] + KINDS[_a][0],
                            KINDS["moeagg"][1] + ["agg_prefix", "agg_bare"] + (["agg_mix"] if os.environ.get("MPK_AGG_MIX") == "1" else [])
                            + [n for n in KINDS[_a][1] if n not in ("x", "prefix")],
                            7168)



# GLM-5.3-Flash (hidden 4096, mHC with 4 streams; amd/tests/test_glm_*_layer.py --codegen): a sublayer kernel reads the
# streams R_in and the sublayer's coefficients (h_post_in, h_res_in), writes R_out and the next sublayer's input (x_ffn /
# x_attn) and coefficients (h_post_out, h_res_out); the scratch tensors stay inside the kernel (the harnesses' --serving
# graphs: node buffers the kernel resets every launch)
_MHC_IO = ["R_in", "h_post_in", "h_res_in", "R_out", "h_post_out", "h_res_out"]
KINDS["gkda"] = (["wproj", "wfb", "wgb", "prm", "conv_w", "w_o", "hc_ffn_fn", "hc_ffn_small", "ffn_norm"],
                 ["x_attn", "state_pool", "conv_pool", "slots", "x_ffn"] + _MHC_IO, 4096)
KINDS["gdsa"] = (["w_qkva", "q_norm", "kv_norm", "w_kgw", "k_norm_w", "k_norm_b", "ape", "w_iqb", "w_qb", "w_kc", "w_vc", "w_o",
                  "hc_ffn_fn", "hc_ffn_small", "ffn_norm"],
                 ["x_attn", "cache", "out_loc", "seq_lens", "req_slots", "pooled_bt", "tail_k", "tail_score", "pcache", "r2t",
                  "req_idx", "x_ffn"] + _MHC_IO, 4096)
KINDS["gmoe"] = (["router", "bias", "w13", "w13_sf", "w2", "w2_sf", "hc_attn_fn", "hc_attn_small", "attn_norm"],
                 ["x_ffn", "x_attn"] + _MHC_IO, 4096)
KINDS["gdense"] = (["w_gu", "w_dn", "hc_attn_fn", "hc_attn_small", "attn_norm"], ["x_ffn", "x_attn"] + _MHC_IO, 4096)
# one kernel per layer (amd/tests/test_glm_pair_layer.py --codegen --serving): KDA + MoE; x_ffn stays inside (the mid streams
# R_mid, h_*_mid are graph tensors: run tensors the hook gives),
# the next layer's input goes to x_attn_next (the hook alternates two x_attn buffers)
KINDS["gdsamoe"] = (KINDS["gdsa"][0] + KINDS["gmoe"][0],
                    [k for k in KINDS["gdsa"][1] if k != "x_ffn"] + ["x_attn_next", "R_mid", "h_post_mid", "h_res_mid"], 4096)
KINDS["gkdadense"] = (KINDS["gkda"][0] + KINDS["gdense"][0],
                      ["x_attn", "state_pool", "conv_pool", "slots", "x_attn_next", "R_mid", "h_post_mid", "h_res_mid"] + _MHC_IO, 4096)
# N consecutive KDA + MoE layers as one kernel (amd/tests/test_glm_chain_layer.py --layers N --codegen --serving): per layer j
# its tensors suffixed _j; the streams between the sublayers are run tensors (R_a_j .. after the KDA part, R_b_j .. after the MoE
# part of every layer but the last)
for _n in (2, 3):
    KINDS[f"gkdamoe{_n}"] = (
        [f"{w}_{j}" for j in range(_n) for w in KINDS["gkda"][0] + KINDS["gmoe"][0]],
        ["x_attn", "slots", "x_attn_next"] + [f"{t}_{j}" for j in range(_n) for t in ("state_pool", "conv_pool")]
        + [f"{t}_{j}" for j in range(_n) for t in ("R_a", "h_post_a", "h_res_a")]
        + [f"{t}_{j}" for j in range(_n - 1) for t in ("R_b", "h_post_b", "h_res_b")] + _MHC_IO, 4096)
# the model's last layer as one kernel without its final hc_pre (test_glm_pair_layer.py --last): no next-layer x / coefficients
KINDS["gkdamoelast"] = ([w for w in KINDS["gkda"][0] + KINDS["gmoe"][0] if w not in ("hc_attn_small", "attn_norm")],
                        ["x_attn", "state_pool", "conv_pool", "slots", "R_mid", "h_post_mid", "h_res_mid", "R_in", "h_post_in", "h_res_in",
                         "R_out", "hfin"], 4096)
KINDS["gkdamoe"] = (KINDS["gkda"][0] + KINDS["gmoe"][0],
                    ["x_attn", "state_pool", "conv_pool", "slots", "x_attn_next", "R_mid", "h_post_mid", "h_res_mid"] + _MHC_IO,
                    4096)


def slot_args(code: str) -> dict:
    """the slot args of the harness's static_host_init call"""
    m = re.search(r"static_host_init\(g_ctx, std::map<std::string, std::string>\{(.*)\}\);", code)
    if not m:
        raise ValueError("layer.cu: no static_host_init call with the slot args")
    return dict(re.findall(r'\{"([a-z]+\.\d+)", "([^"]*)"\}', m.group(1)))


def tables(slots: dict, weights: list, run: list, hidden: int) -> str:
    named, outs = set(), set()
    for k, v in slots.items():
        f = v.split()
        if k.startswith("buf.") and f[0] in ("out", "scratch", "exchange"):
            outs.add(f[1])
        if k.startswith("buf.") and f[0] == "tensor":
            named.add(f[1])
        if k.startswith("map.") and f[1].startswith("t:"):
            named.add(f[1][2:])
    missing = named - set(weights) - set(run) - outs
    if missing:
        raise ValueError(f"tables: graph tensors neither weights, run tensors nor node outputs: {sorted(missing)}")
    unused = (set(weights) | set(run)) - named - outs
    if unused:
        raise ValueError(f"tables: listed tensors no slot uses: {sorted(unused)}")
    q = lambda xs: "{" + ", ".join(f'"{x}"' for x in xs) + "}"
    return "\n".join(["// generated by amd/tools/gen_rank_lib.py for rank_host.cuh", f"#define MPK_H {hidden}",
                      f"#define MPK_NW {len(weights)}", f"#define MPK_NR {len(run)}",
                      "static char const *const mpk_slot_args[][2] = {"] +
                     [f'  {{"{k}", "{v}"}},' for k, v in sorted(slots.items())] +
                     ["};", f"static char const *const mpk_weight_names[] = {q(weights)};",
                      f"static char const *const mpk_run_names[] = {q(run)};", ""])


def generate(layer_cu: str, kind: str, out: str) -> str:
    code = open(layer_cu).read()
    if code.count(HOST_MARK) != 1:
        raise ValueError(f"{layer_cu}: the host part's mark not found once")
    kernel, host = code.split(HOST_MARK)
    if "#define STATIC_TIMING_BUILD" in kernel:
        raise ValueError(f"{layer_cu}: a timing build (--profile); the server library takes the plain one")
    n = int(re.search(r"#define STATIC_GPUS (\d+)", kernel).group(1))
    nl = int(re.search(r"#define STATIC_SMS (\d+)", kernel).group(1))
    for g in range(n):
        if f"static_tasks_gpu{g}[]" not in kernel or f"static_list_begin_gpu{g}[]" not in kernel:
            raise ValueError(f"{layer_cu}: no task tables of GPU {g}")
    weights, run, hidden = KINDS[kind]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "rank_tables.h"), "w") as f:
        f.write(tables(slot_args(host), weights, run, hidden))
    tail = [f"#define MPK_NUM_GPUS {n}", f"#define MPK_NLISTS {nl}",
            f"static StaticTask const *const mpk_tasks_of[{n}] = {{{', '.join(f'static_tasks_gpu{g}' for g in range(n))}}};",
            f"static int const *const mpk_begin_of[{n}] = {{{', '.join(f'static_list_begin_gpu{g}' for g in range(n))}}};",
            '#define MPK_TABLES "rank_tables.h"', '#include "mirage/static_megakernel_hip/rank_host.cuh"', ""]
    path = os.path.join(out, "layer_rank.cu")
    with open(path, "w") as f:
        f.write(kernel + "\n".join(["// ---- host: this process's rank (amd/tools/gen_rank_lib.py) ----"] + tail))
    return path


def build(cu: str, flags: list, kind: str, arch: str = "gfx950") -> str:
    """hipcc; the kernel named mpk_<kind>_kernel (in profiler traces: which library's launch)"""
    so = os.path.join(os.path.dirname(cu), "lib.so")
    cmd = [shutil.which("hipcc") or "hipcc", "-O3", "-std=c++17", f"--offload-arch={arch}", f"-I{INCLUDE}",
           f"-I{os.path.dirname(cu)}", "-shared", "-fPIC", cu, "-o", so, f"-Dlayer_kernel=mpk_{kind}_kernel"] + list(flags)
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-6000:])
        raise SystemExit(f"hipcc failed ({r.returncode})")
    print(f"{so}: {r.stderr.count('warning:')} warnings", flush=True)
    if os.environ.get("MPK_RANK_LIB_ASM"):   # (the device code's assembly beside it: <out>/lib.s, to compare builds)
        asm = os.path.join(os.path.dirname(cu), "lib.s")
        c2 = [x for x in cmd if x not in ("-shared", "-fPIC")]
        c2[c2.index("-o") + 1] = asm
        r2 = subprocess.run(c2 + ["--cuda-device-only", "-S"], capture_output=True, text=True)
        if r2.returncode != 0:
            sys.stderr.write(r2.stderr[-3000:])
            raise SystemExit(f"hipcc -S failed ({r2.returncode})")
        print(f"{asm}: written", flush=True)
    return so


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer-cu", required=True)
    p.add_argument("--kind", required=True, choices=sorted(KINDS))
    p.add_argument("--out", required=True)
    p.add_argument("--flag", action="append", default=[], help="an extra hipcc flag (e.g. -DSTATIC_GEMM_LDSW=3)")
    p.add_argument("--no-build", action="store_true")
    a = p.parse_args()
    cu = generate(a.layer_cu, a.kind, a.out)
    print("wrote", cu, flush=True)
    if not a.no_build:
        build(cu, a.flag, a.kind)


if __name__ == "__main__":
    main()
