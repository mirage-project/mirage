"""SGLang's Kimi K3 MoE layer (sglang.srt.models.kimi_k3.KimiK3MoE, the ROCm path) on the weights and inputs of
amd/tests/test_layer_one_gpu.py, timed the same way, for an A/B against our layer kernel.

What runs: SGLang's own module, built and loaded as its server does for `--tp 8 --dtype bfloat16` with the MI35x recipe's
environment (SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1,
test/registered/amd/perf/mi35x/test_kimi_k3_perf_mi35x.py): ServerArgs + runtime_context.publish, init_parallel_runtime (TP8
groups, AITER custom all-reduce), ModelConfig of the K3 config.json, the checkpoint's quantization config
(model_loader._get_quantization_config), KimiK3MoE(config, quant_config, prefix "...layers.1.mlp"), the weights copied into
its parameters in the checkpoint layout, then every module's quant_method.process_weights_after_loading (AITER's MXFP4
shuffle) and _merge_front_weights, as the loader does. The forward is KimiK3MoE.forward(x, prefix_sum=...), captured in a HIP
graph inside SGLang's graph_capture() (decode runs as a captured graph).

The weights: random_weights() of the harness (same seeds: GPU g = seed + g, the common tensors GPU 0's), converted from the
gfx950 tile layout back to the checkpoint layout (tiles_to_values / scales_to_matrix, gate rows then up rows, two e2m1 per
byte with the lower K in the low nibble); the shared gate_up rows from our blocks of 128 (64 gate, 64 up) to [gate; up].
Checks: SGLang's routing vs the reference top-16 of sigmoid(x Wg^T) + bias, and its y vs a float64 torch reference of the
layer (the harness's arithmetic: MXFP8 of z and h, SiTU(4, 25), RMSNorm eps 1e-5, bf16 roundings of the three terms).

Timing (as test_layer_one_gpu.py --flush-mb 1024 --e2e-reps N): per launch, 1 GB read on every GPU (L2 and the 256 MB
Infinity Cache empty), the ranks lined up (CPU barrier, then one small all-reduce on the stream), a start event, the graph
replay, an end event; e2e = the slowest rank's events; median over N launches.

torchrun --nproc_per_node 8 amd/bench/sglang_moe_layer.py --model-dir DIR [--reps 40] [--flush-mb 1024] [--seed 1]
DIR: a directory with the K3 config.json (no weights are read).
python amd/bench/sglang_moe_layer.py --cpu-check --model-dir DIR   (no GPU: server args, model config, quant config only)
"""
import argparse
import os
import statistics
import sys
import time
from types import SimpleNamespace

for k, v in (("SGLANG_USE_AITER", "1"), ("SGLANG_AITER_K3_OPT", "1"), ("AITER_FLYDSL_FORCE", "1"), ("AITER_SITUV2_A8W4", "1")):
    os.environ.setdefault(k, v)

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "amd", "tests"))
from test_layer_one_gpu import (bf16, dequant, mxfp8, random_weights, relrms, scales_to_matrix, situ,  # noqa: E402
                                tiles_to_values, w13_rows)

# the layer's sizes (demo/kimi_k3_moe/demo.py) and which tensors are each GPU's own shard (demo.WEIGHT_FILES)
SHARDED = {"shared_gate_up_weight", "shared_down_weight", "w13_blocks", "w13_scales", "w2_blocks", "w2_scales"}
D = SimpleNamespace(T=int(os.environ.get("MPK_TOKENS", "8")),   # (MPK_TOKENS: the batch size, as demo.py)
                    H=7168, L=3584, E=896, TOPK=16, IR_LOCAL=384, SHR_LOCAL=768,
                    WEIGHT_FILES=[(n, None, None, None, n in SHARDED) for n in (
                        "router_weight", "score_correction_bias", "latent_down_weight", "latent_up_weight",
                        "routed_expert_norm_weight", "moe_in", "moe_prefix", "shared_gate_up_weight", "shared_down_weight",
                        "w13_blocks", "w13_scales", "w2_blocks", "w2_scales")])


# ---- the gfx950 tile layout -> the checkpoint layout, all experts at once ----
def codes_batched(blocks: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """[E][N/16][K/128][64][16] tile bytes -> e2m1 codes [E][N][K] (tiles_to_values per expert)."""
    e = blocks.shape[0]
    t = blocks.reshape(e, n // 16, k // 128, 64, 16)
    v = torch.stack((t & 0xF, t >> 4), dim=-1).reshape(e, n // 16, k // 128, 4, 16, 32)
    return v.permute(0, 1, 4, 2, 3, 5).reshape(e, n, k)


def scales_batched(sc: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Scale bytes in the gfx950 layout -> [E][N][K/32] (scales_to_matrix per expert)."""
    e, ks = sc.shape[0], k // 128
    if ks % 4 == 0:
        return sc.reshape(e, n // 16, ks // 4, 4, 16, 4).permute(0, 1, 4, 2, 5, 3).reshape(e, n, k // 32)
    return sc.reshape(e, n // 64, 4, 16, 4, ks).permute(0, 1, 4, 3, 5, 2).reshape(e, n, k // 32)


def pack(codes: torch.Tensor) -> torch.Tensor:
    """e2m1 codes [..][K] -> bytes [..][K/2], the lower K in the low nibble."""
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def checkpoint_banks(w: dict, chunk: int = 128) -> dict:
    """This GPU's expert banks in the checkpoint layout: w13 [E][2 IR][L/2] (gate rows, then up rows) and its scales
    [E][2 IR][L/32]; w2 [E][L][IR/2], scales [E][L][IR/32]."""
    E, L, IR = D.E, D.L, D.IR_LOCAL
    gate_r, up_r = w13_rows(IR, w["w13_blocks"].device)
    order = torch.cat((gate_r, up_r))
    out = {"w13": [], "w13_s": [], "w2": [], "w2_s": []}
    for e0 in range(0, E, chunk):
        sl = slice(e0, min(E, e0 + chunk))
        out["w13"].append(pack(codes_batched(w["w13_blocks"][sl], 2 * IR, L)[:, order]))
        out["w13_s"].append(scales_batched(w["w13_scales"][sl], 2 * IR, L)[:, order].contiguous())
        out["w2"].append(pack(codes_batched(w["w2_blocks"][sl], L, IR)))
        out["w2_s"].append(scales_batched(w["w2_scales"][sl], L, IR).contiguous())
    out = {k: torch.cat(v) for k, v in out.items()}
    for e in (0, 1, E // 2, E - 1):   # the batched conversion == the harness's per-expert one
        assert torch.equal(out["w13"][e], pack(tiles_to_values(w["w13_blocks"][e], 2 * IR, L)[order])), e
        assert torch.equal(out["w13_s"][e], scales_to_matrix(w["w13_scales"][e], 2 * IR, L)[order]), e
        assert torch.equal(out["w2"][e], pack(tiles_to_values(w["w2_blocks"][e], L, IR))), e
        assert torch.equal(out["w2_s"][e], scales_to_matrix(w["w2_scales"][e], L, IR)), e
    return out


def shared_gate_up_rows(shr: int, device) -> torch.Tensor:
    """Our shared gate_up rows (blocks of 128: 64 gate features, then their 64 up features) in SGLang's order [gate; up]."""
    f = torch.arange(shr, device=device)
    gate = 128 * (f // 64) + f % 64
    return torch.cat((gate, gate + 64))


# ---- SGLang's runtime, as its scheduler process sets it up ----
def sglang_server_args(model_dir: str, port: int):
    """The MI35x recipe's flags (test_kimi_k3_perf_mi35x.py) through SGLang's own CLI parser; no tokenizer, no weights read."""
    from sglang.srt.server_args import prepare_server_args
    return prepare_server_args(["--model-path", model_dir, "--tp", "8", "--dtype", "bfloat16", "--trust-remote-code",
                                "--attention-backend", "triton", "--disable-radix-cache", "--skip-tokenizer-init",
                                "--load-format", "dummy", "--port", str(port), "--dist-timeout", "180"])


def build_moe(sa, layer_idx: int = 1):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.model_loader.loader import _get_quantization_config
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.models.kimi_k3 import KimiK3MoE
    mc = ModelConfig.from_server_args(sa)
    qc = _get_quantization_config(mc, LoadConfig(load_format="dummy"))
    cfg = mc.hf_text_config
    with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        moe = KimiK3MoE(cfg, quant_config=qc, prefix=f"language_model.model.layers.{layer_idx}.mlp", layer_idx=layer_idx,
                        alt_stream=None)
    return moe, mc, qc


def load_into(moe, w: dict) -> None:
    """The harness's tensors into SGLang's parameters (checkpoint layout, this rank's shard), then the loader's post-load
    steps: every quant_method.process_weights_after_loading, then KimiK3MoE._merge_front_weights."""
    cp = lambda p, t: p.data.copy_(t.to(p.dtype).reshape(p.shape))
    cp(moe.gate.weight, w["router_weight"])
    cp(moe.gate.e_score_correction_bias, w["score_correction_bias"])
    cp(moe.routed_expert_down_proj.weight, w["latent_down_weight"])
    cp(moe.routed_expert_up_proj.weight, w["latent_up_weight"])
    cp(moe.routed_expert_norm.weight, w["routed_expert_norm_weight"])
    cp(moe.shared_experts.gate_up_proj.weight, w["shared_gate_up_weight"][shared_gate_up_rows(D.SHR_LOCAL, w["moe_in"].device)])
    cp(moe.shared_experts.down_proj.weight, w["shared_down_weight"])
    banks = checkpoint_banks(w)
    ex = moe.experts
    assert tuple(ex.w13_weight.shape) == tuple(banks["w13"].shape), (ex.w13_weight.shape, banks["w13"].shape)
    assert tuple(ex.w2_weight.shape) == tuple(banks["w2"].shape), (ex.w2_weight.shape, banks["w2"].shape)
    cp(ex.w13_weight, banks["w13"])
    cp(ex.w13_weight_scale, banks["w13_s"])
    cp(ex.w2_weight, banks["w2"])
    cp(ex.w2_weight_scale, banks["w2_s"])
    del banks
    for _, module in moe.named_modules():
        qm = getattr(module, "quant_method", None)
        if qm is not None and hasattr(qm, "process_weights_after_loading"):
            qm.process_weights_after_loading(module)
    moe._merge_front_weights()


def mxfp8_floor(v: torch.Tensor) -> torch.Tensor:
    """MXFP8 with the OCP MX scale rule (shared exponent floor(log2(amax)) - 8; values above 448 x scale clip), dequantized;
    the harness's mxfp8 takes ceil(log2(amax / 448)) instead, which never clips."""
    shape = v.shape
    b = v.float().reshape(*shape[:-1], shape[-1] // 32, 32)
    amax = b.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    ex = (torch.floor(torch.log2(amax)) - 8).clamp(-127, 127)
    sc = torch.exp2(ex)
    q = (b / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q.double() * sc.double()).reshape(shape)


def fp8_per_token(v: torch.Tensor) -> torch.Tensor:
    """e4m3 with one scale per row (amax / 448), dequantized (a probe of other activation quantizations)."""
    sc = (v.float().abs().amax(-1, keepdim=True) / 448).clamp(min=1e-30)
    return ((v.float() / sc).to(torch.float8_e4m3fn).float() * sc).double()


def routed_partial(w: dict, top, wt, zrule, hrule, per_expert: bool = False):
    """This rank's routed sum (before the all-reduce), float64, with the given rules for z and h (MXFP8 or a probe).
    per_expert: also each (token, k) row's contribution [T][K][L]."""
    T, L, IR = D.T, D.L, D.IR_LOCAL
    x = w["moe_in"].double()
    zq = zrule((x @ w["latent_down_weight"].double().T).float())
    gate_r, up_r = w13_rows(IR, x.device)
    R = torch.zeros(T, L, dtype=torch.float64, device=x.device)
    P = torch.zeros(T, top.shape[1], L, dtype=torch.float64, device=x.device)
    for e in sorted(set(top.flatten().tolist())):
        w13 = dequant(w["w13_blocks"][e], w["w13_scales"][e], 2 * IR, L)
        w2 = dequant(w["w2_blocks"][e], w["w2_scales"][e], L, IR)
        for t, k in (top == e).nonzero().tolist():
            h = situ(w13[gate_r] @ zq[t], w13[up_r] @ zq[t])
            P[t, k] = wt[t, k] * (w2 @ hrule(h.float()))
            R[t] += P[t, k]
    return (R, P) if per_expert else R


# ---- the float64 reference of the layer (the harness's arithmetic), the sums over the ranks by all_reduce ----
def reference(w: dict, group, zrule=mxfp8, hrule=mxfp8) -> tuple:
    """The layer in float64 (top-16 indices, y), the routed and shared sums over the ranks by all_reduce; zrule / hrule: how
    z and h are quantized (the default MXFP8 with ceil(log2(amax / 448)), as our kernel)."""
    import torch.distributed as dist
    T, L, K = D.T, D.L, D.TOPK
    x = w["moe_in"].double()
    sig = torch.sigmoid(x @ w["router_weight"].double().T)
    top = torch.topk(sig + w["score_correction_bias"].double(), K, dim=1).indices
    wt = sig.gather(1, top)
    wt = wt / wt.sum(1, keepdim=True)
    R = routed_partial(w, top, wt, zrule, hrule)
    sgu = x @ w["shared_gate_up_weight"].double()[shared_gate_up_rows(D.SHR_LOCAL, x.device)].T
    act = bf16(situ(sgu[:, :D.SHR_LOCAL], sgu[:, D.SHR_LOCAL:])).double()
    S = act @ w["shared_down_weight"].double().T
    dist.all_reduce(R, group=group)
    dist.all_reduce(S, group=group)
    rstd = torch.rsqrt((R ** 2).sum(1, keepdim=True) / L + 1e-5)
    rn = bf16((R * rstd * w["routed_expert_norm_weight"].double()).float()).double()
    up = rn @ w["latent_up_weight"].double().T
    y = bf16(bf16(bf16(up).float() + bf16(S).float()).float() + w["moe_prefix"].float())
    return top, y


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--reps", type=int, default=40)
    p.add_argument("--flush-mb", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--token-share", type=float, default=0.6)
    p.add_argument("--port", type=int, default=29711)
    p.add_argument("--eager", action="store_true", help="time the eager forward instead of the captured graph")
    p.add_argument("--cpu-check", action="store_true")
    p.add_argument("--compare-y", default=None, help="our layer's y and routing pairs (test_layer_one_gpu.py --dump-y, the "
                   "same --seed / --token-share): rank 0 compares them with SGLang's and with the float64 reference")
    p.add_argument("--kernels", type=int, default=0, help="after the timing: this many more launches under torch.profiler; "
                   "rank 0 prints each kernel's median time and count per launch")
    a = p.parse_args()
    if a.cpu_check:
        os.environ.setdefault("GPU_ARCHS", "gfx950")
        sa = sglang_server_args(a.model_dir, a.port)
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.configs.load_config import LoadConfig
        from sglang.srt.model_loader.loader import _get_quantization_config
        mc = ModelConfig.from_server_args(sa)
        qc = _get_quantization_config(mc, LoadConfig(load_format="dummy"))
        cfg = mc.hf_text_config
        print("hf_text_config:", type(cfg).__name__, {k: getattr(cfg, k, None) for k in (
            "hidden_size", "routed_expert_hidden_size", "moe_intermediate_size", "num_experts", "num_experts_per_token",
            "num_shared_experts", "hidden_act", "routed_scaling_factor", "moe_renormalize", "num_expert_group",
            "topk_group", "rms_norm_eps", "latent_moe_use_norm", "topk_method")})
        print("quant config:", type(qc).__name__, getattr(qc, "quant_format", None))
        print("CPU CHECK OK")
        return

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    # torchrun's agent hosts its own store; SGLang's tcp:// init must host its own on rank 0 (with this variable set every
    # rank would only connect to the agent's store at SGLang's port, and none hosts it)
    os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
    assert world == 8, "TP8: 8 ranks"
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    t0 = time.time()
    sa = sglang_server_args(a.model_dir, a.port)
    from sglang.srt.runtime_context import SpawnRanks, publish
    publish(sa, role="scheduler", ranks=SpawnRanks(world_rank=rank, dp_rank=None, gpu_id=rank))
    from sglang.srt.distributed.bootstrap import init_parallel_runtime
    init_parallel_runtime(server_args=sa, device="cuda", dist_port=a.port + 2)
    from sglang.srt.layers.moe.utils import initialize_moe_config
    from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
    initialize_moe_config()
    initialize_fp8_gemm_config()
    initialize_fp4_gemm_config()
    initialize_bf16_gemm_config()
    from sglang.srt.distributed import get_tp_group, tensor_model_parallel_all_reduce
    from sglang.srt.distributed.parallel_state import graph_capture
    import torch.distributed as dist
    tp = get_tp_group()
    log = lambda *s: print(*s, flush=True) if rank == 0 else None
    log(f"sglang runtime up ({time.time() - t0:.0f} s): tp {tp.world_size}, custom all-reduce {type(tp.ca_comm).__name__}")

    # the harness's weights (GPU 0's common tensors, this rank's own shard)
    w0 = random_weights(D, dev, a.seed, token_share=a.token_share)
    w = w0 if rank == 0 else random_weights(D, dev, a.seed + rank, common=w0, token_share=a.token_share)
    chk = torch.tensor([w["router_weight"].float().sum().item(), w["moe_in"].float().sum().item()], device=dev, dtype=torch.float64)
    allc = [torch.zeros_like(chk) for _ in range(world)]
    dist.all_gather(allc, chk, group=tp.device_group)
    assert all(torch.equal(c, allc[0]) for c in allc), "the common weights differ between ranks"

    moe, mc, qc = build_moe(sa)
    load_into(moe, w)
    torch.cuda.synchronize()
    ex = moe.experts
    log(f"KimiK3MoE built and loaded ({time.time() - t0:.0f} s): quant {type(qc).__name__}, experts' method "
        f"{type(ex.quant_method).__name__}, runner {type(getattr(ex.quant_method, 'runner', None)).__name__}, "
        f"fused front {moe._eligible_for_fused_front}, front fp32 {moe._front_fp32}, w13 {tuple(ex.w13_weight.shape)} "
        f"{ex.w13_weight.dtype} shuffled {getattr(ex.w13_weight, 'is_shuffled', None)}")

    # routing capture (the TopK module's output) for the check
    seen = {}
    topk_fwd = moe.topk.forward

    def topk_hook(*args, **kw):
        out = topk_fwd(*args, **kw)
        seen["topk"] = out
        return out

    x = w["moe_in"].clone()
    pre = w["moe_prefix"].clone()
    moe.topk.forward = topk_hook
    fr, fs = moe._forward_routed, moe._forward_shared

    def routed_hook(hs, rl, ri, latent):
        fr(hs, rl, ri, latent)
        seen["R"] = latent.clone()

    def shared_hook(gate_up, shared_output):
        fs(gate_up, shared_output)
        seen["S"] = shared_output.clone()

    moe._forward_routed, moe._forward_shared = routed_hook, shared_hook
    with torch.no_grad():
        y_eager = moe(x, prefix_sum=pre).clone()
    moe.topk.forward = topk_fwd
    moe._forward_routed, moe._forward_shared = fr, fs
    torch.cuda.synchronize()
    top_ref, y_ref = reference(w, tp.device_group)
    ids = seen["topk"][1] if isinstance(seen.get("topk"), tuple) else getattr(seen.get("topk"), "topk_ids", None)
    same = sum(set(ids[t].tolist()) == set(top_ref[t].tolist()) for t in range(D.T)) if ids is not None else -1
    n_exp = len(set(top_ref.flatten().tolist()))
    e = relrms(y_eager, y_ref)
    ulp = (y_eager.float() - y_ref.float()).abs().max().item()
    log(f"checks (eager, rank 0): routing {same}/{D.T} tokens pick the reference's 16 experts ({n_exp} experts used); "
        f"y vs float64 reference relRMS {e:.2e}, max |d| {ulp:.3e}")
    ok = same == D.T and e < 2e-2
    # this rank's partials before the all-reduce: SGLang's vs float64 references with each MXFP8 rule for z and h
    xr = w["moe_in"].double()
    sig = torch.sigmoid(xr @ w["router_weight"].double().T)
    wt = sig.gather(1, top_ref)
    wt = wt / wt.sum(1, keepdim=True)
    sgu = xr @ w["shared_gate_up_weight"].double()[shared_gate_up_rows(D.SHR_LOCAL, xr.device)].T
    S_ref = bf16(situ(sgu[:, :D.SHR_LOCAL], sgu[:, D.SHR_LOCAL:])).double() @ w["shared_down_weight"].double().T
    rows = [f"S {relrms(seen['S'], S_ref):.2e}"]
    zb = lambda v: mxfp8(v.to(torch.bfloat16).float())                 # z rounded to bf16 first (the bf16 front GEMM)
    hb = lambda v: mxfp8(v.to(torch.bfloat16).float())                 # h rounded to bf16 first
    for name, zr, hr in (("z ceil, h ceil", mxfp8, mxfp8), ("z floor, h floor", mxfp8_floor, mxfp8_floor),
                         ("z bf16+ceil, h ceil", zb, mxfp8), ("z ceil, h bf16+ceil", mxfp8, hb),
                         ("z ceil, h fp8 per token", mxfp8, fp8_per_token), ("z ceil, h exact", mxfp8, lambda v: v.double()),
                         ("z exact, h exact", lambda v: v.double(), lambda v: v.double())):
        rows.append(f"R[{name}] {relrms(seen['R'], routed_partial(w, top_ref, wt, zr, hr)):.2e}")
    # the error's profile against the first reference: per token, per 512-column block, and per (token, expert) rows
    Rr, P = routed_partial(w, top_ref, wt, mxfp8, mxfp8, per_expert=True)
    d = seen["R"].double() - Rr
    tok = (d.norm(dim=1) / Rr.norm(dim=1)).tolist()
    col = (d.reshape(D.T, -1, 512).norm(dim=(0, 2)) / Rr.reshape(D.T, -1, 512).norm(dim=(0, 2))).tolist()
    rows.append("per token " + " ".join(f"{v:.3f}" for v in tok))
    rows.append("per 512 columns " + " ".join(f"{v:.3f}" for v in col))
    msg = torch.zeros(1, device=dev)
    for g in range(world):   # one line per rank, in order
        if g == rank:
            print(f"rank {rank} SGLang partials vs float64 (relRMS): " + ", ".join(rows), flush=True)
        dist.barrier(group=tp.cpu_group)
    # the other references every rank takes part in (all_reduce): z and h unquantized; z rounded to bf16 before MXFP8
    exact = lambda v: v.double()
    _, y_exact = reference(w, tp.device_group, exact, exact)
    _, y_zbf16 = reference(w, tp.device_group, lambda v: mxfp8(v.to(torch.bfloat16).float()), mxfp8)
    if a.compare_y and rank == 0:
        ours = torch.load(a.compare_y)
        assert ours["seed"] == a.seed and ours["token_share"] == a.token_share, "--compare-y: another seed / token share"
        y_ours = ours["y"].to(dev)
        pk = ours["pairs"].to(dev)
        e_ours, w_ours = (pk >> 32).long(), (pk & 0xFFFFFFFF).int().view(torch.float32)
        tw = seen["topk"][0].float()
        same_sets = sum(set(e_ours[t].tolist()) == set(ids[t].tolist()) for t in range(D.T))
        # routing weights per (token, expert): ours vs SGLang's
        ws = torch.zeros(D.T, D.E, device=dev).scatter_(1, ids.long(), tw)
        wo = torch.zeros(D.T, D.E, device=dev).scatter_(1, e_ours, w_ours)
        ul = lambda u, v: (u.to(torch.bfloat16).view(torch.int16).int() - v.to(torch.bfloat16).view(torch.int16).int()).abs()
        d = ul(y_eager, y_ours)
        print(f"OURS vs SGLANG (rank 0, the same weights and inputs): routing {same_sets}/{D.T} tokens the same 16 experts, "
              f"weights max |d| {(ws - wo).abs().max().item():.2e}; y relRMS {relrms(y_eager, y_ours):.2e}, max |d| "
              f"{(y_eager.float() - y_ours.float()).abs().max().item():.3e}, bf16 values equal {(d == 0).sum().item()}, off "
              f"by 1 ulp {(d == 1).sum().item()}, 2-15 {((d > 1) & (d < 16)).sum().item()}, 16+ {(d >= 16).sum().item()} (of "
              f"{d.numel()}); vs the float64 reference: ours {relrms(y_ours, y_ref):.2e}, SGLang {e:.2e}; |y| rms "
              f"{y_ref.float().pow(2).mean().sqrt().item():.3f}", flush=True)
        mo = lambda v: v.float() - pre.float()   # the layer's own part: y minus the residual prefix (the same in both)
        for name, yr in (("z, h unquantized", y_exact), ("MXFP8 of fp32 z (ours)", y_ref), ("MXFP8 of bf16 z", y_zbf16)):
            print(f"REFERENCE [{name}], y - prefix relRMS: ours {relrms(mo(y_ours), mo(yr)):.2e}, SGLang "
                  f"{relrms(mo(y_eager), mo(yr)):.2e}", flush=True)
        print(f"OURS vs SGLANG, y - prefix: relRMS {relrms(mo(y_eager), mo(y_ours)):.2e}; vs the float64 reference: ours "
              f"{relrms(mo(y_ours), mo(y_ref)):.2e}, SGLang {relrms(mo(y_eager), mo(y_ref)):.2e}; rms of y - prefix "
              f"{mo(y_ref).pow(2).mean().sqrt().item():.3f}", flush=True)

    # the captured graph, as SGLang's decode runs it
    with torch.no_grad():
        if a.eager:
            run = lambda: moe(x, prefix_sum=pre)
        else:
            with graph_capture() as gctx:
                for _ in range(2):
                    moe(x, prefix_sum=pre)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=gctx.stream):
                    y_graph = moe(x, prefix_sum=pre)
            run = g.replay
            run()
            torch.cuda.synchronize()
            e2 = relrms(y_graph, y_eager)
            # (not bit for bit: measured 4.27e-3 on MI355X 10-13, the same every run, so a different but fixed summation
            # order in the captured graph than in eager mode; which kernel differs is not known)
            log(f"graph replay vs eager y: relRMS {e2:.2e}")
            ok = ok and e2 < 2e-2
    flush = torch.zeros(a.flush_mb << 18, dtype=torch.int32, device=dev)
    line = torch.zeros(8, dtype=torch.bfloat16, device=dev)
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e2e, mine = [], []
    with torch.no_grad():
        for i in range(a.reps + 3):
            torch.amax(flush)
            torch.cuda.synchronize()
            dist.barrier(group=tp.cpu_group)
            tensor_model_parallel_all_reduce(line)       # the GPUs lined up on the stream
            st.record()
            run()
            en.record()
            en.synchronize()
            us = torch.tensor([st.elapsed_time(en) * 1e3], device=dev)
            allu = [torch.zeros_like(us) for _ in range(world)]
            dist.all_gather(allu, us, group=tp.device_group)
            if i >= 3:
                e2e.append(max(u.item() for u in allu))
                mine.append([u.item() for u in allu])
    if a.kernels:   # the kernels of a launch (the same flush / line-up / replay sequence), from the profiler's device times
        from torch.profiler import ProfilerActivity, profile
        with torch.no_grad(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i in range(a.kernels):
                torch.amax(flush)
                torch.cuda.synchronize()
                dist.barrier(group=tp.cpu_group)
                tensor_model_parallel_all_reduce(line)
                torch.cuda.synchronize()
                with torch.profiler.record_function("layer"):
                    run()
                    torch.cuda.synchronize()
        if rank == 0:
            evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            lay = [e for e in prof.events() if e.name == "layer" and e.device_type != torch.autograd.DeviceType.CUDA]
            per = {}
            for L in lay:   # the device kernels that ran inside this launch's window (graph kernels included)
                t0, t1 = L.time_range.start, L.time_range.end
                ks = sorted((e for e in evs if t0 <= e.time_range.start < t1 + 50), key=lambda e: e.time_range.start)
                for j, e in enumerate(ks):
                    per.setdefault((j, e.name), []).append((e.time_range.start - ks[0].time_range.start, e.time_range.elapsed_us()))
            print(f"KERNELS (rank 0, {a.kernels} launches, 1 GB flushed before each; start after the first kernel's start, "
                  f"duration, medians in us):", flush=True)
            for (j, name), v in sorted(per.items()):
                if len(v) >= a.kernels // 2:
                    print(f"  {j:2d} start {statistics.median(x for x, _ in v):6.1f} dur {statistics.median(d for _, d in v):6.1f}"
                          f"  {name[:110]}", flush=True)
    if rank == 0:
        e2e.sort()
        per = [statistics.median(r[g] for r in mine) for g in range(world)]
        print(f"SGLANG LATENCY ({a.reps} launches, 8 GPUs, {'eager' if a.eager else 'HIP graph'}, {a.flush_mb} MB flushed "
              f"before each): e2e median {e2e[len(e2e) // 2]:.1f} us (min {e2e[0]:.1f}, max {e2e[-1]:.1f}); per GPU "
              f"medians {[round(v, 1) for v in per]}", flush=True)
        print("ALL CHECKS PASSED" if ok else "CHECKS FAILED", flush=True)
    dist.barrier(group=tp.cpu_group)


if __name__ == "__main__":
    main()
