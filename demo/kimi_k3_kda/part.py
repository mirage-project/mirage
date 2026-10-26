"""The K3 KDA layer (TP8, 8 tokens) for the static compiler: the graph and its start plan (mpk-kernels tools/gen_kernels.py
generates the kernel from them), and the loop's hooks (python/mirage/mpk/loop/part.py).

Graph (SGLang's KDA target-verify contract: state / conv window read from the framework's pools by slot, ReplaySSM rings and
per-token conv windows written, pools unchanged; the bank rows' number and strides at run time):
  x -> projection (gemm_tile, W_proj K-block-major, padded to 6400 rows, 3 uneven K parts in slots)
    -> short_conv_silu (the K parts added, q / k / v convolved + SiLU; its output 0xFF-armed)
    -> delta_rule_recurrence (polls the conv output; gated delta rule + o_norm; y_rec 0xFF-armed)
    -> o_proj (gemm_tile, warp MMAs polling y_rec, 64-row tiles, each tile's columns pushed to every GPU)
    -> attn_residual (2 parts per token on the 2 CTAs of a cluster: the GPUs' partials added, u = the new prefix out,
       softmax mix over the bank rows and u, RMSNorm).
Inputs: the KDA harness's dump of one layer on rank 0 (the hand-written KDA kernel's harness before this compiler, 1 bank row build); build(input_dir=None): zero
tensors of the shapes (code generation without a GPU). Every GPU gets rank 0's tensors.
Graph options: num_sms (0: the device's; code generation: 148), eps (0: 1e-6, the harness's; K3: rms_norm_eps).
"""
import os

import numpy as np
import torch

T, H, K = 8, 7168, 1536      # tokens, hidden, o_proj's K per GPU (12 heads x 128)
REL, ABS = 2e-2, 1e-2        # part.yaml error_limits (the KDA harness's)
EPS = 1e-6                   # the KDA bodies' and harness's RMSNorm eps
PROJ_PAD = 6400              # W_proj with zero rows up to this many (128-row tiles: 50)
PROJ_PARTS, PROJ_BIG = 3, 24 # the projection's K parts: 2 of 24 K tiles, the rest (8) in the third

# the serving kernel (mpk-kernels tools/gen_kernels.py): the host's weights and run tensors (mpk_layer_host.cuh), meta.json's
# description, the launch's cluster size (attn_residual's 2 parts per token)
WEIGHTS = ["wproj_kb", "conv_w", "wfb", "prm", "oproj_weight", "q", "gain", "ow", "layer_info"]
RUN = ["x", "bank", "prefix", "slot", "icw_row", "conv_pool", "state_pool", "ring_k", "ring_v", "ring_g", "ring_beta", "icw",
       "hidden", "new_prefix"]
DESCRIPTION = ("Kimi K3 KDA layer, TP8, 8 tokens per launch (q/k/v/g/f_a/beta projection, short conv, gated delta-rule recurrence "
               "+ o_norm, o_proj, all-reduce, MLP-side attention residual over 1..16 bank rows; SGLang's KDA target-verify contract: "
               "state / conv window read from the pools by slot, ReplaySSM rings and per-token conv windows written, pools unchanged)")
CLUSTER = 2


def serving_build():
    """The serving kernel's build (static_schedule.Build; every define measured in SGLang on 10-09/10; core.cuh, config.cuh):
      STATIC_RESET_IN_KERNEL   the kernel resets its buffers (the launch number in device memory: one launch per run)
      STATIC_RESET_PARITY      no start barrier: two copies of the reset buffers, a launch resets the next one's copy
      STATIC_PDL_PREFILL       the first projection tiles' weight stages issued before the PDL wait (gemm_tile.cuh PREFILL_*)
      STATIC_PDL_TRIGGER 1     the next kernel may launch at this one's start (its own wait still waits for this grid)
      STATIC_NOWAIT_NODE       the recurrence's CTAs skip the PDL wait (they read only this kernel's outputs and the pools)
    plus the task times hooks (a -DMPK_TASK_TIMES build); the host is mpk_layer_host.cuh."""
    from mirage.mpk.static_schedule import Build, NodeRef
    return Build(defines=(("STATIC_RESET_IN_KERNEL", 1), ("STATIC_RESET_PARITY", 1), ("STATIC_PDL_PREFILL", 1),
                          ("STATIC_PDL_TRIGGER", 1), ("STATIC_NOWAIT_NODE", NodeRef("delta_rule_recurrence"))),
                 switches=frozenset({"task_times"}), host=False)


def read_bf16(path, shape):
    return torch.from_numpy(np.fromfile(path, dtype=np.uint16).reshape(shape)).view(torch.bfloat16)


INPUTS = [("y_rec", "yrec_dev_bf16", (T, K)), ("oproj_weight", "wop_bf16", (H, K)), ("bank", "bank_bf16", (T, 2, H)),
          ("prefix", "partial_bf16", (T, H)), ("q", "q_bf16", (H,)), ("gain", "gain_bf16", (H,)), ("ow", "ow_bf16", (H,))]


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).double()


def ref_attn_residual(a, prefix, bank, q, gain, ow, eps=EPS) -> torch.Tensor:
    """50_harness.cuh's float64 reference of the attention residual (:526-536), in torch: a = the summed partials [T][H]
    (float64), the rest as the task reads them. src = (bank row 0, bank row 1, u = bf16(a + prefix)); per source
    s = (q * gain . src) / sqrt(mean(src^2) + eps); w = softmax(s); h = bf16(w . src); out = bf16(h / sqrt(mean(h^2) + eps) * ow)."""
    u = bf16(a + prefix.double())
    src = torch.stack([bank[:, 0].double(), bank[:, 1].double(), u], dim=1)          # [T][3][H]
    c = q.double() * gain.double()
    sc = (src * c).sum(-1) / torch.sqrt((src * src).mean(-1) + eps)                    # [T][3]
    w = torch.softmax(sc, dim=-1)
    h = bf16((w.unsqueeze(-1) * src).sum(1))                                            # [T][H]
    rstd = 1.0 / torch.sqrt((h * h).mean(-1, keepdim=True) + eps)
    return bf16(h * rstd * ow.double())


HEADS, PROJ_N, PRM_N = 12, 6336, 12 + 12 * 128 + 128
# the recurrence's inputs (graph option recurrence): name, file, shape, numpy dtype ("bf16": bf16 bits)
NCONV = 3 * K
PROJ_INPUTS = [("x", "x_bf16", (T, H), "bf16"), ("wproj", "wproj_bf16", (PROJ_N, H), "bf16"),
               ("conv_w", "convW_f32", (NCONV, 4), np.float32), ("conv_s", "convS_bf16", (NCONV, 4), "bf16")]
REC_INPUTS = [("conv_out", "conv_dev_bf16", (3, T, K), "bf16"), ("proj_out", "proj_bf16", (T, PROJ_N), "bf16"),
              ("wfb", "wfb_bf16", (HEADS * 128, 128), "bf16"), ("prm", "prm_f32", (PRM_N,), np.float32),
              ("state", "state_f32", (HEADS, 128, 128), np.float32)]


def read(path, shape, dtype):
    if SHAPES_ONLY:   # build(input_dir=None): the graph from zero tensors of the inputs' shapes (code generation only)
        return torch.zeros(shape, dtype=torch.bfloat16 if dtype == "bf16" else torch.from_numpy(np.zeros(1, dtype=dtype)).dtype)
    return read_bf16(path, shape) if dtype == "bf16" else torch.from_numpy(np.fromfile(path, dtype=dtype).reshape(shape))


SHAPES_ONLY = False   # build(input_dir=None): no dump read; zero CPU tensors, the tensors stay on the CPU (no GPU needed)

def build(input_dir: str, gpus: int, tokens: int, options: dict):
    from mirage.core import bfloat16, float32
    from mirage.mpk.static_megakernel import StaticMegakernel
    global SHAPES_ONLY
    if tokens != T:
        raise ValueError(f"the KDA dump holds {T} tokens, asked for {tokens}")
    SHAPES_ONLY = input_dir is None
    inputs = {name: read(os.path.join(input_dir or "", f"{file}.bin"), shape, "bf16") for name, file, shape in INPUTS}
    inputs["y_rec"] = torch.zeros(T, K, dtype=torch.bfloat16)   # the recurrence writes it
    inputs.update({name: read(os.path.join(input_dir or "", f"{file}.bin"), shape, dt) for name, file, shape, dt in REC_INPUTS})
    inputs.update({name: read(os.path.join(input_dir or "", f"{file}.bin"), shape, dt) for name, file, shape, dt in PROJ_INPUTS})
    inputs["wproj"] = torch.cat([inputs["wproj"], torch.zeros(PROJ_PAD - PROJ_N, H, dtype=torch.bfloat16)])
    inputs["rec_in"] = torch.zeros(T, PROJ_PAD, dtype=torch.bfloat16)   # the conv's output
    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    ws = []
    for g in range(gpus):
        dev = torch.device("cpu") if SHAPES_ONLY else torch.device("cuda", g)
        w = {name: t.to(dev) for name, t in inputs.items()}
        # W_proj K-block-major, [H / 64][PROJ_PAD][64] (viewed [PROJ_PAD, H]); the reference keeps wproj
        w["wproj_kb"] = w["wproj"].view(PROJ_PAD, H // 64, 64).transpose(0, 1).contiguous().view(PROJ_PAD, H)
        w["oproj_partial"] = torch.zeros(T, H, dtype=f32, device=dev)
        w["hidden"] = torch.zeros(T, H, dtype=bf, device=dev)
        w["new_prefix"] = torch.zeros(T, H, dtype=bf, device=dev)   # u = bf16(prefix + o_proj's sum), the next layer's prefix
        # the framework's pools, rings and per-token conv windows, bound per run; here one slot / scratch row and a ring of T
        # (the shapes only size them)
        w.pop("conv_s", None)
        w.pop("state", None)
        w["conv_pool"] = torch.zeros(3, NCONV, dtype=bf, device=dev)          # [slots x 3, NCONV] (oldest first)
        w["state_pool"] = torch.zeros(HEADS * 128, 128, dtype=f32, device=dev)  # [slots x heads x 128 v, 128 k]
        w["ring_k"] = torch.zeros(HEADS * T, 128, dtype=bf, device=dev)         # [slots x heads x L, 128]
        w["ring_v"] = torch.zeros(HEADS * T, 128, dtype=bf, device=dev)
        w["ring_g"] = torch.zeros(HEADS * T, 128, dtype=f32, device=dev)
        w["ring_beta"] = torch.zeros(HEADS, T, dtype=f32, device=dev)          # [slots x heads, L]
        w["icw"] = torch.zeros(T * 3, NCONV, dtype=bf, device=dev)              # [rows x steps x 3, NCONV]
        w["slot"] = torch.zeros(1, dtype=i32, device=dev)
        w["icw_row"] = torch.zeros(1, dtype=i32, device=dev)
        w["layer_info"] = torch.tensor([2, 2 * H, H, T, T, 0, 0, 0], dtype=i32, device=dev)   # nvb, bank strides, L, steps
        ws.append(w)
        if SHAPES_ONLY:
            break
    if not SHAPES_ONLY:
        torch.cuda.set_device(0)
    mpk = StaticMegakernel(num_gpus=gpus, num_sms=int(options.get("num_sms") or 0) or None)
    mpk.max_bufs = 32   # the serving tensors need more than config.cuh's default 24 slots (the layer defines STATIC_MAX_BUFS)
    a = lambda k: mpk.attach_input(torch_tensor=ws[0][k], name=k)
    t = {k: a(k) for k in ws[0]}
    eps = float(options.get("eps") or EPS)
    part_ = mpk.new_tensor(dims=(T, PROJ_PAD), dtype=float32, name="proj_partial")
    mpk.gemm_tile_layer(input=t["x"], weight=t["wproj_kb"], output=part_, combine="slots", weight_kblocked=True,
                        uneven_parts=(2, PROJ_BIG, PROJ_PARTS))
    mpk.short_conv_silu_layer(input=part_, conv_weight=t["conv_w"], conv_state=t["conv_pool"], output=t["rec_in"], armed=True,
                              qkv_fp32=True,   # q / k / v in fp32 into the recurrence, as SGLang
                              serve={k: t[k] for k in ("icw", "icw_row", "ring_k", "ring_v")} | {"slot": t["slot"], "info": t["layer_info"]})
    mpk.delta_rule_recurrence_layer(conv=t["rec_in"], proj=t["rec_in"], wfb=t["wfb"], prm=t["prm"], state=t["state_pool"],
                                    output=t["y_rec"],
                                    serve={"slot": t["slot"], "ring_g": t["ring_g"], "ring_beta": t["ring_beta"], "info": t["layer_info"]},
                                    daf=True,     # polls rec_in, no counter
                                    qkv_fp32=True,   # q / k / v in fp32 into the recurrence, as SGLang
                                    armed=True,   # o_proj polls y_rec
                                    onorm_eps=float(options["eps"]) if options.get("eps") else None)
    sent = mpk.new_tensor(dims=(T, H), dtype=bfloat16, name="oproj_sent")
    mpk.gemm_tile_layer(input=t["y_rec"], weight=t["oproj_weight"], output=sent, combine="push", warp_mma=True)
    mpk.attn_residual_layer(input=sent, bank=t["bank"], prefix=t["prefix"], q=t["q"], gain=t["gain"], ow=t["ow"],
                            output=t["hidden"], u_output=t["new_prefix"], bank_info=t["layer_info"], eps=eps)
    return ws, mpk


def start_plan(mpk, gpus: int, num_sms: int) -> dict:
    """The projection's big K parts 0, 1 of tile x on SMs 2x, 2x + 1, its small part round robin on the SMs idle during the
    projection (o_proj-only 100..111, attn 124..147); conv task c on the SM of its tile's half; recurrence head h on SM 112 + h;
    o_proj tile i (64 rows) on SM i; attn_residual token t's parts on SMs 124 + 2t, 125 + 2t (one cluster)."""
    from mirage.mpk.compiler import graph_nodes
    pg, sc, rec, gemm, res = graph_nodes(mpk)
    assert [n.name for n in (pg, sc, rec, gemm, res)] == ["gemm_tile", "short_conv_silu", "delta_rule_recurrence", "gemm_tile",
                                                          "attn_residual"]
    lists = [[] for _ in range(num_sms)]
    cols = pg.inputs[1].dim(0) // pg.grid[0]   # the projection tile's columns (128)
    # (the uneven split: k152 / k154, 28.9 vs 29.2 with 24 + 24 + 8 K tiles)
    idle = [i for i in list(range(100, 112)) + list(range(124, 148)) if i < num_sms]
    for x in range(pg.grid[0]):
        lists[2 * x].append((pg.graph_idx, (x, 0, 0)))
        lists[2 * x + 1].append((pg.graph_idx, (x, 1, 0)))
    k = 0
    for x in range(pg.grid[0]):
        for y_ in range(2, PROJ_PARTS):
            lists[idle[k % len(idle)]].append((pg.graph_idx, (x, y_, 0)))
            k += 1
    for c in range(sc.grid[0]):
        x, h = c * 64 // cols, (c * 64 % cols) // 64
        lists[2 * x + min(h, 1)].append((sc.graph_idx, (c, 0, 0)))
    for h_ in range(rec.grid[0]):   # before anything else there
        lists[(112 + h_) % num_sms].append((rec.graph_idx, (h_, 0, 0)))
    for i in range(gemm.grid[0]):
        lists[i % num_sms].append((gemm.graph_idx, (i, 0, 0)))
    for t_ in range(res.grid[0]):   # (the recurrence's SMs are busy until ~26.6 us, k126)
        for y_ in range(res.grid[1]):
            lists[(124 + 2 * t_ + y_) % num_sms].append((res.graph_idx, (t_, y_, 0)))
    return {"grids": {pg.graph_idx: (pg.grid[0], PROJ_PARTS, 1)}, "lists": [lists for _ in range(gpus)]}


def _check(name, got: torch.Tensor, ref: torch.Tensor) -> bool:
    got, ref = got.double().cpu(), ref.double()
    d = (got - ref).abs()
    bad = int((d > REL * ref.abs() + ABS).sum())
    rel = ((got - ref).norm() / ref.norm()).item()
    print(f"  {name:<22} max|diff| {d.max().item():.3g} (max|ref| {ref.abs().max().item():.3g}) relRMS {rel:.2e} "
          f"bad {bad}/{ref.numel()}{'  <-- MISMATCH' if bad else ''}", flush=True)
    return bad == 0


def _bits_differ(got: torch.Tensor, ref: torch.Tensor) -> int:
    ints = {2: torch.int16, 4: torch.int32}
    return int((got.cpu().contiguous().view(ints[got.element_size()]) != ref.contiguous().view(ints[ref.element_size()])).sum())


def check(h, ws, input_dir: str, out_dir: str) -> bool:
    """GPU 0's hidden vs the harness's float64 attention-residual reference (ref_attn_residual, the same math as
    50_harness.cuh) on o_proj's sum of the y_rec the kernel computed; the hidden bit-identical on all GPUs. ref_attn_residual
    itself first checked on the harness's own inputs against its dumped reference (hid_ref_f64)."""
    h.launch(536870912, timeout_s=60)
    y = ws[0]["y_rec"].double().cpu()
    w = ws[0]["oproj_weight"].double().cpu()
    p_ref = y @ w.T                                                     # [T][H] float64
    cpu = {k: ws[0][k].cpu() for k in ("bank", "prefix", "q", "gain", "ow")}
    a_h = torch.from_numpy(np.fromfile(os.path.join(input_dir, "attn_sum_f64.bin"), dtype=np.float64).reshape(T, H))
    hid_h = torch.from_numpy(np.fromfile(os.path.join(input_dir, "hid_ref_f64.bin"), dtype=np.float64).reshape(T, H))
    ok = _check("ref fn vs harness ref", ref_attn_residual(a_h, **cpu), hid_h)
    a_ours = bf16(p_ref) * len(ws)   # every GPU pushed its partial as bf16; the task adds the 8 in fp32
    ok &= _check("hidden GPU 0", ws[0]["hidden"], ref_attn_residual(a_ours, **cpu))
    same = sum(torch.equal(ws[g]["hidden"].cpu().view(torch.int16), ws[0]["hidden"].cpu().view(torch.int16))
               for g in range(1, len(ws)))
    print(f"  hidden bit-identical on {same}/{len(ws) - 1} other GPUs", flush=True)
    h.report()
    return ok and same == len(ws) - 1


def watch(ws) -> dict:
    return {"hidden": ws[0]["hidden"]}


def before_launch(ws, i: int) -> None:
    """Nothing: the pools are read, not written (every launch computes the same)."""
