"""DeepSeek-V4.1-Flash's LM head as one static kernel (11-20, DSV41_HEAD=1 in dsv41_static_hook): the logits of a decode step.

  x [T, 5120] bf16      the final norm of the streams' collapse (inference/model.py Transformer.forward: hc_pre with the last
                        layer's FFN pre, then norm): the last MoE kernel's next input (its post, gamma = norm.weight)
  head_part             gemm_tile: x . head.weight^T for this GPU's vocab rows [r V, r V + V) (V = 129280 / GPUs, padded to whole
                        128-row blocks), fp32 K parts in slots (bf16 x bf16, fp32 accumulation; the official keeps the weight
                        in fp32 and multiplies x.float(): the same products, another order of the fp32 sums)
  head_sent             allreduce_send: each GPU's row of the parts' sum, bf16 (as SGLang's bf16 LM head output), to every GPU
  logits                gather_f32: [T, 129280] fp32 on every GPU (SGLang's next_token_logits_buffer, which its sampler reads)

python head_demo.py --gpus 4 --checkpoint DIR [--reps 30]   (logits against x . W^T in fp64, as bf16; the time of a launch)
"""
import argparse
import copy
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
T = int(os.environ.get("STATIC_TOKENS", "8"))   # tokens (rows) per step: the kernel's T (8, 16, 32 or 64)
HID = 5120
# (11-23) the MoE demo of this module's T: set by its loader (dsv41_static_layer._head_demo) to the copy of the same T, else the
# package's (a process of one T); as deepseek_v4_1_csa2.demo.MOE_DEMO
MOE_DEMO = None


def moe_demo_mod():
    """deepseek_v4_1_moe.demo of this module's T (MOE_DEMO, or the package's)"""
    m = MOE_DEMO
    if m is None:
        sys.path.insert(0, os.path.dirname(HERE))
        from deepseek_v4_1_moe import demo as m
    assert m.T == T, f"head_demo T = {T} with a MoE demo of T = {m.T}"
    return m
KP = int(os.environ.get("DSV41_HEAD_KP", 1))        # the GEMM's K parts (11-20: 1: 253 tasks, at most 2 an SM: GEMM 52.6 us vs 58.2 at 2, about 61 at 4; a task costs about 2.75 us besides its bytes)
SEND = int(os.environ.get("DSV41_HEAD_SEND", 148))  # the sends' tasks
GQ = int(os.environ.get("DSV41_HEAD_GQ", 16))       # the gather's tasks a token
ARGMAX = os.environ.get("DSV41_HEAD_ARGMAX", "0") != "0"   # (11-20) the gather also stores each token's greedy next token


def head_tensors(read, tp: int, rank: int, dev, shared: dict = None) -> dict:
    """GPU rank's tensors: its vocab rows of head.weight (bf16, zero rows to whole 128-row blocks), x and the logits. shared
    (11-21: a layer of another T): its head_w and logits' width instead of reading head.weight."""
    if shared is not None:
        hw, n = shared["head_w"], shared["logits"].shape[1]
    else:
        w = read("head.weight")
        V = w.shape[0] // tp
        assert V * tp == w.shape[0] and w.shape[1] == HID, f"head.weight {tuple(w.shape)}"
        hw = torch.zeros(-(-V // 128) * 128, HID, dtype=torch.bfloat16)
        hw[:V] = w[rank * V:(rank + 1) * V]
        hw, n = hw.to(dev).contiguous(), tp * V
    w = {"head_w": hw, "x": torch.zeros(T, HID, dtype=torch.bfloat16, device=dev),
         "logits": torch.zeros(T, n, dtype=torch.float32, device=dev)}
    if ARGMAX:
        w["head_ids"] = torch.zeros(T, dtype=torch.int32, device=dev)
    return w


def build_graph(mpk, w: dict) -> None:
    head_nodes = moe_demo_mod().head_nodes   # (this T's: MOE_DEMO)
    t = {}
    a = lambda k: t.setdefault(k, mpk.attach_input(torch_tensor=w[k], name=k))
    head_nodes(mpk, a, w, "x")


def hand_plan(mpk, sms: int) -> dict:
    """The GEMM's tasks over every SM in turn, then the sends (after the K parts they add), then the gathers."""
    from mirage.mpk import compiler
    nodes = compiler.graph_nodes(mpk)
    names = {n.graph_idx: compiler.tname(mpk, n.outputs[0]) for n in nodes}
    Vp = next(n.inputs[1].dim(0) for n in nodes if names[n.graph_idx] == "head_part")
    grid = {"head_part": (Vp // 128, KP, 1), "head_sent": (SEND, 1, 1), "logits": (GQ, T, 1), "logits_am": (GQ, T, 1)}
    grids = {idx: grid[name] for idx, name in names.items()}
    tasks = lambda name: [(idx, (x, y, 0)) for idx in grids if names[idx] == name
                          for x in range(grids[idx][0]) for y in range(grids[idx][1])]
    lists = [[] for _ in range(sms)]
    k = 0
    for name in ("head_part", "head_sent", "logits", "logits_am"):
        for tk in tasks(name):
            lists[k % sms].append(tk)
            k += 1
    return {"grids": grids, "lists": [copy.deepcopy(lists) for _ in range(mpk.num_gpus)]}


def build(ws: list, out: str, extra_flags: list = None, profile: bool = False, device: int = 0, num_gpus: int = None,
          rank: int = None, exchange_pool=None):
    """As moe_demo.build: the kernel, each GPU's node buffers, the information from compile_plan."""
    from mirage.mpk import compiler
    from mirage.mpk.static_megakernel import StaticMegakernel
    gpus = num_gpus or len(ws)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    mpk = StaticMegakernel(num_gpus=gpus, num_sms=sms)
    build_graph(mpk, ws[0])
    p = hand_plan(mpk, sms)
    paths, info = compiler.compile_plan(mpk, p, os.path.join(out, "schedules"))
    nodes = compiler.graph_nodes(mpk)
    nb = any("STATIC_NO_START_BARRIER" in f for f in (extra_flags or []))
    bufs = []
    for wg in ws:
        d = wg["x"].device
        bufs.append({compiler.tname(mpk, n.outputs[0]): torch.zeros(n.info["out_buffer"](n)[0], dtype=torch.uint8, device=d)
                     for n in nodes if n.info["out_buffer"] is not None and not n.info["exchange"] and
                     not (nb and n.info["out_buffer"](n)[1] >= 0 and not n.info["out_buffer"](n)[2])})
    sk = mpk.compile_static(paths, gpu_tensors=[{**wg, **bg} for wg, bg in zip(ws, bufs)], out_dir=os.path.join(out, "build"),
                            extra_flags=extra_flags, profile=profile, device=device,
                            **({} if rank is None else {"rank": rank, "exchange": exchange_pool.take(mpk.exchange_region_bytes())}))
    for wg in ws:   # (moe_demo.build: every GPU's init stores land before any launch)
        torch.cuda.synchronize(wg["x"].device)
    return sk, bufs, dict(info, paths=paths, mpk=mpk, plan=p)


def main():
    sys.path.insert(0, os.path.dirname(HERE))
    from deepseek_v4_1_csa2 import weights as attn_weights
    from mirage.mpk.static_harness import Harness
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--out", default="/tmp/head_demo")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--flag", action="append", default=[])
    ap.add_argument("--timeline", action="store_true", help="the timing build: each node's first start and last end, GPU 0")
    a = ap.parse_args()
    read = attn_weights.checkpoint_reader(a.checkpoint)
    devs = [torch.device("cuda", g) for g in range(a.gpus)]
    ws = [head_tensors(read, a.gpus, r, d) for r, d in enumerate(devs)]
    x = (torch.randn(T, HID, generator=torch.Generator().manual_seed(5)) * 1.0).bfloat16()
    for wg, d in zip(ws, devs):
        wg["x"].copy_(x.to(d))
    sk, bufs, info = build(ws, a.out, extra_flags=a.flag, profile=a.timeline)
    h = Harness(sk)
    h.launch(timeout_s=60)
    torch.cuda.synchronize()
    W = read("head.weight")
    ref = (x.double() @ W.double().T).to(torch.bfloat16).float()   # (each GPU's row as bf16, then fp32)
    got = [wg["logits"].cpu() for wg in ws]
    rel = ((got[0].double() - ref.double()).norm() / ref.double().norm()).item()
    eq = (got[0] == ref).float().mean().item()
    same = all(torch.equal(g, got[0]) for g in got[1:])
    am = (got[0].argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"logits relRMS {rel:.2e}, equal {eq:.4f}, argmax equal {am:.3f}, the same on every GPU {same}", flush=True)
    spans = []
    for _ in range(a.reps):
        h.launch(512 << 20, timeout_s=60)
        spans.append(h.span_us())
    spans.sort()
    if a.timeline:   # the last launch, GPU 0: per node the first start, the median task, the last end (after the start barrier)
        import json
        import statistics
        from mirage.mpk import compiler
        from mirage.mpk.cost_records import stamps_of_gpu
        sch = json.load(open(info["paths"][0]))
        names = {n.graph_idx: compiler.tname(info["mpk"], n.outputs[0]) for n in compiler.graph_nodes(info["mpk"])}
        bar = h.timing()[0][0]
        per = {}
        for t, v in stamps_of_gpu(sch, h.task_times()[0]).items():
            per.setdefault(names[sch["all_tasks"][t]["node"]], []).append(((v["start"] - bar) / 1e3, (v["end"] - bar) / 1e3))
        for nm, v in sorted(per.items(), key=lambda kv: max(e for _, e in kv[1])):
            print(f"  {nm:<10} tasks {len(v):>4} first start {min(s for s, _ in v):6.2f} median task "
                  f"{statistics.median(e - s for s, e in v):5.2f} last end {max(e for _, e in v):6.2f}", flush=True)
    again = all(torch.equal(wg["logits"].cpu(), got[0]) for wg in ws)
    print(f"head: median span {spans[len(spans) // 2]:.1f} us (min {spans[0]:.1f}); logits after {a.reps} launches the same "
          f"{again}", flush=True)
    if ARGMAX:   # the kernel's ids against torch.argmax of its own logits: a random x, x = 0 (every logit 0: index 0), and
        # a tie (token 0's best row, on whichever GPU, copied over row 5 of GPU 0: the earlier index wins)
        def ids_ok(tag):
            lg = [wg["logits"].cpu() for wg in ws]
            ok = all(torch.equal(wg["head_ids"].cpu().long(), lg[r].argmax(-1)) for r, wg in enumerate(ws))
            print(f"argmax {tag}: ids {ws[0]['head_ids'].cpu().tolist()}, torch.argmax {lg[0].argmax(-1).tolist()}, every GPU "
                  f"equal {ok}", flush=True)
            return ok
        ok = ids_ok("random x")
        for wg in ws:
            wg["x"].zero_()
        h.launch(timeout_s=60)
        torch.cuda.synchronize()
        ok = ids_ok("x = 0") and ok
        V = W.shape[0] // len(ws)
        for wg, d in zip(ws, devs):
            wg["x"].copy_(x.to(d))
        best = int(ref[0].argmax())
        if best > 5:
            row = ws[best // V]["head_w"][best % V].cpu()
            ws[0]["head_w"][5].copy_(row.to(devs[0]))
            h.launch(timeout_s=60)
            torch.cuda.synchronize()
            ok = ids_ok(f"tie (token 0's row {best} on GPU {best // V} copied to row 5 of GPU 0)") and ok
            ok = ok and int(ws[0]["head_ids"][0]) == 5
        print(f"argmax checks {'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
