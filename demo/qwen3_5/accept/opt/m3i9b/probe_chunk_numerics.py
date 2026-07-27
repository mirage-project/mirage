#!/usr/bin/env python3
"""M3-I9b: per-position logit probe + prefill-state fingerprint for the
admission-cap (chunk-boundary) numerics question.

Why this exists
---------------
Stage 4 of the M3-I9 window found that ``--per-request-token-cap auto`` changes
TOKENS at bs4 only: p10-logic flips at generated position 49 against a
reference margin of 0.625 (not tie-waivable). The cap changes exactly one
observable thing -- the prefill CHUNK DECOMPOSITION (and, at bs4, which
requests prefill in the same iteration). This probe measures the ENGINE's own
logits so the mechanism can be named instead of guessed.

Two levers make it cheap:

1. ``config.max_seq_length`` is a RUNTIME field (persistent_kernel.cuh:282
   reads ``config.max_seq_length``, while the ``tokens`` row stride uses the
   compile-time ``MPK_MAX_SEQ_LENGTH``).  So ONE compiled kernel can be
   truncated to stop right after the iteration that emits generated position N,
   by re-initialising the runtime config with ``max_seq_length = plen + N + 1``.
   ``finalize_persistent_kernel()`` frees everything the previous init
   allocated, so the re-init loop does not leak.

   Retirement is on the ABSOLUTE step (``step + advance + 1 >= max_seq_length``)
   and is identical for every slot, so truncating cannot change anything that
   happens BEFORE the truncation point: the schedule the probe measures is the
   same schedule the full run would have executed.

2. ``builder.expose_logits`` makes ``argmax_in`` a host-visible ``[mbt, vocab]``
   buffer holding the LAST iteration's logits, i.e. exactly the logit vector
   that produced generated position N.

Modes
-----
``poscurve``   for each requested position N: truncate, run, read the target
               request's full logit row -> engine top1/top2 margin, the logits
               at the reference's own candidate ids, and a hash of the row.
               Both arms run the identical token prefix up to the first flip,
               so the per-position row hash IS the arm-to-arm divergence test.
``statedump``  run prefill ONLY (truncate at N=0) and dump every persistent
               per-layer state -- GDN conv state, GDN fp32 recurrent state, the
               paged K/V cache -- so a layer-level bisect between two arms is a
               tensor diff instead of another GPU run.

Nothing here writes to the AC-3 gate or the reference; it only reads the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

import os

HERE = Path(__file__).resolve().parent
# MPK_ACCEPT_DIR lets the probe live outside the (shared) mirage clone so the
# clone stays clean for whoever owns the next window.
ACCEPT = Path(os.environ.get("MPK_ACCEPT_DIR", str(HERE.parents[1]))).resolve()
REPO = ACCEPT.parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(ACCEPT / "opt" / "m3i9"))

import mirage as mi                                                # noqa: E402
from mirage.mpk.models.qwen3_5.builder import Qwen35Builder        # noqa: E402
from protocol_sim import simulate                                  # noqa: E402

META_ORDER = ["step", "tokens", "input_tokens", "output_tokens",
              "num_new_tokens", "prompt_lengths", "qo_indptr_buffer",
              "paged_kv_indptr_buffer", "paged_kv_indices_buffer",
              "paged_kv_last_page_len_buffer", "paged_kv_indices_snapshot"]


def log(m: str) -> None:
    print(f"[m3i9b] {m}", flush=True)


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()[:16]


# --------------------------------------------------------------------------
def build(args):
    dev = "cuda"
    bs, mbt, msl = args.bs, args.mbt, args.compile_msl
    num_pages_per_req = -(-msl // args.page_size)
    max_num_pages = max(bs * num_pages_per_req + 4, 8)
    meta = {
        "step": torch.zeros(bs, dtype=torch.int32, device=dev),
        "tokens": torch.zeros((bs, msl), dtype=torch.long, device=dev),
        "input_tokens": torch.zeros((mbt, 1), dtype=torch.long, device=dev),
        "output_tokens": torch.zeros((mbt, 1), dtype=torch.long, device=dev),
        "num_new_tokens": torch.ones(bs, dtype=torch.int32, device=dev),
        "prompt_lengths": torch.zeros(bs, dtype=torch.int32, device=dev),
        "qo_indptr_buffer": torch.zeros(bs + 1, dtype=torch.int32, device=dev),
        "paged_kv_indptr_buffer": torch.zeros(bs + 1, dtype=torch.int32, device=dev),
        "paged_kv_indices_buffer": torch.zeros(max_num_pages, dtype=torch.int32, device=dev),
        "paged_kv_last_page_len_buffer": torch.zeros(bs, dtype=torch.int32, device=dev),
        "paged_kv_indices_snapshot": torch.zeros(max_num_pages, dtype=torch.int32, device=dev),
    }
    torch.set_default_dtype(torch.bfloat16)
    nw, ns = mi.get_configurations_from_gpu(0)
    kw = {}
    if args.cap:
        kw["max_tokens_per_request"] = args.cap
    mpk = mi.PersistentKernel(
        mode="offline", world_size=1, mpi_rank=0, num_workers=nw,
        num_local_schedulers=ns, num_remote_schedulers=0,
        max_seq_length=msl, max_num_batched_requests=bs,
        max_num_batched_tokens=mbt, max_num_pages=max_num_pages,
        page_size=args.page_size, eos_token_id=-1, meta_tensors=meta,
        profiler_tensor=None, trace_name="", spec_decode_config=None,
        use_cutlass_kernel=True, **kw)
    builder = Qwen35Builder(mpk)
    builder.expose_logits = True
    t0 = time.time()
    builder.build_from_model(model_name=args.model, model_path=args.model_path)
    log(f"graph assembled in {time.time() - t0:.1f}s (bs={bs} mbt={mbt} "
        f"msl={msl} cap={args.cap})")
    kdir = Path(args.kernel_dir)
    kdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.reuse_kernel and (kdir / "task_graph_rank0.json").exists():
        mpk.load_mpk_kernel(output_dir=str(kdir), skip_validation=True)
        log(f"reused kernel {kdir} ({time.time() - t0:.1f}s)")
    else:
        mpk.compile(output_dir=str(kdir))
        log(f"compiled in {time.time() - t0:.1f}s -> {kdir}")
    return mpk, builder, meta


def launcher_module(mpk):
    """The compile() path sets init/launch/finalize on the object but NOT
    init_request_func, and never registers the module in sys.modules
    (persistent_kernel.py:3236-3241 vs :3358-3364).  A C module-level function
    carries its module in __self__, so this recovers the whole table either
    way."""
    mod = getattr(mpk.init_func, "__self__", None)
    if mod is None:
        mod = sys.modules.get("__mirage_launcher")
    return mod


def reinit(mpk, new_msl: int, kdir: Path) -> None:
    """Re-point the runtime config at a smaller max_seq_length.

    finalize frees every gpu_malloc the previous init made
    (persistent_kernel.cuh:2123), so this is leak-free; init ends by calling
    init_request_resources(), so step/request_ids/page queue come back clean.
    """
    mpk.finalize_func()
    mpk.max_seq_length = new_msl
    ptrs = [mpk.meta_tensors[k].data_ptr() for k in META_ORDER]
    names = list(mpk._model_tensors.keys())
    tptrs = [t.data_ptr() for t in mpk._model_tensors.values()]
    mpk.init_func(ptrs, 0, mpk.mpi_rank, mpk.num_workers,
                  mpk.num_local_schedulers, mpk.num_remote_schedulers,
                  new_msl, mpk.total_num_requests, mpk.eos_token_id,
                  mpk.allocate_nvshmem_teams, names, tptrs,
                  str(kdir / f"task_graph_rank{mpk.mpi_rank}.json"))


def watchdog_run(mpk, meta, timeout_s: float = 180.0):
    """Launch and poll `step` on a side stream so a wedge raises instead of
    hanging (same contract as mpk_engine_run._watchdog)."""
    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    done = torch.cuda.Event()
    mpk()
    done.record()
    t0 = time.time()
    last = None
    while not done.query():
        if time.time() - t0 > timeout_s:
            with torch.cuda.stream(side):
                st = meta["step"].to("cpu").tolist()
            side.synchronize()
            raise RuntimeError(f"WATCHDOG: no completion in {timeout_s}s, "
                               f"step={st}")
        time.sleep(2.0)
        with torch.cuda.stream(side):
            cur = meta["step"].to("cpu").tolist()
        side.synchronize()
        if cur != last:
            last = cur
    torch.cuda.synchronize()


def load_wave(args):
    ref = json.load(open(ACCEPT / "reference" / "reference_outputs.json"))["results"]
    ids = args.prompts.split(",")
    slots = [{"prompt_id": p, "input_ids": ref[p]["input_ids"],
              "output_ids": ref[p]["output_ids"],
              "topk_ids": ref[p].get("topk_ids_per_step"),
              "topk_logits": ref[p].get("topk_logits_per_step")} for p in ids]
    return slots


def fill(meta, slots, extra_prefix=0):
    meta["tokens"].zero_()
    meta["step"].zero_()
    meta["num_new_tokens"].fill_(1)
    for i, s in enumerate(slots):
        ids = list(s["input_ids"])
        if extra_prefix:
            ids = ids + list(s["output_ids"])[:extra_prefix]
        meta["tokens"][i, :len(ids)] = torch.tensor(ids, dtype=torch.long,
                                                    device="cuda")
        meta["prompt_lengths"][i] = len(ids)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--mbt", type=int, default=16)
    ap.add_argument("--page-size", type=int, default=256)
    ap.add_argument("--cap", type=int, default=None,
                    help="MPK_MAX_TOKENS_PER_REQUEST; omit for the uncapped arm")
    ap.add_argument("--compile-msl", type=int, required=True)
    ap.add_argument("--kernel-dir", required=True)
    ap.add_argument("--reuse-kernel", action="store_true")
    ap.add_argument("--prompts", required=True,
                    help="comma list of prompt ids IN SLOT ORDER")
    ap.add_argument("--target", required=True, help="prompt id under test")
    ap.add_argument("--state-hash", action="store_true",
                    help="at EVERY probed position, record a per-layer sha of "
                         "the TARGET SLOT's persistent state (GDN conv, GDN "
                         "fp32 recurrent, its K/V cache page).  Comparing two "
                         "arms' hash tables localises the first (position, "
                         "layer) at which the slot's state diverges without "
                         "writing a single big tensor to disk.")
    ap.add_argument("--state-at", type=int, default=None,
                    help="also dump every persistent per-layer state at this "
                         "generated position (needs --raw-dir)")
    ap.add_argument("--positions", default="0",
                    help="comma list of generated positions, or a:b:step")
    ap.add_argument("--dump-rows", default="",
                    help="comma list of positions whose FULL logit row is saved")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if ":" in args.positions:
        a, b, st = (int(x) for x in args.positions.split(":"))
        positions = list(range(a, b, st))
    else:
        positions = [int(x) for x in args.positions.split(",") if x != ""]
    dump_rows = {int(x) for x in args.dump_rows.split(",") if x != ""}
    raw = Path(args.raw_dir) if args.raw_dir else None
    if raw:
        raw.mkdir(parents=True, exist_ok=True)

    slots = load_wave(args)
    assert len(slots) == args.bs, f"{len(slots)} prompts for bs={args.bs}"
    t_slot = [i for i, s in enumerate(slots) if s["prompt_id"] == args.target]
    assert t_slot, f"{args.target} not in the wave"
    t_slot = t_slot[0]
    plen = len(slots[t_slot]["input_ids"])
    plens = [len(s["input_ids"]) for s in slots]

    mpk, builder, meta = build(args)
    kdir = Path(args.kernel_dir)
    logits_buf = builder.buffers["argmax_in"]
    _mod = launcher_module(mpk)
    init_request = getattr(_mod, "init_request_func", None) if _mod else None
    log(f"launcher module {_mod}; init_request_func={init_request is not None}")

    report = {"arm": {"bs": args.bs, "mbt": args.mbt, "cap": args.cap,
                      "compile_msl": args.compile_msl},
              "prompts": args.prompts, "target": args.target,
              "target_slot": t_slot, "target_plen": plen, "plens": plens,
              "state_at": args.state_at, "points": []}

    if args.state_at is not None and args.state_at not in positions:
        positions = sorted(set(positions) | {args.state_at})

    for n in positions:
        msl_rt = plen + n + 1
        if msl_rt > args.compile_msl:
            log(f"skip N={n}: needs msl {msl_rt} > compiled {args.compile_msl}")
            continue
        reinit(mpk, msl_rt, kdir)
        fill(meta, slots)
        if init_request is not None:
            init_request()
        torch.cuda.synchronize()
        t0 = time.time()
        watchdog_run(mpk, meta)
        wall = time.time() - t0

        steps = meta["step"].tolist()
        toks = meta["tokens"].cpu()
        emitted = int(toks[t_slot, plen + n].item()) if plen + n < args.compile_msl else None
        gen = toks[t_slot, plen:plen + n + 1].tolist()
        ref_gen = slots[t_slot]["output_ids"][:n + 1]

        # which packed row carried the target's last token this iteration?
        sim_c = simulate(plens, args.mbt, msl_rt, cap=args.cap)
        last_it = sim_c["iters"][-1] if sim_c["iters"] else None
        pred_row = None
        if last_it is not None:
            off = 0
            for j, r in enumerate(last_it["slots"]):
                if r == t_slot and last_it["chunks"][j] > 0:
                    pred_row = off + last_it["chunks"][j] - 1
                    break
                off += last_it["chunks"][j]

        lg = logits_buf.float()
        row_argmax = [int(torch.argmax(lg[i]).item()) for i in range(lg.shape[0])]
        match_rows = [i for i, a in enumerate(row_argmax) if a == emitted]
        row = pred_row if (pred_row is not None and pred_row in match_rows) else \
            (match_rows[0] if match_rows else pred_row)

        pt = {"position": n, "runtime_msl": msl_rt, "wall_s": round(wall, 3),
              "steps": steps, "emitted": emitted,
              "ref_token": slots[t_slot]["output_ids"][n],
              "tokens_match_reference_so_far": gen == ref_gen,
              "first_token_mismatch": next(
                  (k for k, (x, y) in enumerate(zip(gen, ref_gen)) if x != y), None),
              "predicted_row": pred_row, "rows_matching_emitted": match_rows,
              "row_used": row,
              "sim_iterations": sim_c["n_iterations"],
              "sim_last_iter": last_it}
        if row is not None:
            v = lg[row]
            top = torch.topk(v, 8)
            pt["engine_top8"] = [[int(i), float(x)] for i, x in
                                 zip(top.indices.tolist(), top.values.tolist())]
            pt["engine_margin_top1_top2"] = float(top.values[0] - top.values[1])
            pt["row_sha"] = sha(logits_buf[row])
            rk = slots[t_slot]["topk_ids"]
            rl = slots[t_slot]["topk_logits"]
            if rk and n < len(rk):
                pt["ref_topk"] = [[int(i), float(x)] for i, x in zip(rk[n], rl[n])]
                pt["engine_at_ref_ids"] = [float(v[int(i)].item()) for i in rk[n]]
                pt["ref_margin_top1_top2"] = float(rl[n][0] - rl[n][1])
            if raw and n in dump_rows:
                torch.save(logits_buf[row].clone().cpu(),
                           raw / f"logits_pos{n}.pt")
        if args.state_hash:
            # slot -> page: the page FIFO starts as [0, 1, .., max_pages-1] and
            # each slot takes one page for these <=256-token sequences, in slot
            # order (persistent_kernel.cuh init_kernel + step 3), so slot i owns
            # page i for the whole wave (no slot migrates in these waves).
            pg = t_slot
            pt["state_sha"] = {
                "conv": [sha(builder.conv_state[l, t_slot])
                         for l in range(builder.conv_state.shape[0])],
                "recur": [sha(builder.recurrent_state[l, t_slot])
                          for l in range(builder.recurrent_state.shape[0])],
                "k": [sha(builder.k_cache[l, pg]) for l in range(builder.k_cache.shape[0])],
                "v": [sha(builder.v_cache[l, pg]) for l in range(builder.v_cache.shape[0])],
            }
        report["points"].append(pt)
        log(f"N={n:<3d} msl={msl_rt:<4d} emitted={emitted} "
            f"row={row} pred={pred_row} margin="
            f"{pt.get('engine_margin_top1_top2')} sha={pt.get('row_sha')} "
            f"({wall:.1f}s)")

        if args.state_at is not None and n == args.state_at:
            assert raw, "--raw-dir is required for --state-at"
            st = {"conv_state": builder.conv_state.clone().cpu(),
                  "recurrent_state": builder.recurrent_state.clone().cpu(),
                  # only the pages this wave actually touched
                  "k_cache": builder.k_cache[:, :args.bs + 1].clone().cpu(),
                  "v_cache": builder.v_cache[:, :args.bs + 1].clone().cpu(),
                  "argmax_in": logits_buf.clone().cpu(),
                  "meta": {"steps": steps, "plens": plens,
                           "target_slot": t_slot, "cap": args.cap}}
            torch.save(st, raw / "state.pt")
            report["state_sha"] = {k: sha(v) for k, v in st.items()
                                   if isinstance(v, torch.Tensor)}
            log(f"state dumped -> {raw/'state.pt'} {report['state_sha']}")

    Path(args.out).write_text(json.dumps(report, indent=1))
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
