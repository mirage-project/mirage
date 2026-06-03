"""Single-layer unit test for the v3 Channel-based linear (linear_sm100_v3.cuh),
driven through the v2 runtime — NO full model, just ONE linear task.

Models the harness on demo/argmax_v2_minimal.py: build a PersistentKernel
(use_v2_runtime=True), attach external tensors, register a single
linear_layer_v3 / linear_with_residual_layer_v3, compile, run, check vs torch.

Lets us reproduce in isolation:
  * the lm_head HANG:   --N 151936 --K 4096   (~1187 tiles, heavy ring reuse)
  * the SLOW case:      --N 2048   --K 4096   (qkv/mlp/attn shapes)

Shape convention (from register_linear_sm100_v3_task):
  input  x  = [M, K]   (M = batched tokens, K = reduction)
  weight w  = [N, K]
  output o  = [M, N]
  ref: o = x @ w.T  (+ residual)

Usage:
  python linear_v3_unittest.py --M 8 --K 4096 --N 2048 --tiles-per-task 3
  python linear_v3_unittest.py --M 8 --K 4096 --N 151936 --tiles-per-task 3
  python linear_v3_unittest.py --M 8 --K 4096 --N 2048 --residual
"""
import argparse
import os

import torch

import mirage as mi


def pick_device() -> int:
    free = []
    for idx in range(torch.cuda.device_count()):
        torch.cuda.set_device(idx)
        free_bytes, _ = torch.cuda.mem_get_info()
        free.append((free_bytes, idx))
    return max(free)[1]


def build_mpk(args, x, w, residual, out, chain_w):
    # Minimal serving scaffold (linear uses none of it, but the ctor needs it).
    max_seq_length = 2
    max_num_pages = 4
    page_size = 16
    max_num_batched_requests = 1
    max_num_batched_tokens = args.M

    step = torch.zeros((1,), dtype=torch.int32, device="cuda")
    tokens = torch.zeros((1, 2), dtype=torch.int64, device="cuda")
    input_tokens = torch.zeros((1,), dtype=torch.int64, device="cuda")
    output_tokens = torch.zeros((1,), dtype=torch.int64, device="cuda")
    num_new_tokens = torch.zeros((1,), dtype=torch.int32, device="cuda")
    prompt_lengths = torch.zeros((1,), dtype=torch.int32, device="cuda")
    qo_indptr_buffer = torch.zeros(
        (max_num_batched_requests + 1,), dtype=torch.int32, device="cuda")
    paged_kv_indptr_buffer = torch.zeros(
        (max_num_batched_requests + 1,), dtype=torch.int32, device="cuda")
    paged_kv_indices_buffer = torch.zeros(
        (max_num_pages,), dtype=torch.int32, device="cuda")
    paged_kv_last_page_len_buffer = torch.zeros(
        (max_num_batched_requests,), dtype=torch.int32, device="cuda")

    mpk = mi.PersistentKernel(
        mode="offline",
        world_size=1,
        mpi_rank=0,
        num_workers=args.num_workers,
        num_local_schedulers=1,
        num_remote_schedulers=0,
        max_seq_length=max_seq_length,
        max_num_batched_requests=max_num_batched_requests,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_pages=max_num_pages,
        page_size=page_size,
        eos_token_id=-1,
        meta_tensors={
            "step": step,
            "tokens": tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "num_new_tokens": num_new_tokens,
            "prompt_lengths": prompt_lengths,
            "qo_indptr_buffer": qo_indptr_buffer,
            "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
            "paged_kv_indices_buffer": paged_kv_indices_buffer,
            "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
        },
        profiler_tensor=None,
        trace_name="",
        spec_decode_config=None,
        use_cutlass_kernel=True,
        use_v2_runtime=True,
    )

    def plain(inp, wt, o):
        if args.variant == "v3":
            mpk.linear_layer_v3(input=inp, weight=wt, output=o,
                                tiles_per_task=args.tiles_per_task)
        else:
            mpk.linear_layer_v2(input=inp, weight=wt, output=o,
                                tiles_per_task=args.tiles_per_task)

    x_dt = mpk.attach_input(torch_tensor=x, name="lin_x")

    # --chain C: prepend C square (K->K) linears feeding the final one, so the
    # graph has C+1 dependent linear ops cycling the instruction ring — this is
    # what reproduces cross-task overlap (slow) and cross-op slot reuse (hang),
    # which single-op runs cannot. Square weights are constant (ones/K) so the
    # chained activations stay finite; we don't correctness-check the chain.
    cur = x_dt
    for c in range(args.chain):
        wc = mpk.attach_input(torch_tensor=chain_w, name=f"chain_w_{c}")
        tc = mpk.new_tensor(dims=(args.M, args.K), dtype=mi.bfloat16,
                            name=f"chain_t_{c}")
        plain(cur, wc, tc)
        cur = tc

    w_dt = mpk.attach_input(torch_tensor=w, name="lin_w")
    out_dt = mpk.attach_input(torch_tensor=out, name="lin_out")
    if args.residual:
        res_dt = mpk.attach_input(torch_tensor=residual, name="lin_res")
        if args.variant == "v3":
            mpk.linear_with_residual_layer_v3(
                input=cur, weight=w_dt, residual=res_dt, output=out_dt,
                tiles_per_task=args.tiles_per_task)
        else:
            mpk.linear_with_residual_layer_v2(
                input=cur, weight=w_dt, residual=res_dt, output=out_dt)
    else:
        plain(cur, w_dt, out_dt)
    return mpk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=8, help="batched tokens (rows)")
    p.add_argument("--K", type=int, default=4096, help="reduction dim")
    p.add_argument("--N", type=int, default=2048, help="output dim (must %128==0)")
    p.add_argument("--tiles-per-task", type=int, default=3)
    p.add_argument("--variant", choices=["v2", "v3"], default="v3")
    p.add_argument("--residual", action="store_true")
    p.add_argument("--chain", type=int, default=0,
                   help="prepend C square (K->K) linears before the final one")
    p.add_argument("--num-workers", type=int, default=128)
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--output-dir", default="outputs/linear_v3_unittest")
    args = p.parse_args()

    assert args.N % 128 == 0, f"N must be divisible by 128, got {args.N}"
    device = pick_device() if args.device is None else args.device
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    x = torch.randn((args.M, args.K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((args.N, args.K), dtype=torch.bfloat16, device="cuda")
    residual = torch.randn((args.M, args.N), dtype=torch.bfloat16, device="cuda")
    out = torch.zeros((args.M, args.N), dtype=torch.bfloat16, device="cuda")
    # Square chain weight ~ I/sqrt(K)-ish: small entries so chained activations
    # stay bounded across C hops. (Chain correctness is not checked.)
    chain_w = (torch.randn((args.K, args.K), dtype=torch.bfloat16, device="cuda")
               * (1.0 / (args.K ** 0.5)))

    # Reference (only meaningful for chain==0): same bf16 round-trip as v3.
    ref = torch.matmul(x.float(), w.float().t())
    if args.residual:
        ref = ref.to(torch.bfloat16).float() + residual.float()
    ref = ref.to(torch.bfloat16)

    print(f"[cfg] variant={args.variant} M={args.M} K={args.K} N={args.N} "
          f"tiles_per_task={args.tiles_per_task} residual={args.residual} "
          f"chain={args.chain} num_tiles={args.N//128} device={device}")

    mpk = build_mpk(args, x, w, residual, out, chain_w)
    os.makedirs(args.output_dir, exist_ok=True)
    mpk.compile(output_dir=args.output_dir)
    mpk.init_request_func()

    # Single run — correctness (only for chain==0; chained output has no ref).
    torch.cuda.synchronize()
    mpk()
    torch.cuda.synchronize()

    if args.chain == 0:
        diff = (out.float() - ref.float()).abs()
        rel = diff / (ref.float().abs() + 1e-3)
        max_abs = diff.max().item()
        max_rel = rel.max().item()
        ok = torch.allclose(out.float(), ref.float(), atol=1.0, rtol=0.05)
        print(f"[correctness] max_abs={max_abs:.4f} max_rel={max_rel:.4f} "
              f"allclose(atol=1,rtol=0.05)={ok}")
    else:
        ok = bool(torch.isfinite(out.float()).all().item())
        print(f"[correctness] chain={args.chain} (no ref) finite={ok}")

    # Latency.
    for _ in range(16):
        mpk()
    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(args.reps):
        mpk()
    ender.record()
    torch.cuda.synchronize()
    avg = starter.elapsed_time(ender) / args.reps
    print(f"[latency] {avg:.4f} ms/call over {args.reps} reps")

    if not ok:
        print("[RESULT] CORRECTNESS FAIL")
        raise SystemExit(1)
    print("[RESULT] PASS")


if __name__ == "__main__":
    main()
