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


def build_mpk(args, logits, output_tokens, tokens):
    max_seq_length = 2
    max_num_pages = 4
    page_size = 16
    max_num_batched_requests = 1
    max_num_batched_tokens = 1

    step = torch.zeros((1,), dtype=torch.int32, device="cuda")
    input_tokens = torch.zeros((1,), dtype=torch.int64, device="cuda")
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

    logits_dt = mpk.attach_input(torch_tensor=logits, name="argmax_in")
    part_value = mpk.new_tensor(
        dims=(1, args.num_workers), dtype=mi.bfloat16, name="argmax_part_value")
    part_index = mpk.new_tensor(
        dims=(1, args.num_workers), dtype=mi.int64, name="argmax_part_index")
    output_dt = mpk.attach_input(torch_tensor=output_tokens, name="output_token")

    mpk.argmax_partial_layer(
        input=logits_dt,
        output=(part_value, part_index),
        grid_dim=(args.num_workers, 1, 1),
        block_dim=(128, 1, 1),
    )
    mpk.argmax_reduce_layer(
        input=(part_value, part_index),
        output=output_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    return mpk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--output-dir", default="outputs/argmax_v2_minimal")
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()

    device = pick_device() if args.device is None else args.device
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    logits = torch.randn((1, args.vocab_size), dtype=torch.bfloat16, device="cuda")
    expected = int(torch.argmax(logits[0]).item())
    tokens = torch.zeros((1, 2), dtype=torch.int64, device="cuda")
    output_tokens = torch.full((1, 1), -1, dtype=torch.int64, device="cuda")

    mpk = build_mpk(args, logits, output_tokens, tokens)
    os.makedirs(args.output_dir, exist_ok=True)
    mpk.compile(output_dir=args.output_dir)
    mpk.init_request_func()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    starter.record()
    mpk()
    ender.record()
    torch.cuda.synchronize()

    output = int(output_tokens[0, 0].item())
    copied = int(tokens[0, 1].item())
    latency_ms = starter.elapsed_time(ender)
    print(
        f"device={device} expected={expected} output_tokens={output} "
        f"tokens[0,1]={copied} latency_ms={latency_ms:.3f}")
    if output != expected or copied != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
