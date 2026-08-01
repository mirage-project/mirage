"""Check MPK's GPT-OSS against the HuggingFace reference.

Greedy decoding is deterministic, so the two token streams must agree until
bf16 rounding flips a near-tie. Divergence at step n is reported rather than
asserted away: what matters is whether it happens immediately (a wiring bug)
or late and on a close call (rounding).

Runs the reference first and frees it before building MPK, so only one copy of
the 39 GB model is resident at a time.

    python demo/gpt_oss/verify_against_hf.py --num-tokens 32
"""

import argparse
import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mirage as mi
from mirage.mpk.models.gpt_oss.builder import GptOssBuilder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="/raid/catalyst/models/gpt-oss-20b-bf16")
    p.add_argument("--prompt", type=str,
                   default="Give me a short introduction to large language models.")
    p.add_argument("--num-tokens", type=int, default=32)
    p.add_argument("--max-seq-length", type=int, default=512)
    return p.parse_args()


def hf_reference(args, tokenizer, ids):
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    with torch.no_grad():
        out = model.generate(
            ids.unsqueeze(0).cuda(),
            max_new_tokens=args.num_tokens,
            do_sample=False,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=tokenizer.pad_token_id or 199999)
        # Logits for the last prompt position, which localises a mismatch to
        # the prefill rather than to the decode loop.
        logits = model(ids.unsqueeze(0).cuda()).logits[0, -1].float().cpu()
    generated = out[0, ids.shape[-1]:].cpu()
    del model, out
    gc.collect()
    torch.cuda.empty_cache()
    return generated, logits


def run_mpk(args, ids):
    device = "cuda"
    prompt_len = ids.shape[-1]
    total = 1
    tokens = torch.zeros(total, args.max_seq_length, dtype=torch.long,
                         device=device)
    tokens[0, :prompt_len] = ids.to(device)
    meta = {
        "step": torch.zeros(total, dtype=torch.int32, device=device),
        "tokens": tokens,
        "input_tokens": torch.zeros(8, 1, dtype=torch.long, device=device),
        "output_tokens": torch.zeros(8, 1, dtype=torch.long, device=device),
        "num_new_tokens": torch.ones(total, dtype=torch.int32, device=device),
        "prompt_lengths": torch.full((total,), prompt_len, dtype=torch.int32,
                                     device=device),
        "qo_indptr_buffer": torch.empty(2, dtype=torch.int32, device=device),
        "paged_kv_indptr_buffer": torch.empty(2, dtype=torch.int32,
                                              device=device),
        "paged_kv_indices_buffer": torch.empty(16, dtype=torch.int32,
                                               device=device),
        "paged_kv_indices_snapshot": torch.empty(16, dtype=torch.int32,
                                                 device=device),
        "paged_kv_last_page_len_buffer": torch.empty(1, dtype=torch.int32,
                                                     device=device),
    }
    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    mpk = mi.PersistentKernel(
        mode="offline", world_size=1, mpi_rank=0,
        num_workers=num_workers, num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=args.max_seq_length,
        max_num_batched_requests=1, max_num_batched_tokens=8,
        max_num_pages=16, page_size=4096,
        eos_token_id=-1,  # never stop early, so the comparison has full length
        meta_tensors=meta, profiler_tensor=None, trace_name="",
        spec_decode_config=None, use_cutlass_kernel=False,
    )
    GptOssBuilder(mpk).build_from_model(model_name=args.model,
                                        model_path=args.model)
    mpk.compile(output_dir=None)
    mpk()
    torch.cuda.synchronize()
    end = meta["step"][0].item() + 1
    generated = tokens[0, prompt_len:end].cpu()
    mpk.finalize()
    return generated


def main():
    args = parse_args()
    torch.set_default_dtype(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True)
    ids = tokenizer([text], return_tensors="pt").input_ids[0]
    print(f"prompt is {ids.shape[-1]} tokens")

    print("=== HuggingFace reference")
    hf_tokens, hf_logits = hf_reference(args, tokenizer, ids)

    print("=== MPK")
    mpk_tokens = run_mpk(args, ids)

    n = min(len(hf_tokens), len(mpk_tokens), args.num_tokens)
    hf_tokens, mpk_tokens = hf_tokens[:n], mpk_tokens[:n]
    same = (hf_tokens == mpk_tokens)
    first_diff = int((~same).nonzero()[0].item()) if not same.all() else -1

    print(f"\ncompared {n} greedy tokens")
    print(f"  HF : {tokenizer.decode(hf_tokens)!r}")
    print(f"  MPK: {tokenizer.decode(mpk_tokens)!r}")
    if first_diff < 0:
        print(f"\nPASSED: all {n} tokens identical")
        return 0
    print(f"\n  first divergence at token {first_diff}: "
          f"HF {hf_tokens[first_diff].item()} "
          f"({tokenizer.decode(hf_tokens[first_diff:first_diff+1])!r}) vs "
          f"MPK {mpk_tokens[first_diff].item()} "
          f"({tokenizer.decode(mpk_tokens[first_diff:first_diff+1])!r})")
    print(f"  matched {first_diff}/{n} before diverging")
    # A near-tie at the divergence point means rounding, not a wiring bug.
    top2 = hf_logits.topk(2).values
    print(f"  reference top-2 logit gap at the first generated position: "
          f"{(top2[0] - top2[1]).item():.4f}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
