import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT = "Write a detailed 20000 word essay on the history of artificial intelligence, covering its origins, key milestones, major researchers, and future prospects."

MODELS = {
    "qwen": "Qwen/Qwen3-0.6B",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
}

CHAT_TEMPLATES = {
    "qwen": [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": PROMPT},
    ],
    "llama": [
        {"role": "system", "content": "You are llama, a helpful AI assistant."},
        {"role": "user", "content": PROMPT},
    ],
}


def run_transformers(model, tokenizer, input_ids, output_len, batch_size):
    # Duplicate input for batch
    batched_input_ids = input_ids.repeat(batch_size, 1)
    attention_mask = torch.ones_like(batched_input_ids)
    prompt_len = input_ids.shape[-1]

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_ids=batched_input_ids, attention_mask=attention_mask,
                           max_new_tokens=16, do_sample=False,
                           eos_token_id=-1)
    torch.cuda.synchronize()

    # Measure prefill time (generate 1 token = prefill + 1 decode step)
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        prefill_start.record()
        _ = model.generate(input_ids=batched_input_ids, attention_mask=attention_mask,
                           max_new_tokens=1, do_sample=False,
                           eos_token_id=-1)
        prefill_end.record()
    torch.cuda.synchronize()
    prefill_ms = prefill_start.elapsed_time(prefill_end)

    # Benchmark full generation
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        starter.record()
        output_ids = model.generate(input_ids=batched_input_ids, attention_mask=attention_mask,
                                    max_new_tokens=output_len, do_sample=False,
                                    eos_token_id=-1)
        ender.record()
    torch.cuda.synchronize()
    total_time_ms = starter.elapsed_time(ender)

    # Subtract prefill to get decode-only time
    decode_time_ms = total_time_ms - prefill_ms
    num_generated = output_ids.shape[-1] - prompt_len
    return num_generated, total_time_ms, decode_time_ms


def main():
    parser = argparse.ArgumentParser(description="Transformers baseline benchmark")
    parser.add_argument("--output-len", type=int, default=2048, help="Max new tokens to generate")
    args = parser.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    device = torch.device("cuda")

    results = []

    for model_name in ["qwen", "llama"]:
        model_id = MODELS[model_name]
        print(f"\n{'='*60}")
        print(f"Loading {model_id}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        model.eval()

        # Tokenize prompt
        text = tokenizer.apply_chat_template(
            CHAT_TEMPLATES[model_name], tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)
        prompt_len = input_ids.shape[-1]
        print(f"Prompt length: {prompt_len} tokens")

        for bs in [1, 2, 4, 8, 16]:
            print(f"\n>>> {model_name}  BS={bs}  output_len={args.output_len}")
            try:
                num_generated, total_time_ms, decode_time_ms = run_transformers(
                    model, tokenizer, input_ids, args.output_len, bs
                )
                decode_per_token_ms = decode_time_ms / max(num_generated - 1, 1)
                total_per_token_ms = total_time_ms / num_generated if num_generated > 0 else float('inf')
                line = f"{model_name}  BS={bs}  gen_len={num_generated}  total={total_time_ms:.1f}ms  decode_per_token={decode_per_token_ms:.3f}ms  total_per_token={total_per_token_ms:.3f}ms"
                print(line)
                results.append(line)
            except torch.cuda.OutOfMemoryError:
                line = f"{model_name}  BS={bs}  OOM"
                print(line)
                results.append(line)
                torch.cuda.empty_cache()

        # Free model before loading next
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for line in results:
        print(line)

    with open("bench_transformers_results.txt", "w") as f:
        for line in results:
            f.write(line + "\n")
    print("\nSaved to bench_transformers_results.txt")


if __name__ == "__main__":
    main()
