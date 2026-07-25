#!/usr/bin/env python3
"""M1-I5 step 8: vLLM smoke test for Qwen/Qwen3.5-35B-A3B-FP8.

Generates ONE pinned prompt (greedy, 64 tokens), confirms the FP8 path is
active (not a silent dequant fallback), records version/config/tokens-per-s,
and compares the output token ids against the HF transformers reference
produced by generate_reference.py (informational match/mismatch, not a gate).
"""
import argparse
import json
import time
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-35B-A3B-FP8")
    ap.add_argument("--revision", default="9d1823d2dee688a6b25e77009dc727688c44936e")
    ap.add_argument("--prompt-id", default="p01-history")
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--reference-json", required=True, help="reference_outputs.json from generate_reference.py")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    import vllm
    print(f"[vllm_smoke] vllm.__version__ = {vllm.__version__}", flush=True)
    from vllm import LLM, SamplingParams

    prompts = [json.loads(l) for l in open(args.prompts_file) if l.strip()]
    row = next(p for p in prompts if p["id"] == args.prompt_id)
    messages = row["messages"]

    t_load0 = time.time()
    llm = LLM(
        model=args.model_id,
        revision=args.revision,
        dtype="auto",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )
    load_elapsed = time.time() - t_load0
    print(f"[vllm_smoke] model load took {load_elapsed:.1f}s", flush=True)

    # Try a couple of attribute paths across vllm versions to report the
    # active quantization method directly rather than relying on log-scraping
    # alone.
    quant_method = None
    for path in [
        lambda: llm.llm_engine.model_config.quantization,
        lambda: llm.llm_engine.vllm_config.model_config.quantization,
    ]:
        try:
            quant_method = path()
            if quant_method is not None:
                break
        except Exception:
            continue
    print(f"[vllm_smoke] detected quantization method: {quant_method!r}", flush=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    t0 = time.time()
    outputs = llm.chat([messages], sampling_params=sampling_params)
    elapsed = time.time() - t0

    out = outputs[0].outputs[0]
    output_ids = list(out.token_ids)
    tokens_per_s = len(output_ids) / elapsed if elapsed > 0 else float("nan")
    print(f"[vllm_smoke] generated {len(output_ids)} tokens in {elapsed:.2f}s -> {tokens_per_s:.2f} tok/s", flush=True)
    print(f"[vllm_smoke] output_ids: {output_ids}", flush=True)
    print(f"[vllm_smoke] output_text: {out.text!r}", flush=True)

    ref = json.load(open(args.reference_json))
    ref_ids = ref["results"][args.prompt_id]["output_ids"]
    match = output_ids == ref_ids
    print(f"[vllm_smoke] HF reference output_ids: {ref_ids}", flush=True)
    print(f"[vllm_smoke] MATCH vs HF reference: {match}", flush=True)
    if not match:
        n = min(len(output_ids), len(ref_ids))
        first_diff = next((i for i in range(n) if output_ids[i] != ref_ids[i]), n)
        print(f"[vllm_smoke] first divergence at generated step {first_diff}", flush=True)

    result = {
        "vllm_version": vllm.__version__,
        "model_id": args.model_id,
        "revision": args.revision,
        "prompt_id": args.prompt_id,
        "quantization_method_detected": quant_method,
        "load_seconds": round(load_elapsed, 1),
        "generate_seconds": round(elapsed, 3),
        "num_generated": len(output_ids),
        "tokens_per_second": round(tokens_per_s, 2),
        "output_ids": output_ids,
        "output_text": out.text,
        "hf_reference_output_ids": ref_ids,
        "matches_hf_reference": match,
        "note": "single-request smoke, NOT the AC-4 throughput benchmark protocol",
    }
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[vllm_smoke] wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
