# GLM-5.3-Flash on SGLang, TP4, 8 requests, eager (no CUDA graph), the bf16 latent cache (kv_cache_dtype bfloat16; "auto" is FP8 for DSA on SM100): a few decode steps, for the static-DSA
# check (DSA_STATIC_LAYERS: dsa_static_hook compares the static layer with SGLang's at each decode step).
# python sgl_check.py [steps] [prompt words] [max_total_tokens | 0] [spec | off] [requests]: the prompts grow with the words (longer contexts; the indexer selects 2048 of the
# positions once a context passes 4 x 512 + 3).
import sys

import sglang as sgl

MODEL = "/raid/catalyst/models/GLM-5.3-Flash"

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    words = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    extra = {"max_total_tokens": int(sys.argv[3])} if len(sys.argv) > 3 and sys.argv[3] != "0" else {}   # static MoE: 65536
    if len(sys.argv) > 4 and sys.argv[4] != "off":   # MTP (sgl_spec_bench.py's --spec, e.g. 5/1/6): the decode steps are
        steps, topk, draft = (int(x) for x in sys.argv[4].split("/"))   # verify steps of bs x draft tokens rows
        extra.update(speculative_algorithm="EAGLE", speculative_num_steps=steps, speculative_eagle_topk=topk,
                     speculative_num_draft_tokens=draft, enable_linear_replayssm_spec=True)   # KDA: the ReplaySSM rings
    reqs = int(sys.argv[5]) if len(sys.argv) > 5 else 8   # requests (MTP: 1 = one request's 6 rows per verify step)
    filler = " ".join(f"item{j}" for j in range(words))
    prompts = [f"Request {i}: {filler[:len(filler) * (i + 1) // 8]} write a long story about the history of computing, "
               f"chapter {i}." for i in range(reqs)]
    llm = sgl.Engine(model_path=MODEL, tp_size=4, max_running_requests=reqs, context_length=8192, mem_fraction_static=0.6,
                     dsa_prefill_backend="trtllm", dsa_decode_backend="trtllm", kv_cache_dtype="bfloat16",
                     moe_runner_backend="flashinfer_trtllm", disable_radix_cache=True, disable_cuda_graph=True,
                     log_level="warning", **extra)
    out = llm.generate(prompts, {"max_new_tokens": steps, "ignore_eos": True, "temperature": 0.0})
    for i, o in enumerate(out[:3]):
        print(f"output {i}:", repr(o["text"]), flush=True)
    llm.shutdown()
