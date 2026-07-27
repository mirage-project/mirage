#!/usr/bin/env python3
"""Short decode run for Nsight Compute, at the binding baseline engine config.

Boot + warmup happen with the profiler OFF (--profile-from-start off); cudaProfilerStart is
raised only inside a steady decode window, so ncu sees decode kernels and not the thousands of
boot / cudagraph-capture / autotune launches.
"""
import argparse
import os
import sys
import time

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH_DIR)
import bench_vllm as BV  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=40)
    ap.add_argument("--start-step", type=int, default=12)
    ap.add_argument("--n-steps", type=int, default=4)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="disable CUDA graphs so ncu can serialise/replay kernels; kernel "
                         "SELECTION is static at create_weights() time (vllm-graph.md 3.5) so "
                         "the same kernels at the same shapes still run")
    ap.add_argument("--max-num-batched-tokens", type=int, default=None,
                    help="shrink vLLM's memory-profiling warmup forward; does not affect the "
                         "decode shapes profiled here")
    args = ap.parse_args()

    import torch
    from vllm import LLM, SamplingParams
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    gpu_id = BV.get_gpu_id()
    try:
        BV.preflight_gpu_check(gpu_id)
        print("[ncu] GPU exclusive", flush=True)
    except RuntimeError as e:
        # ncu collects hardware counters with the target kernel serialised; an IDLE co-tenant
        # context (0 % util, no running kernels) does not invalidate SOL/occupancy/launch config
        # the way it would invalidate a timing baseline. Record it loudly instead of pretending.
        print(f"[ncu] CO-TENANT PRESENT, proceeding for COUNTER collection only: {e}", flush=True)
        print(f"[ncu] cotenants={BV.nvidia_smi_compute_pids(gpu_id)}", flush=True)
    kw = {}
    if args.max_num_batched_tokens:
        kw["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(model=BV.MODEL_ID_DEFAULT, revision=BV.REVISION_DEFAULT, dtype="auto",
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.input_len + args.output_len,
              enforce_eager=args.enforce_eager,
              disable_log_stats=False, language_model_only=False, seed=0, **kw)
    vc = llm.llm_engine.vllm_config
    print(f"[ncu] enforce_eager={vc.model_config.enforce_eager} "
          f"mnbt={vc.scheduler_config.max_num_batched_tokens} "
          f"quant={vc.model_config.quantization}", flush=True)
    print("[ncu] engine up", flush=True)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, top_p=1.0, seed=0, max_tokens=args.output_len,
                        min_tokens=args.output_len, ignore_eos=True)

    # warmup generate, profiler still off
    llm.generate(BV.build_synthetic_prompts(tok, args.batch_size, args.input_len, 11),
                 sampling_params=sp, use_tqdm=False)
    print("[ncu] warmup done", flush=True)

    rt = torch.cuda.cudart()
    st = {"n": 0, "on": False}
    orig = GPUModelRunner.execute_model

    def patched(self, *a, **kw):
        st["n"] += 1
        if st["n"] == args.start_step:
            torch.cuda.synchronize()
            rt.cudaProfilerStart()
            st["on"] = True
            print(f"[ncu] cudaProfilerStart at step {st['n']}", flush=True)
        out = orig(self, *a, **kw)
        if st["on"] and st["n"] >= args.start_step + args.n_steps:
            torch.cuda.synchronize()
            rt.cudaProfilerStop()
            st["on"] = False
            print(f"[ncu] cudaProfilerStop at step {st['n']}", flush=True)
        return out

    GPUModelRunner.execute_model = patched
    t0 = time.time()
    llm.generate(BV.build_synthetic_prompts(tok, args.batch_size, args.input_len, 22),
                 sampling_params=sp, use_tqdm=False)
    print(f"[ncu] profiled generate done in {time.time() - t0:.1f}s, steps={st['n']}", flush=True)


if __name__ == "__main__":
    main()
