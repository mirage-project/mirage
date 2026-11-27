"""Compare default MPK loading with the frozen native HF BF16 audit.

Require exact logits on all saved token-at-a-time histories, exact independent
streaming continuations, and correct EOS truncation. Also require the new
default whole-prompt prefill to match frozen native HF generate() tokens.
This runner works in the MPK environment without the HF Qwen3.5 module.
The input files are read-only and retain the original audit's twelve cases.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from mirage.mpk.models.qwen3_5.model import Qwen35Model
from test_model_gpu import metrics


def digest(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def first_difference(actual, expected):
    return next((i for i in range(max(len(actual), len(expected)))
                 if i >= len(actual) or i >= len(expected) or actual[i] != expected[i]), None)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prompt-logits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reference = json.loads(args.reference.read_text())
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose one allocated GPU")
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    expected_uuid = os.environ.get("QWEN35_EXPECTED_GPU_UUID", uuid)
    if uuid.lower().removeprefix("gpu-") != expected_uuid.lower().removeprefix("gpu-"):
        raise ValueError("Unexpected GPU UUID")
    if (reference["gpu_uuid"] != uuid or reference["transformers"] != "5.14.1"
            or reference["torch"] != str(torch.__version__)):
        raise ValueError("Expected same-GPU pinned native HF reference")
    for name, expected in reference["checkpoint_sha256"].items():
        with (args.model / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Reference/checkpoint mismatch: " + name)
    expected_logits = torch.load(args.prompt_logits, map_location="cpu", weights_only=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(passed=False, weight_policy="hf_bf16", default_prefill_mode="block",
                  checked_prefill_modes=["stream", "block"], gpu_uuid=uuid,
                  reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  prompt_logits_sha256=hashlib.sha256(args.prompt_logits.read_bytes()).hexdigest(),
                  cases=[])
    model = None
    try:
        model = Qwen35Model.from_pretrained(args.model)
        eos = reference["loaded_generation_config"]["eos_token_id"]
        if model.config.eos_token_id != eos:
            raise ValueError("HF/MPK checkpoint EOS mismatch")
        model.compile(args.output_dir / "kernel")
        for case in reference["cases"]:
            model.reset()
            hashes = []
            for step, token in enumerate(case["stream_inputs"]):
                logits = model.step(token)
                hashes.append(digest(logits))
                if step + 1 == len(case["prompt_ids"]):
                    prompt_logits = logits.detach().cpu().clone()
            model.reset()
            fixed = model.generate(case["prompt_ids"], case["length"], eos_token_ids=(), prefill_mode="stream")
            model.reset()
            stream_default = model.generate(case["prompt_ids"], case["length"], prefill_mode="stream")
            model.reset()
            block = model.generate(case["prompt_ids"], case["length"])
            stream_eos = case["hf_stream"]
            if eos in stream_eos:
                stream_eos = stream_eos[:stream_eos.index(eos) + 1]
            row = dict(name=case["name"], fixed_generated=fixed, default_generated=block,
                       stream_default_generated=stream_default,
                       block_generated=block,
                       block_tokens_exact=block == case["hf_default_greedy"],
                       stream_logits_total=len(hashes),
                       stream_logits_exact=sum(a == b for a, b in zip(hashes, case["stream_logits_sha256"])),
                       stream_tokens_exact=fixed == case["hf_stream"],
                       default_eos_matches_stream=stream_default == stream_eos,
                       stream_vs_native_generate_tokens_exact=stream_default == case["hf_default_greedy"],
                       native_generate_tokens_exact=block == case["hf_default_greedy"],
                       first_native_generate_difference=first_difference(block, case["hf_default_greedy"]),
                       native_generate_expected=case["hf_default_greedy"],
                       prompt_logits={mode: metrics(prompt_logits, expected_logits[case["name"] + "." + mode])
                                      for mode in ("stream", "block")})
            row["passed"] = (hashes == case["stream_logits_sha256"]
                             and row["stream_tokens_exact"] and row["default_eos_matches_stream"]
                             and row["block_tokens_exact"])
            report["cases"].append(row)
            print(case["name"], "exact logits", row["stream_logits_exact"], "/", len(hashes),
                  "stream", row["stream_tokens_exact"], "block generate", row["block_tokens_exact"], flush=True)
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        report["passed"] = all(c["passed"] for c in report["cases"])
        report["all_native_generate_exact"] = all(c["native_generate_tokens_exact"] for c in report["cases"])
        report["all_block_generate_exact"] = all(c["block_tokens_exact"] for c in report["cases"])
        if not report["passed"]:
            raise AssertionError("Native BF16 streaming/prefill/EOS regression failed; see report.json")
    finally:
        if model is not None:
            model.close()
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
