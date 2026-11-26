"""EOS regression against frozen, unmodified HF generate() continuations.

The JSON reference is produced by the native HF output-semantics audit and
records checkpoint hashes, GPU UUID, prompt IDs and hf_default_greedy tokens.
Only cases that naturally reached EOS before the length limit are selected.
This tests generation stopping, not full numerical parity with native HF.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from mirage.mpk.models.qwen3_5.model import Qwen35Model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
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
        raise ValueError("Expected a same-GPU, pinned native HF reference")
    for name, expected in reference["checkpoint_sha256"].items():
        with (args.model / name).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError("Reference/checkpoint mismatch: " + name)
    eos_id = reference["loaded_generation_config"]["eos_token_id"]
    cases = [c for c in reference["cases"]
             if 0 < len(c["hf_default_greedy"]) < c["length"]
             and c["hf_default_greedy"][-1] == eos_id]
    if len(cases) < 2:
        raise ValueError("Reference must contain at least two natural early-EOS cases")
    report = dict(passed=False, gpu_uuid=uuid,
                  weight_policy="hf_bf16",
                  reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  checkpoint_eos_id=eos_id, cases=[])
    model = None
    try:
        model = Qwen35Model.from_pretrained(args.model)
        if model.config.eos_token_id != eos_id:
            raise ValueError("HF and MPK checkpoint EOS disagree")
        model.compile(args.output_dir / "kernel")
        for case in cases:
            prompt, expected = case["prompt_ids"], case["hf_default_greedy"]
            limit = case["length"]
            record = dict(name=case["name"], expected=expected, checks={})
            report["cases"].append(record)

            def check(name, condition):
                record["checks"][name] = bool(condition)
                if not condition:
                    raise AssertionError(f"{case['name']}: {name}")

            model.reset()
            default = model.generate(prompt, limit)
            record["default_generated"] = default
            check("default_matches_native_hf", default == expected)
            check("eos_not_fed_back", model.position == len(prompt) + len(expected) - 1)

            model.reset()
            check("explicit_checkpoint_eos_matches_default",
                  model.generate(prompt, limit, eos_token_ids=[eos_id]) == default)

            model.reset()
            disabled = model.generate(prompt, limit, eos_token_ids=[])
            record["disabled_generated"] = disabled
            check("empty_override_runs_full_length", len(disabled) == limit
                  and disabled[:len(expected)] == expected
                  and model.position == len(prompt) + limit - 1)

            model.reset()
            custom = model.generate(prompt, limit, eos_token_ids=default[0])
            check("integer_override_stops_on_first_token",
                  custom == default[:1] and model.position == len(prompt))

            model.reset()
            unused = next(i for i in range(model.config.vocab_size)
                          if i not in disabled and i != eos_id)
            check("override_replaces_checkpoint_eos",
                  model.generate(prompt, limit, eos_token_ids=[unused]) == disabled
                  and model.config.eos_token_id == eos_id)

            model.reset()
            check("length_limit_before_eos", model.generate(prompt, 1) == default[:1])

            model.reset()
            check("default_unchanged_after_overrides", model.generate(prompt, limit) == default)
            print(case["name"], "default_tokens=", len(default),
                  "disabled_tokens=", len(disabled), "all EOS checks passed", flush=True)
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if model is not None:
            model.close()
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
