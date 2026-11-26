"""Whole-prompt eager CUDA prefill -> MPK decode vs independent native HF.

Prefill logits, last-token layer outputs and every valid cache element must be
exact. Decode retains the original 2% RMS / 5% peak criteria and exact greedy
tokens; strict equality is separately reported. Frozen twelve-case prompts,
prompt logits and native generate outputs are read-only inputs.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import transformers
from transformers import Qwen3_5ForConditionalGeneration
import transformers.models.qwen3_5.modeling_qwen3_5 as hf_module

from mirage.mpk.models.qwen3_5.model import Qwen35Model
from test_model_gpu import accepted, metrics


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--prompt-logits", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="eager")
    p.add_argument("--backend-reference", type=Path,
                   help="Original live-generation report, required for SDPA token checks")
    p.add_argument("--quick", action="store_true", help="Three frozen cases and two boundary lengths only")
    args = p.parse_args()
    if args.attn_implementation == "sdpa" and args.backend_reference is None:
        p.error("SDPA requires --backend-reference; do not replace the eager frozen reference")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reference = json.loads(args.reference.read_text())
    backend_reference = (json.loads(args.backend_reference.read_text())
                         if args.backend_reference else None)
    if backend_reference is not None:
        assert backend_reference["reference_sha256"] == sha256(args.reference)
        assert backend_reference["completed"]
    if transformers.__version__ != "5.14.1" or str(torch.__version__) != reference["torch"]:
        raise ValueError("Use the pinned HF/PyTorch reference environment")
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose one allocated GPU")
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    if uuid != reference["gpu_uuid"] or uuid.lower().removeprefix("gpu-") != os.environ[
            "QWEN35_EXPECTED_GPU_UUID"].lower().removeprefix("gpu-"):
        raise ValueError("Use the same GPU as the frozen reference")
    if sha256(inspect.getfile(hf_module)) != reference["source_sha256"]:
        raise ValueError("Native HF source changed")
    for name, expected in reference["checkpoint_sha256"].items():
        if sha256(args.model / name) != expected:
            raise ValueError("Checkpoint changed: " + name)
    frozen_logits = torch.load(args.prompt_logits, weights_only=True, map_location="cpu")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    repo = Path(__file__).resolve().parents[2]
    source_paths = list((repo / "python/mirage/mpk/models/qwen3_5").glob("*.py")) + [Path(__file__).resolve()]
    report = dict(passed=False, quick=args.quick, gpu_uuid=uuid, torch=str(torch.__version__),
        transformers=transformers.__version__, prefill_backend="torch_" + args.attn_implementation + "_cuda",
        decode_backend="mpk", attn_implementation=args.attn_implementation,
        backend_reference_sha256=sha256(args.backend_reference) if args.backend_reference else None,
        reference_sha256=sha256(args.reference), prompt_logits_sha256=sha256(args.prompt_logits),
        source_sha256={str(f.relative_to(repo)): sha256(f) for f in source_paths},
        thresholds=dict(logits_relative_rms=.02, logits_reference_peak=.05, logits_absolute_floor=1e-5,
                        prefill_exact=True, greedy_tokens_exact=True), checks=[], generation=[], boundaries=[])
    def save():
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    def check(name, actual, expected, *, exact=True, numeric_gate=False):
        a, e = actual.detach().cpu().clone(), expected.detach().cpu().clone()
        m = metrics(a, e)
        equal = torch.equal(a, e)
        passed = equal if exact else accepted(m) if numeric_gate else True
        row = dict(name=name, exact=equal, passed=passed, actual_sha256=digest(a), expected_sha256=digest(e), **m)
        report["checks"].append(row)
        if not passed:
            torch.save(dict(name=name, actual=a, expected=e), args.output_dir / "failure.pt")
            save()
            raise AssertionError(f"{name}: {row}")
    mpk = None
    handles = []
    try:
        mpk = Qwen35Model.from_pretrained(args.model, attn_implementation=args.attn_implementation)
        mpk.compile(args.output_dir / "kernel")
        hf = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation, local_files_only=True, device_map="cuda").eval()
        last_hidden = {}
        for i, layer in enumerate(hf.model.language_model.layers):
            def capture(module, inputs, output, index=i):
                last_hidden[index] = output[:, -1].detach().clone()
            handles.append(layer.register_forward_hook(capture))
        def hf_step(ids, cache):
            out = hf(input_ids=torch.tensor([ids], device="cuda"), past_key_values=cache,
                     use_cache=True, logits_to_keep=1)
            return out.logits[:, -1], out.past_key_values
        def compare(name, actual, expected, cache, *, prefill):
            check(name + ".logits", actual, expected, exact=prefill, numeric_gate=True)
            for i, layer in enumerate(mpk.layers):
                check(f"{name}.layer{i}", layer.tensors["decoder_output"], last_hidden[i], exact=prefill)
                hf_cache = cache.layers[i]
                if i in mpk.config.gdn_layers:
                    pairs = (("state", hf_cache.recurrent_states[0][0]),
                             ("conv_state", hf_cache.conv_states[0][0, :, -3:]))
                else:
                    pairs = (("key_cache", hf_cache.keys[0].transpose(0, 1)),
                             ("value_cache", hf_cache.values[0].transpose(0, 1)))
                for field, value in pairs:
                    actual_state = layer.tensors[field]
                    if field in ("key_cache", "value_cache"):
                        actual_state = actual_state[:mpk.position]
                    check(f"{name}.layer{i}.{field}", actual_state, value, exact=prefill)
            next_token = int(mpk.tensors["next_token"].item())
            if next_token != int(actual.argmax()) or next_token != int(expected.argmax()):
                raise AssertionError(f"{name}: greedy token differs")
        cases = [c for c in reference["cases"] if not args.quick or c["name"] in
                 ("explanation", "chinese", "chat_multiturn")]
        for case in cases:
            name, prompt = case["name"], case["prompt_ids"]
            expected_tokens = (case["hf_default_greedy"] if args.attn_implementation == "eager" else
                next(c["hf_generated"] for c in backend_reference["backends"]["default"]["cases"]
                     if c["name"] == name))
            mpk.reset()
            expected, cache = hf_step(prompt, None)
            if args.attn_implementation == "eager":
                check(name + ".frozen_hf_prompt", expected, frozen_logits[name + ".block"])
            compare(name + ".prefill", mpk.prefill(prompt), expected, cache, prefill=True)
            for j, token in enumerate(expected_tokens[:-1]):
                expected, cache = hf_step([token], cache)
                compare(f"{name}.decode{j}", mpk.step(token), expected, cache, prefill=False)
            mpk.reset()
            generated = mpk.generate(prompt, case["length"])
            row = dict(name=name, generated=generated, expected=expected_tokens,
                       exact=generated == expected_tokens,
                       position=mpk.position, expected_position=len(prompt) + len(generated) - 1)
            report["generation"].append(row)
            if not row["exact"] or row["position"] != row["expected_position"]:
                raise AssertionError(f"{name}: independent generation or EOS position differs")
            print(name, "prefill/cache exact; independent tokens exact", flush=True)
            save()
        lengths = [63, 65] if args.quick else [1, 2, 63, 64, 65, 127, 128, 129, 1023, 1024]
        generator = torch.Generator().manual_seed(351115)
        for length in lengths:
            ids = torch.randint(0, mpk.config.vocab_size, (length,), generator=generator).tolist()
            mpk.reset()
            expected, cache = hf_step(ids, None)
            compare(f"length{length}.prefill", mpk.prefill(ids), expected, cache, prefill=True)
            # Decode into the last available slot, then test capacity rejection.
            for j in range(min(3, 1024 - length)):
                token = int(expected.argmax())
                expected, cache = hf_step([token], cache)
                compare(f"length{length}.decode{j}", mpk.step(token), expected, cache, prefill=False)
            report["boundaries"].append(dict(length=length, passed=True))
            if mpk.position == 1024:
                before = state_hashes(mpk)
                try:
                    mpk.prefill([0, 1])
                except ValueError:
                    if before != state_hashes(mpk):
                        raise AssertionError("Overflow changed cached state")
                else:
                    raise AssertionError("Overflow accepted")
            print("length", length, "passed", flush=True)
            save()
        plans = [[1, 1, 2, 65], [63, 2, 1, 64]] if args.quick else [
            [1, 1, 2, 65], [63, 2, 1, 64], [65, 63, 1], [1018, 4, 2]]
        report["cached_chunks"] = []
        for plan in plans:
            ids = torch.randint(0, mpk.config.vocab_size, (sum(plan),), generator=generator).tolist()
            mpk.reset()
            cache, offset = None, 0
            for length in plan:
                chunk = ids[offset:offset + length]
                expected, cache = hf_step(chunk, cache)
                compare(f"chunks{plan}.offset{offset}", mpk.prefill(chunk), expected, cache, prefill=True)
                offset += length
            report["cached_chunks"].append(dict(plan=plan, passed=True))
            print("chunks", plan, "passed", flush=True)
            save()
        # Invalid chunks and a mid-computation failure must not partially commit
        # any layer's cache, position, or visible last-token output.
        prompt = cases[0]["prompt_ids"]
        mpk.reset()
        mpk.prefill(prompt)
        before = state_hashes(mpk)
        with patch("mirage.mpk.models.qwen3_5.prefill._attention", side_effect=RuntimeError("injected prefill error")):
            try:
                mpk.prefill([7, 8])
            except RuntimeError as exc:
                if "injected" not in str(exc):
                    raise
            else:
                raise AssertionError("Failure injection did not execute")
        if state_hashes(mpk) != before:
            raise AssertionError("Failed prefill changed state")
        for tokens in ([], [1, -1], [1, True], [1, mpk.config.vocab_size], [0] * 1025):
            try:
                mpk.prefill(tokens)
            except ValueError:
                if state_hashes(mpk) != before:
                    raise AssertionError("Invalid input changed state")
            else:
                raise AssertionError("Invalid input accepted")
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            try:
                mpk.prefill([7, 8])
            except ValueError as exc:
                if "tf32" not in str(exc):
                    raise
            else:
                raise AssertionError("TF32 was silently used for FP32 GDN")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False
        if state_hashes(mpk) != before:
            raise AssertionError("TF32 rejection changed state")
        mpk.reset()
        mpk.prefill([0, 1, 2])
        mpk.reset()
        mpk.prefill(prompt)
        if state_hashes(mpk) != before:
            raise AssertionError("A/B/A reset replay differs")
        report.update(atomic_rejection=True, failed_computation_atomic=True, reset_replay_exact=True,
                      strict_exact=all(c["exact"] for c in report["checks"]), passed=True)
    finally:
        for handle in handles:
            handle.remove()
        if mpk is not None:
            mpk.close()
        save()


def state_hashes(model):
    result = dict(position=model.position)
    for name in ("logits", "next_token", "normalized", "hidden"):
        result[name] = digest(model.tensors[name])
    for i, layer in enumerate(model.layers):
        result[f"{i}.position"] = layer.mixer.position
        for name in ("state", "conv_state", "key_cache", "value_cache", "decoder_output", "position"):
            if name in layer.tensors:
                result[f"{i}.{name}"] = digest(layer.tensors[name])
    return result


if __name__ == "__main__":
    main()
