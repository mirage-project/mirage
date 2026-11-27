"""Full-model acceptance: fixed-input logits and independent greedy generation.

Declared before the first run: logits relative RMS <= 2%, peak error <= 5%
of reference peak + 1e-5, and exact streaming-reference greedy continuations.
Layer/state errors and ordinary HF whole-prompt generation are also reported;
they are diagnostic, not silently substituted for the streaming oracle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from mirage.mpk.models.qwen3_5.model import Qwen35Model
from mirage.mpk.models.qwen3_5.weight_loader import Qwen35WeightLoader


def metrics(actual, expected):
    a, e = actual.detach().cpu().float(), expected.float()
    if a.shape != e.shape or not torch.isfinite(a).all() or not torch.isfinite(e).all():
        raise AssertionError(f"Invalid shape/finiteness: {a.shape}, {e.shape}")
    error = a - e
    return dict(relative_rms=(error.norm() / e.norm().clamp_min(1e-8)).item(),
                max_abs=error.abs().max().item(), reference_peak=e.abs().max().item())


def accepted(m):
    return m["relative_rms"] <= .02 and m["max_abs"] <= .05 * m["reference_peak"] + 1e-5


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cross-device-reference", action="store_true",
                        help="Explicitly compare with a frozen oracle from another GPU; retain all thresholds")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one allocated GPU")
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    expected_uuid = os.environ.get("QWEN35_EXPECTED_GPU_UUID", uuid)
    if uuid.lower().removeprefix("gpu-") != expected_uuid.lower().removeprefix("gpu-"):
        raise ValueError("Unexpected GPU UUID")
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    if (fixture["transformers"] != "5.14.1" or fixture["oracle_device"] != "cuda"
            or (fixture["oracle_uuid"] != uuid and not args.cross_device_reference)
            or not fixture["native_empty_cache"]):
        raise ValueError("Expected same-GPU pinned native full-model oracle")
    for name, expected in fixture["checkpoint_sha256"].items():
        if hashlib.sha256((args.model / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Oracle/checkpoint mismatch: " + name)
    if json.loads((args.model / "config.json").read_text()) != fixture["config"]:
        raise ValueError("Oracle/config mismatch")
    # Old frozen fixtures predate policy metadata and preserve stored dtypes.
    weight_policy = fixture.get("weight_policy", "checkpoint")
    report = dict(passed=False, gpu=torch.cuda.get_device_name(), gpu_uuid=uuid,
                  weight_policy=weight_policy, prefill_mode="stream",
                  oracle_gpu=fixture["oracle_gpu"], oracle_uuid=fixture["oracle_uuid"],
                  same_gpu_reference=fixture["oracle_uuid"] == uuid,
                  compute_capability=list(torch.cuda.get_device_capability()),
                  slurm_job_id=os.environ.get("SLURM_JOB_ID"), torch=str(torch.__version__),
                  oracle_transformers=fixture["transformers"],
                  fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                  thresholds=dict(logits_relative_rms=.02, logits_reference_peak=.05,
                                  logits_absolute_floor=1e-5, streaming_tokens_exact=True), cases=[])
    model = None
    def snapshot():
        state = {"logits": model.tensors["logits"].cpu().clone()}
        for i, layer in enumerate(model.layers):
            names = ("state", "conv_state") if i in model.config.gdn_layers else ("key_cache", "value_cache")
            for n in names:
                state[f"{i}.{n}"] = layer.tensors[n].cpu().clone()
        return state
    def assert_exact(expected):
        for n, t in snapshot().items():
            torch.testing.assert_close(t, expected[n], atol=0, rtol=0)
    try:
        loader = Qwen35WeightLoader(args.model)
        weights = loader.load(weight_policy=weight_policy)
        if fixture.get("parameter_hashes") is not None:
            expected_parameters = fixture["parameter_hashes"]
            if weights.keys() != expected_parameters.keys():
                raise AssertionError("HF/MPK parameter names differ")
            for name, tensor in weights.items():
                actual = dict(dtype=str(tensor.dtype), shape=list(tensor.shape),
                              sha256=hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
                if actual != expected_parameters[name]:
                    raise AssertionError("HF/MPK loaded parameter differs: " + name)
            report["loaded_parameters_exact"] = len(weights)
        model = Qwen35Model(loader.config, weights)
        del weights
        model.compile(args.output_dir / "kernel")
        first_final = None
        for case in fixture["cases"]:
            model.reset()
            steps = []
            first_token_difference = None
            for i, (token, reference, hidden) in enumerate(zip(case["inputs"], case["logits"], case["hidden"])):
                logits = model.step(token)
                m = metrics(logits, reference)
                current = torch.stack([l.tensors["decoder_output"][0].cpu() for l in model.layers])
                layer_metrics = [metrics(a, b) for a, b in zip(current, hidden)]
                next_id = int(model.tensors["next_token"].item())
                if next_id != int(logits.argmax(-1).item()):
                    raise AssertionError("GPU argmax differs from PyTorch's stable argmax")
                top = reference.float().topk(2).values[0]
                steps.append(dict(logits=m, layers=layer_metrics, actual_token=next_id,
                                  reference_token=int(reference.argmax(-1).item()),
                                  reference_margin=float(top[0] - top[1]), accepted=accepted(m)))
                if not accepted(m) and "first_logits_failure" not in report:
                    report["first_logits_failure"] = dict(case=case["name"], step=i)
                    torch.save(dict(actual_logits=logits.cpu(), expected_logits=reference,
                                    actual_hidden=current, expected_hidden=hidden), args.output_dir / "failure.pt")
                if (first_token_difference is None and i >= len(case["prompt_ids"]) - 1
                        and next_id != int(reference.argmax(-1).item())):
                    first_token_difference = i
                    torch.save(dict(step=i, actual_logits=logits.cpu(), expected_logits=reference,
                                    actual_hidden=current, expected_hidden=hidden),
                               args.output_dir / (case["name"] + "-first-token-difference.pt"))
            states = []
            for layer, expected in zip(model.layers, case["states"]):
                entry = {}
                for name, value in expected.items():
                    actual = layer.tensors[name]
                    if name in ("key_cache", "value_cache"):
                        actual = actual[:model.position]
                    entry[name] = metrics(actual, value)
                states.append(entry)
            final = snapshot()
            if first_final is None:
                first_final = final
            # Every case tests exact replay of all 24 state buffers and logits.
            model.reset()
            model.prefill(case["inputs"], mode="stream")
            assert_exact(final)
            model.reset()
            # The frozen oracle generates a fixed number of tokens without EOS
            # stopping. Keep that workload explicit as the public default uses
            # the checkpoint EOS; default stopping is covered by test_eos_gpu.py.
            generated = model.generate(case["prompt_ids"], len(case["generated"]), eos_token_ids=(),
                                       prefill_mode="stream")
            record = dict(name=case["name"], prompt_ids=case["prompt_ids"], steps=steps, states=states,
                          generated=generated, reference_generated=case["generated"],
                          block_generated=case["block_generated"],
                          streaming_tokens_exact=generated == case["generated"],
                          block_tokens_exact=generated == case["block_generated"],
                          hf_stream_block_tokens_exact=case["generated"] == case["block_generated"],
                          reset_replay_exact=True)
            record["first_forced_token_difference"] = first_token_difference
            # Compare identical prompt-boundary logits with ordinary HF prefill.
            model.reset()
            record["block_prefill_logits"] = metrics(model.prefill(case["prompt_ids"], mode="stream"), case["block_prefill_logits"])
            record["passed"] = all(s["accepted"] for s in steps) and record["streaming_tokens_exact"]
            report["cases"].append(record)
            print(f"{case['name']}: logits RMS max={max(s['logits']['relative_rms'] for s in steps):.6g}, "
                  f"stream tokens={record['streaming_tokens_exact']}, block tokens={record['block_tokens_exact']}", flush=True)
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
        # A/B/A reuse, invalid inputs, and full-capacity rejection leave state intact.
        model.reset()
        model.prefill(fixture["cases"][0]["inputs"], mode="stream")
        assert_exact(first_final)
        for invalid in (-1, model.config.vocab_size, True, 1.5):
            try:
                model.step(invalid)
            except ValueError:
                assert_exact(first_final)
            else:
                raise AssertionError("Accepted invalid token")
        previous = model.position
        model.position = 1024
        try:
            model.step(0)
        except ValueError:
            assert_exact(first_final)
        else:
            raise AssertionError("Accepted overflow")
        finally:
            model.position = previous
        model.reset()
        for key, value in snapshot().items():
            if key != "logits" and torch.count_nonzero(value):
                raise AssertionError("Reset left state: " + key)
        report.update(a_b_a_reuse_exact=True, invalid_input_without_mutation=True,
                      overflow_without_mutation=True, reset_all_states_zero=True)
        report["passed"] = all(c["passed"] for c in report["cases"])
        if not report["passed"]:
            raise AssertionError("Full-model comparison failed; see report.json for all cases")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
        if model is not None:
            model.close()


if __name__ == "__main__":
    main()
