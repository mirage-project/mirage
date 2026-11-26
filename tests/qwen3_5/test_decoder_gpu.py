"""Complete single-layer decoder check, with the same predeclared 1%/3% limits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch

from mirage.mpk.models.qwen3_5.config import Qwen35TextConfig
from mirage.mpk.models.qwen3_5.decoder import Qwen35DecoderProbe
from test_gdn_gpu import compare


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one allocated GPU")
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    expected_uuid = os.environ.get("QWEN35_EXPECTED_GPU_UUID", uuid)
    if uuid.lower().removeprefix("gpu-") != expected_uuid.lower().removeprefix("gpu-"):
        raise ValueError("Unexpected GPU UUID")
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    if fixture["transformers"] != "5.14.1" or not fixture.get("decoder"):
        raise ValueError("Expected pinned complete-decoder fixture")
    report = {"passed": False, "gpu": torch.cuda.get_device_name(), "gpu_uuid": uuid,
              "compute_capability": list(torch.cuda.get_device_capability()),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "layer": fixture["layer"],
              "kind": fixture["kind"], "torch": str(torch.__version__),
              "oracle_device": fixture.get("oracle_device", "cpu"),
              "oracle_gpu": fixture.get("oracle_gpu"),
              "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
              "thresholds": {"relative_rms": 0.01, "max_abs_reference_peak": 0.03, "max_abs_floor": 1e-5},
              "cases": []}
    probe = None
    try:
        probe = Qwen35DecoderProbe(Qwen35TextConfig.from_dict(fixture["config"]), fixture["weights"], layer=fixture["layer"])
        probe.compile(args.output_dir / "kernel")
        states = ("state", "conv_state") if fixture["kind"] == "gdn" else ("key_cache", "value_cache")
        first_final = None
        def snapshot():
            return {n: probe.tensors[n].cpu().clone() for n in ("decoder_output", *states)}
        def assert_exact(final):
            for name, value in final.items():
                torch.testing.assert_close(probe.tensors[name].cpu(), value, rtol=0, atol=0)
        for case in fixture["cases"]:
            probe.reset(**case["initial"])
            assert len(case["inputs"]) == len(case["steps"]) > 0
            steps = []
            for i, (token, expected) in enumerate(zip(case["inputs"], case["steps"])):
                probe.step(token)
                actual = {n: probe.tensors[n] for n in expected}
                if fixture["kind"] == "attention":
                    for name in states:
                        actual[name] = actual[name][:probe.position]
                    actual["attention_weights"] = actual["attention_weights"][:, :probe.position]
                metrics = {}
                for name, reference in expected.items():
                    try:
                        metrics[name] = compare(actual[name], reference)
                    except AssertionError as exc:
                        report["failure"] = {"case": case["name"], "step": i, "tensor": name,
                                             "preceding_metrics": metrics}
                        torch.save({"actual": {n: t.cpu().clone() for n, t in actual.items()},
                                    "expected": expected}, args.output_dir / "failure.pt")
                        raise AssertionError(f"{case['name']} step={i} tensor={name}: {exc}") from exc
                steps.append(metrics)
            final = snapshot()
            if first_final is None:
                first_final = final
            if probe.position == 1024:
                try:
                    probe.step(case["inputs"][-1])
                except ValueError:
                    assert_exact(final)
                else:
                    raise AssertionError("Accepted a token beyond capacity")
                report["overflow_without_mutation"] = True
            probe.reset(**case["initial"])
            for token in case["inputs"]:
                probe.step(token)
            assert_exact(final)
            report["cases"].append({"name": case["name"], "steps": steps, "reset_replay_exact": True})
            print(f"PASS {fixture['kind']} decoder {case['name']}: {len(steps)} steps, exact reset/replay", flush=True)
        probe.reset(**fixture["cases"][0]["initial"])
        for token in fixture["cases"][0]["inputs"]:
            probe.step(token)
        assert_exact(first_final)
        report["a_b_a_reuse_exact"] = True
        probe.reset()
        for name in states:
            assert not torch.count_nonzero(probe.tensors[name]), "Reset left model state behind"
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
        if probe is not None:
            probe.close()


if __name__ == "__main__":
    main()
