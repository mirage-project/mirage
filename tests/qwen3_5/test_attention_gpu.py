"""MPK full attention vs an independently saved HF 5.14.1 CPU oracle.

Before execution: every tensor's relative RMS <= 1%, max absolute error <=
3% of reference peak + 1e-5. These are single-layer, not model-level limits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from mirage.mpk.models.qwen3_5.config import Qwen35TextConfig
from mirage.mpk.models.qwen3_5.attention import Qwen35AttentionProbe
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
    expected_gpu = os.environ.get("QWEN35_EXPECTED_GPU_UUID")
    actual_gpu = str(torch.cuda.get_device_properties(0).uuid)
    if expected_gpu and actual_gpu.lower().removeprefix("gpu-") != expected_gpu.lower().removeprefix("gpu-"):
        raise ValueError(f"Allocated GPU {actual_gpu} differs from requested {expected_gpu}")
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    if fixture["transformers"] != "5.14.1":
        raise ValueError("Unexpected oracle version")
    report = {"gpu": torch.cuda.get_device_name(), "gpu_uuid": actual_gpu,
              "compute_capability": list(torch.cuda.get_device_capability()),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "torch": str(torch.__version__),
              "oracle_transformers": fixture["transformers"], "layer": fixture["layer"],
              "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
              "thresholds": {"relative_rms": 0.01, "max_abs_reference_peak": 0.03, "max_abs_floor": 1e-5},
              "cases": [], "passed": False}
    probe = None
    final_names = ("output", "key_cache", "value_cache")
    def snapshot():
        return {n: probe.tensors[n].cpu().clone() for n in final_names}
    def assert_exact(expected):
        for name, value in expected.items():
            torch.testing.assert_close(probe.tensors[name].cpu(), value, rtol=0, atol=0, equal_nan=True)
    def reset_case(case):
        probe.reset(**case["initial"])
        # Unused slots must never participate in attention, nor be written.
        for name in ("key_cache", "value_cache"):
            probe.tensors[name][probe.position:].fill_(float("nan"))
    try:
        probe = Qwen35AttentionProbe(Qwen35TextConfig.from_dict(fixture["config"]),
                                    fixture["weights"], layer=fixture["layer"])
        probe.compile(args.output_dir / "kernel")
        first_final = None
        for case in fixture["cases"]:
            assert len(case["inputs"]) == len(case["steps"]) > 0
            reset_case(case)
            steps = []
            for i, (token, expected) in enumerate(zip(case["inputs"], case["steps"])):
                probe.step(token)
                metrics = {}
                actual = {name: probe.tensors[name] for name in expected}
                for name in ("key_cache", "value_cache"):
                    actual[name] = actual[name][:probe.position]
                actual["attention_weights"] = actual["attention_weights"][:, :probe.position]
                for name, reference in expected.items():
                    try:
                        metrics[name] = compare(actual[name], reference)
                    except AssertionError as exc:
                        report["failure"] = {"case": case["name"], "step": i,
                                             "tensor": name, "preceding_metrics": metrics}
                        torch.save({"actual": {n: t.cpu().clone() for n, t in actual.items()},
                                    "expected": expected}, args.output_dir / "failure.pt")
                        raise AssertionError(f"{case['name']} step={i} tensor={name}: {exc}") from exc
                for name in ("key_cache", "value_cache"):
                    assert torch.isnan(probe.tensors[name][probe.position:]).all(), "Wrote outside valid KV prefix"
                assert not torch.count_nonzero(probe.tensors["attention_weights"][:, probe.position:]), "Future attention probability"
                steps.append(metrics)
            final = snapshot()
            if first_final is None:
                first_final = final
            if probe.position == 1024:
                try:
                    probe.step(case["inputs"][-1])
                except ValueError as exc:
                    assert "capacity" in str(exc)
                else:
                    raise AssertionError("Accepted a token beyond cache capacity")
                assert probe.position == 1024
                assert_exact(final)
                report["overflow_without_mutation"] = True
            try:
                probe.reset(torch.zeros(1, 2, 256), torch.zeros(1, 2, 256))
            except ValueError:
                assert_exact(final)
            else:
                raise AssertionError("Accepted an FP32 KV prefix")
            reset_case(case)
            for token in case["inputs"]:
                probe.step(token)
            assert_exact(final)
            report["cases"].append({"name": case["name"], "steps": steps,
                                    "reset_replay_exact": True, "future_slots_untouched": True})
            print(f"PASS {case['name']}: {len(steps)} steps, all intermediates/KV prefixes, exact reset/replay", flush=True)
        reset_case(fixture["cases"][0])
        for token in fixture["cases"][0]["inputs"]:
            probe.step(token)
        assert_exact(first_final)
        report["a_b_a_reuse_exact"] = True
        probe.reset()
        assert probe.position == 0
        for name in ("key_cache", "value_cache", "attention_weights"):
            assert not torch.count_nonzero(probe.tensors[name]), "Reset left state behind"
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
