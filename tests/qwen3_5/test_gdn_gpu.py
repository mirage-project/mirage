"""MPK graph vs saved HF CPU module outputs; run only on an allocated idle GPU.

Thresholds are fixed before running: normalized RMS error <= 1% and maximum
absolute error <= 3% of reference peak + 1e-5 for every intermediate/state.
Reset/replay must be bitwise identical, including FP32 recurrent state.
This is single-layer correctness validation, not full-model acceptance.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from mirage.mpk.models.qwen3_5.config import Qwen35TextConfig
from mirage.mpk.models.qwen3_5.gdn import Qwen35GDNProbe


def compare(actual, expected):
    a, e = actual.detach().cpu().float(), expected.float()
    if a.shape != e.shape or not torch.isfinite(a).all():
        raise AssertionError(f"Invalid result shape/finiteness: {a.shape}, {e.shape}")
    error = (a - e).abs()
    peak = e.abs().max().item()
    relative_rms = (torch.linalg.vector_norm(error) /
                    torch.linalg.vector_norm(e).clamp_min(1e-8)).item()
    maximum = error.max().item()
    if relative_rms > 0.01 or maximum > 0.03 * peak + 1e-5:
        raise AssertionError(f"relative RMS={relative_rms:.6g}, max abs={maximum:.6g}, peak={peak:.6g}")
    return {"relative_rms": relative_rms, "max_abs": maximum, "reference_peak": peak}


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
              "oracle_transformers": fixture["transformers"], "cases": [], "passed": False}
    probe = None
    try:
        probe = Qwen35GDNProbe(Qwen35TextConfig.from_dict(fixture["config"]), fixture["weights"])
        probe.compile(args.output_dir / "kernel")
        for case in fixture["cases"]:
            probe.reset()
            if torch.count_nonzero(probe.tensors["state"]) or torch.count_nonzero(probe.tensors["conv_state"]):
                raise AssertionError("Reset left state behind")
            steps = []
            for i, (token, expected) in enumerate(zip(case["inputs"], case["steps"])):
                probe.step(token)
                metrics = {}
                for name, reference in expected.items():
                    try:
                        metrics[name] = compare(probe.tensors[name], reference)
                    except AssertionError as exc:
                        report["failure"] = {"case": case["name"], "step": i,
                                             "tensor": name, "preceding_metrics": metrics}
                        torch.save({"actual": {n: probe.tensors[n].cpu().clone()
                                               for n in expected}, "expected": expected},
                                   args.output_dir / "failure.pt")
                        raise AssertionError(f"{case['name']} step={i} tensor={name}: {exc}") from exc
                steps.append(metrics)
            final = {n: probe.tensors[n].cpu().clone() for n in ("output", "state", "conv_state")}
            probe.reset()
            for token in case["inputs"]:
                probe.step(token)
            for name, expected in final.items():
                torch.testing.assert_close(probe.tensors[name].cpu(), expected, rtol=0, atol=0)
            report["cases"].append({"name": case["name"], "steps": steps, "reset_replay_exact": True})
            print(f"PASS {case['name']}: {len(steps)} steps, every intermediate/state, exact reset/replay", flush=True)
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
