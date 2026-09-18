"""Print Torch vs MPK latency comparison (informational, never fails)."""

import json
import os
import sys

DEFAULT_OUTPUT_DIR = os.path.join("outputs", "qwen3")
TORCH_OUTPUT = os.path.join(DEFAULT_OUTPUT_DIR, "torch_output.json")
MPK_OUTPUT = os.path.join(DEFAULT_OUTPUT_DIR, "mpk_output.json")


def _load_meta(path: str):
    if not os.path.exists(path):
        print(f"Missing output file: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def main():
    torch_meta = _load_meta(TORCH_OUTPUT)
    mpk_meta = _load_meta(MPK_OUTPUT)
    if torch_meta is None or mpk_meta is None:
        return

    torch_lat = torch_meta.get("latency_ms_per_token")
    mpk_lat = mpk_meta.get("latency_ms_per_token")
    torch_len = torch_meta.get("generate_length", "?")
    mpk_len = mpk_meta.get("generate_length", "?")

    if torch_lat is None or mpk_lat is None:
        print("latency_ms_per_token missing in output JSON, skipping comparison")
        return

    speedup = torch_lat / mpk_lat if mpk_lat > 0 else float("inf")

    print("")
    print("==================== Performance Comparison ====================")
    print(f"  Torch:  {torch_lat:.3f} ms/token  (generated {torch_len} tokens)")
    print(f"  MPK:    {mpk_lat:.3f} ms/token  (generated {mpk_len} tokens)")
    print(f"  Speedup: {speedup:.2f}x")
    print("===============================================================")


if __name__ == "__main__":
    sys.exit(main())
