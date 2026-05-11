"""torchrun-launchable script for the TP/EP reference.

Usage (from `/home/muhengl/mirage`):

    torchrun --nproc_per_node=4 \
      tests/dpskv3_reference/runner_distributed.py \
      --model-path /raid/catalyst/models/DeepSeek-V3 \
      --layers 0-3 \
      --tp-size 4 --ep-size 2 \
      --prompt "Hello, world." \
      --max-new-tokens 4 \
      --enable-mtp \
      --dump-dir outputs/dpskv3_ref_<tag>

`--layers` accepts a comma-separated list (e.g., "0,1,2") or a range
("0-19"). MTP layer index = num_hidden_layers (typically 61) is
included automatically when `--enable-mtp` is passed.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Ensure repo root on sys.path so `tests.dpskv3_reference.*` is importable
# when launched via torchrun (which doesn't always set PYTHONPATH).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from tests.dpskv3_reference.runner import run_reference  # noqa: E402


def _parse_layers(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x) for x in s.split(",")]
    return [int(s)]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--prompt", default="Give me a short introduction to large language model.")
    p.add_argument("--prompt-length", type=int, default=0,
                   help="If > 0, use synthetic deterministic prompt "
                        "(arange(N) % 4096 + 1024) — matches MPK demo's "
                        "--prompt-length mode. Else tokenize --prompt.")
    p.add_argument("--layers", default="0", help='comma list or "a-b" range')
    p.add_argument("--enable-mtp", action="store_true")
    p.add_argument("--spec-length", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--max-num-batched-tokens", type=int, default=128)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--ep-size", type=int, default=1)
    p.add_argument("--dump-dir", default=None)
    p.add_argument("--skip-weight-load", action="store_true",
                   help="random init for unit testing")
    p.add_argument("--fp8-faithful", action="store_true",
                   help="route FP8 linears through a quantize-then-matmul "
                        "PyTorch simulation so numerics match MPK's FP8 "
                        "dense/group GEMM (otherwise weights are "
                        "dequantized to BF16 once at load and the matmul "
                        "is run in BF16, which diverges from MPK by FP8 "
                        "activation-quantization noise).")
    args = p.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    layers = _parse_layers(args.layers)

    result = run_reference(
        model_path=args.model_path,
        prompt=args.prompt,
        prompt_length=args.prompt_length,
        layers=layers,
        enable_mtp=args.enable_mtp,
        spec_length=args.spec_length,
        max_new_tokens=args.max_new_tokens,
        max_num_batched_tokens=args.max_num_batched_tokens,
        dump_dir=args.dump_dir,
        device="cuda",
        dtype=torch.bfloat16,
        skip_weight_load=args.skip_weight_load,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
        rank=rank,
        fp8_faithful=args.fp8_faithful,
    )
    if rank == 0:
        print(f"DUMP_DIR={result.dump_dir}")
        print(f"TOKENS={result.token_ids}")
        print(f"ELAPSED_S={result.elapsed_s:.2f}")
        if result.prefill_ms is not None:
            print(f"PREFILL_MS={result.prefill_ms:.1f}")
        if result.decode_tpot_ms is not None:
            print(f"DECODE_TPOT_MS={result.decode_tpot_ms:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
