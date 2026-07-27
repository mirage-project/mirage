#!/bin/bash
# M3-I10 part D: SGLang feasibility timebox. Install phase only (no GPU touched).
set -uo pipefail
R="$HOME/mpk-qwen35/m3i10-profile"
mkdir -p "$R/logs" "$R/sglang"
exec > "$R/logs/sglang_install.log" 2>&1
set -x
export PATH="$HOME/.local/bin:/usr/local/cuda-12.8/bin:$PATH"
export UV_CACHE_DIR="$HOME/mpk-qwen35/.uv-cache"
export HF_HOME="$HOME/mpk-qwen35/hf"
mkdir -p "$UV_CACHE_DIR"
df -h /raid | tail -2
date -Is

cd "$R/sglang"
python3 -m venv venv-sglang || uv venv venv-sglang
source venv-sglang/bin/activate
python -m pip install -U pip wheel setuptools 2>&1 | tail -3

echo "=== available sglang versions ==="
pip index versions sglang 2>&1 | head -5 || pip install "sglang==" 2>&1 | head -5

echo "=== installing sglang[all] ==="
pip install "sglang[all]" 2>&1 | tail -40
EXIT=$?
echo "=== PIP_EXIT=$EXIT ==="

echo "=== smoke import (ABI check) ==="
python -c "import sglang, torch; print('sglang', sglang.__version__); print('torch', torch.__version__); print('cuda', torch.version.cuda)" 2>&1 | tail -20
echo "=== model support probe ==="
python - <<'PY' 2>&1 | tail -40
import json, traceback
try:
    from sglang.srt.models.registry import ModelRegistry
    keys = sorted(ModelRegistry.models.keys())
    print("n_registered_archs:", len(keys))
    print("qwen-ish:", [k for k in keys if "wen" in k.lower()])
except Exception:
    traceback.print_exc()
PY
echo "=== SGLANG_INSTALL_DONE $(date -Is) ==="
df -h /raid | tail -2
