#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-$ROOT}"

echo "MIRAGE_HOME=${MIRAGE_HOME}"
echo "Running Torch baseline..."
python "$ROOT/demo/qwen3/demo.py" --save-tokens

echo "Running MPK..."
python "$ROOT/demo/qwen3/demo.py" --use-mirage --save-tokens

echo "Comparing outputs..."
pytest -q "$ROOT/tests/ci-tests/test_inference_output.py"
