#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-$ROOT}"

MODEL_PATH_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH_ARG="--model-path $2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--model-path <PATH>]"
      exit 1
      ;;
  esac
done

echo "MIRAGE_HOME=${MIRAGE_HOME}"
echo "Running Torch baseline..."
python "$ROOT/demo/qwen3/demo.py" --save-tokens $MODEL_PATH_ARG

echo "Running MPK..."
python "$ROOT/demo/qwen3/demo.py" --use-mirage --save-tokens $MODEL_PATH_ARG

echo "Comparing outputs..."
pytest -q "$ROOT/tests/ci-tests/test_inference_output.py"
