#! /usr/bin/env bash
set -euo pipefail

# Setup project paths and environment variables
MIRAGE_HOME=$(realpath "${BASH_SOURCE[0]%/*}/../..")
export MIRAGE_HOME
BUILD_FOLDER="${MIRAGE_HOME}/build"
export BUILD_FOLDER
CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda"}
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
pushd ${MIRAGE_HOME}/src/search/abstract_expr/abstract_subexpr
cargo build --release --target-dir ../../../../build
popd
export LD_LIBRARY_PATH+="${MIRAGE_HOME}/build/release:${LD_LIBRARY_PATH}"
# Temporary file paths 
OUTPUT_DIR="/tmp"
PREFIX="mirage"
RESPONSE_BEFORE="${OUTPUT_DIR}/${PREFIX}_before_response.txt"
RESPONSE_AFTER="${OUTPUT_DIR}/${PREFIX}_after_response.txt"
LATENCY_BEFORE="${OUTPUT_DIR}/${PREFIX}_before_latency.txt"
LATENCY_AFTER="${OUTPUT_DIR}/${PREFIX}_after_latency.txt"

# Get installation status parameter
installation_status=${1:-"before-installation"}
echo "Running Python interface tests (installation status: ${installation_status})"

# Ensure we're in the correct directory
cd "${MIRAGE_HOME}/tests/ci-tests/qwen2.5" || exit 1

run_test() {
  local disable_flag=$1
  local mode=$2
  local status_msg=$3

  echo "Running ${status_msg}..."
  python demo.py $disable_flag --output-dir "$OUTPUT_DIR" --prefix "$PREFIX"
  
  local latency_file="${OUTPUT_DIR}/${PREFIX}_${mode}_latency.txt"
  
  echo "${status_msg} completed, latency: $(cat "$latency_file") ms"
}

if [[ "$installation_status" == "before-installation" ]]; then
  export LD_LIBRARY_PATH="${BUILD_FOLDER}:${LD_LIBRARY_PATH}"
  
  # Check Mirage module availability
  python -c "import mirage; print('Mirage module loaded successfully!'); exit()"
  
  # Run baseline test (with Mirage disabled)
  run_test "--disable-mirage" "before" "baseline test"
  
  unset LD_LIBRARY_PATH

elif [[ "$installation_status" == "after-installation" ]]; then
  # Check Mirage module availability after installation
  python -c "import mirage; print('Mirage module loaded successfully!'); exit()"
  
  # Run optimized test (with Mirage enabled)
  run_test "" "after" "optimized test"
  
  # Compare responses - they should be identical
  if ! diff -q "$RESPONSE_BEFORE" "$RESPONSE_AFTER" > /dev/null; then
    echo "ERROR: Response content differs between before and after installation!"
    exit 1
  fi
  echo "Response validation: PASSED - Responses are identical"
  
  # Compare latency - after should be faster than before
  BEFORE_LATENCY=$(cat "$LATENCY_BEFORE")
  AFTER_LATENCY=$(cat "$LATENCY_AFTER")
  if awk "BEGIN {exit !($AFTER_LATENCY > $BEFORE_LATENCY)}"; then
    echo "ERROR: Performance regression detected!"
    echo "Before latency: $BEFORE_LATENCY ms"
    echo "After latency: $AFTER_LATENCY ms"
    LATENCY_DIFF=$(awk "BEGIN {print $AFTER_LATENCY - $BEFORE_LATENCY}")
    echo "Latency increase: $LATENCY_DIFF ms/token"
    exit 1
  fi
  
  # Calculate performance improvement
  IMPROVEMENT=$(awk "BEGIN {print $BEFORE_LATENCY - $AFTER_LATENCY}")
  PERCENTAGE=$(awk "BEGIN {print ($BEFORE_LATENCY - $AFTER_LATENCY) / $BEFORE_LATENCY * 100}")
  echo "Performance validation: PASSED - Optimized latency ($AFTER_LATENCY ms) is faster than baseline ($BEFORE_LATENCY ms)"
  echo "Performance improvement: $IMPROVEMENT ms/token ($PERCENTAGE% faster)"
  
  # Clean up temporary files
  rm -f "$RESPONSE_BEFORE" "$RESPONSE_AFTER" "$LATENCY_BEFORE" "$LATENCY_AFTER"

else
  echo "Invalid installation status!"
  echo "Usage: $0 {before-installation, after-installation}"
  exit 1
fi

echo "All Python tests completed successfully!" 