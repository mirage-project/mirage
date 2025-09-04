#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_linear.sh [experiment_name]
# Example: ./run_linear.sh linear_p2_64
# If not provided, defaults to "linear_p2_64".

EXP_NAME="${1:-linear_p2_64}"

PROJECT_ROOT="/home/jianan/mirage"
TEST_DIR="${PROJECT_ROOT}/tests_cu"
OUT_BIN="test_${EXP_NAME}"
OUT_PATH="${TEST_DIR}/${OUT_BIN}"
REPORT_DIR="${PROJECT_ROOT}/report"
REPORT_PATH="${REPORT_DIR}/${EXP_NAME}.ncu-rep"
ANALYZE_SCRIPT="${PROJECT_ROOT}/analyze.sh"

mkdir -p "${REPORT_DIR}"

echo "[Build] make -C ${TEST_DIR} test_linear OUT=${OUT_BIN}"
make -C "${TEST_DIR}" test_linear "OUT=${OUT_BIN}"

echo "[Run] ${OUT_PATH}"
"${OUT_PATH}"

echo "[Analyze] ${ANALYZE_SCRIPT} ${REPORT_PATH} linear_kernel_launcher ${OUT_PATH}"
"${ANALYZE_SCRIPT}" "${REPORT_PATH}" linear_kernel_launcher "${OUT_PATH}"

echo "Done. Report: ${REPORT_PATH}"


