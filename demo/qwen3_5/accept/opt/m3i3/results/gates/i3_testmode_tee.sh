#!/bin/bash
# GDN recurrent TEST-MODE pipeline gate, captured with tee.
# Exercises the full MPK path (python layer API -> task registration -> C++
# codegen -> nvcc -> runtime dispatch) with the decode split ENABLED
# (GDN_SPLIT=4), so it covers the codegen branch where prefill chunks run
# unsplit (split 0 does the chunk, the other splits are no-ops).
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=/home/muhengl/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
OUT=/home/muhengl/mpk-qwen35/m3i3/testmode.log
GPU=$(bash /home/muhengl/mpk-qwen35/i3_gpu_guard.sh "") || { echo "NO FREE GPU" | tee "$OUT"; exit 1; }
echo "CLAIMED GPU $GPU  $(date -Is)" | tee "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU"
cd /home/muhengl/mpk-qwen35/mirage-rm/tests/runtime_python/blackwell/sm100_gdn_recurrent
echo "gdn_recurrent_sm100.cuh md5: $(md5sum ../../../../include/mirage/persistent_kernel/tasks/blackwell/gdn_recurrent_sm100.cuh)" | tee -a "$OUT"
echo "GDN_SPLIT/GDN_DEPTH from test file: $(grep -E '^GDN_(SPLIT|DEPTH)' test_gdn_recurrent_testmode.py | tr '\n' ' ')" | tee -a "$OUT"
rm -rf test_output_gdn_recurrent
timeout 1500 /home/muhengl/mpk-qwen35/venv-rm/bin/python test_gdn_recurrent_testmode.py 2>&1 | tee -a "$OUT"
echo "TESTMODE_EXIT=${PIPESTATUS[0]}" | tee -a "$OUT"
