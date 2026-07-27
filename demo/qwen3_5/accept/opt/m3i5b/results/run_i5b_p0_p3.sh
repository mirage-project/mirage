#!/usr/bin/env bash
# M3-I9x EXTRA-B: M3-I5b P0-P3, exactly the commands in prep.md section 4,
# under the M3-I9 GPU guard (SM-residency law: never run CUDA-initializing
# work without exclusive GPU claim).
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
MIRAGE=$HOME/mpk-qwen35/mirage
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
M5B=$HOME/mpk-qwen35/m3i5b
CANDS=6,0,1,2,3,4,5,7
mkdir -p "$M5B/logs" "$M5B/ac3_dumps"

guard() { bash "$MIRAGE/demo/qwen3_5/accept/opt/m3i9/gpu_guard_m3i9.sh" "$CANDS" -- "$@"; }

STAGES=${STAGES:-P0,P1,P2,P3}
has() { case ",$STAGES," in *,$1,*) return 0;; *) return 1;; esac; }

# ---------------------------------------------------------------- P0
if has P0; then
echo "########## P0 unit tests $(date -Is)"

echo "--- P0.1 test_gate_topk.py $(date -Is)"
guard bash -c "cd '$MIRAGE/tests/runtime_python/blackwell/sm100_moe' && '$PY' setup.py build_ext --inplace && '$PY' test_gate_topk.py" \
    > "$M5B/logs/p0_gate_topk.log" 2>&1
echo "rc=$?"; tail -15 "$M5B/logs/p0_gate_topk.log"

echo "--- P0.2 test_gate_topk_sigmoid.py $(date -Is)"
guard bash -c "cd '$MIRAGE/tests/runtime_python/blackwell/sm100_moe_sigmoid' && '$PY' setup.py build_ext --inplace && '$PY' test_gate_topk_sigmoid.py" \
    > "$M5B/logs/p0_gate_topk_sigmoid.log" 2>&1
echo "rc=$?"; tail -15 "$M5B/logs/p0_gate_topk_sigmoid.log"

echo "--- P0.3 test_topk_sigmoid_testmode.py $(date -Is)"
guard bash -c "cd '$MIRAGE/tests/runtime_python/blackwell/sm100_moe_sigmoid' && '$PY' test_topk_sigmoid_testmode.py" \
    > "$M5B/logs/p0_topk_sigmoid_testmode.log" 2>&1
echo "rc=$?"; tail -15 "$M5B/logs/p0_topk_sigmoid_testmode.log"

echo "--- P0.4 test_router_oracle.py $(date -Is)"
guard bash -c "cd '$MIRAGE/tests/runtime_python/blackwell/sm100_moe_block_qwen35' && '$PY' setup.py build_ext --inplace && '$PY' test_router_oracle.py" \
    > "$M5B/logs/p0_router_oracle.log" 2>&1
echo "rc=$?"; tail -25 "$M5B/logs/p0_router_oracle.log"
fi

# ---------------------------------------------------------------- P1
if has P1; then
echo "########## P1 mbt=32 prefill probe $(date -Is)"
guard bash -c "cd '$MIRAGE/demo/qwen3_5/accept/results' && '$PY' probe_prefill.py --prompt-id p01-history --mbt 32 --rows 16 --out probe_prefill_mbt32_i5b.json" \
    > "$M5B/logs/p1_probe.log" 2>&1
echo "rc=$?"; tail -30 "$M5B/logs/p1_probe.log"
cp "$MIRAGE/demo/qwen3_5/accept/results/probe_prefill_mbt32_i5b.json" "$M5B/" 2>&1
fi

# ---------------------------------------------------------------- P2
if has P2; then
echo "########## P2 full-sweep AC-3 byte-diff vs dumps_final $(date -Is)"
for BS in 1 2 4 8 16; do
  echo "--- P2 bs=$BS $(date -Is)"
  guard "$PY" -u "$MIRAGE/demo/qwen3_5/accept/mpk_engine_run.py" --batch-size "$BS" \
      --max-seq-length 132 \
      --out-dir "$M5B/ac3_dumps" \
      --kernel-dir "$M5B/kernel_ac3_bs$BS" \
      > "$M5B/logs/p2_bs${BS}.log" 2>&1
  echo "rc=$?"; tail -8 "$M5B/logs/p2_bs${BS}.log"
done
echo "--- P2 run_ac3.py harness $(date -Is)"
"$PY" -u "$MIRAGE/demo/qwen3_5/accept/harness/run_ac3.py" \
    --engine-dump-dir "$M5B/ac3_dumps" --batch-sizes 1,2,4,8,16 \
    --output-json "$M5B/run_report.json" > "$M5B/logs/p2_run_ac3.log" 2>&1
echo "rc=$?"; tail -20 "$M5B/logs/p2_run_ac3.log"
echo "--- P2 bytediff vs results/dumps_final $(date -Is)"
"$PY" -u "$HOME/mpk-qwen35/m3i2a/bytediff.py" \
    "$MIRAGE/demo/qwen3_5/accept/results/dumps_final" \
    "$M5B/ac3_dumps" 1,2,4,8,16 > "$M5B/logs/p2_bytediff.log" 2>&1
echo "rc=$?"; cat "$M5B/logs/p2_bytediff.log"
fi

# ---------------------------------------------------------------- P3
if has P3; then
echo "########## P3 Qwen3-8B CI smoke $(date -Is)"
guard bash -c "cd '$MIRAGE/demo/qwen3' && '$PY' -u demo.py --use-mirage --max-new-tokens 50 --max-num-batched-tokens 8 --max-num-batched-requests 1" \
    > "$M5B/logs/p3_smoke.log" 2>&1
echo "rc=$?"; tail -40 "$M5B/logs/p3_smoke.log"
fi

echo "=== EXTRA-B DRIVER DONE $(date -Is)"
