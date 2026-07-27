#!/usr/bin/env bash
# M3-I9x EXTRA-A: M3-I8 F1 per-iteration closure capture.
# Profiled-lane invocation pattern reused verbatim from
# opt/m3i8/run_m3i8.sh's run_profiled(); v1 IS the current HEAD default
# (gate_padding_rows=True), so no arm install is needed post git-reset.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
export MPK_ACCEPT_DIR=$HOME/mpk-qwen35/mirage/demo/qwen3_5/accept
OPT=$HOME/mpk-qwen35/m3i1/opt
M9X=$HOME/mpk-qwen35/m3i9x
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
CANDS=6,0,1,2,3,4,5,7
mkdir -p "$M9X/f1"

guard() { bash "$HOME/mpk-qwen35/mirage/demo/qwen3_5/accept/opt/m3i9/gpu_guard_m3i9.sh" "$CANDS" -- "$@"; }

echo "=== EXTRA-A bs1 profiled capture (reuse kernel_v1_bs1_prof) $(date -Is)"
guard "$PY" -u "$OPT/profile_wave.py" --batch-size 1 --prompt-ids p06-poem \
    --out-dir "$M9X/f1" --kernel-dir "$HOME/mpk-qwen35/m3i8/kernel_v1_bs1_prof" \
    --reuse-kernel --rep 0 --save-raw \
    > "$M9X/f1/log_bs1.txt" 2>&1
echo "rc=$?"; tail -5 "$M9X/f1/log_bs1.txt"

echo "=== EXTRA-A bs2 profiled capture (reuse kernel_v1_bs2_prof) $(date -Is)"
guard "$PY" -u "$OPT/profile_wave.py" --batch-size 2 --prompt-ids p06-poem,p01-history \
    --out-dir "$M9X/f1" --kernel-dir "$HOME/mpk-qwen35/m3i8/kernel_v1_bs2_prof" \
    --reuse-kernel --rep 0 --save-raw \
    > "$M9X/f1/log_bs2.txt" 2>&1
echo "rc=$?"; tail -5 "$M9X/f1/log_bs2.txt"

ls -la "$M9X/f1"
echo "=== EXTRA-A DRIVER DONE $(date -Is)"
