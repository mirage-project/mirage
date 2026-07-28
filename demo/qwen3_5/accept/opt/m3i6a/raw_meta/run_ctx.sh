#!/usr/bin/env bash
# M3-I6a measurement arm: ONE deep-context wave per (bs, Q_PASS).
#
# Geometry mirrors M3-I10's late-context closure (remeasure/scripts/
# run_armA_latectx.sh) exactly -- 256-token synthetic prompt, same seed
# formula, msl=897 = 256 + 640 + 1 -- so the numbers are directly comparable to
# ferret_targets.json's primary attention basis.  One wave walks context
# 257..896, and ctx_curve.py recovers the whole trajectory from it, so the
# short-vs-deep comparison needs no extra runs.
#
# Q_PASS is set through MPK_ATTN_Q_PASS, which changes the generated code, so
# every value gets its OWN kernel dir (never --reuse-kernel across values).
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1

T=$HOME/mpk-qwen35/mirage-i6a
M=$HOME/mpk-qwen35/i6a
OPT=$T/demo/qwen3_5/accept/opt
export MPK_ACCEPT_DIR=$T/demo/qwen3_5/accept
export PYTHONPATH=$T/python
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
SEED_BASE=20260725
MSL=${MSL:-897}
NEW=${NEW:-96}

mkdir -p "$M/prof" "$M/logs" "$M/tables"

AVAIL=$(df -BG --output=avail /raid | tail -1 | tr -dc '0-9')
echo "df /raid avail=${AVAIL}G"
if [ "${AVAIL:-0}" -lt 10 ]; then
  echo "REFUSING: /raid headroom ${AVAIL}G < 10G" >&2; exit 96
fi

for QP in ${QPLIST:-4 2}; do
  for BS in ${BSLIST:-1 8}; do
    SEED=$((SEED_BASE + BS*1000 + 0))
    KDIR=$M/kernel_qp${QP}_bs${BS}_msl${MSL}
    RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
    TAG="qp${QP}_bs${BS}_msl${MSL}"
    echo "===== [$TAG] msl=$MSL new=$NEW ${RK:-(compile)} $(date -Is) ====="
    MPK_ATTN_Q_PASS=$QP $PY -u "$OPT/profile_wave.py" --batch-size "$BS" \
        --max-seq-length "$MSL" --max-new-tokens "$NEW" --mbt 16 --page-size 256 \
        --synthetic-prompt-len 256 --synthetic-seed "$SEED" \
        --out-dir "$M/prof" --kernel-dir "$KDIR" \
        --rep 0 --slots 260000000 --save-raw $RK \
        > "$M/logs/${TAG}.log" 2>&1
    rc=$?
    echo "rc=$rc $(grep -hE 'wall=|profiler:' "$M/logs/${TAG}.log" | tail -2)"
    if [ "$rc" -ne 0 ]; then tail -25 "$M/logs/${TAG}.log"; continue; fi
    # rename the rep-0 outputs so the two Q_PASS arms do not overwrite each other
    for f in raw meta tokens; do
      src="$M/prof/${f}_bs${BS}_rep0"
      ext=npz; [ "$f" = "raw" ] || ext=json
      [ -f "$src.$ext" ] && mv "$src.$ext" "$M/prof/${f}_${TAG}.$ext"
    done
    echo "--- ctx_curve $TAG ---"
    I6A_OPT_DIR=$OPT I6A_LABEL_QPASS=$QP $PY -u "$M/ctx_curve.py" \
        --raw "$M/prof/raw_${TAG}.npz" --meta "$M/prof/meta_${TAG}.json" \
        --names "$M/prof/task_names.json" \
        --graph "$KDIR/task_graph_rank0.json" \
        --out "$M/tables/ctx_${TAG}.json" 2>&1 | tail -30
    echo "ctx_curve rc=$?"
  done
done
echo "RUN_CTX_DONE $(date -Is)"
