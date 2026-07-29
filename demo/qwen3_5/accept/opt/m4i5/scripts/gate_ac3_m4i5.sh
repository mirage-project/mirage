#!/usr/bin/env bash
# M4-I5 AC-3: correctness of the SPLIT arm.
#
# The knob's default is unchanged, so the default arm is the committed tree and
# needs no re-litigation.  What needs a gate is the arm that actually differs:
# MPK_MOE_N_SPLITS=4, the best-performing value.  Recipe copied from
# m3i6a/scripts/gate_all.sh's `run_ac3_arm`.
#
# Under the re-pinned AC-3 the pass condition is coherence + a >=90% agreement
# floor, with bit-exactness REPORTED rather than required.  The claim here is the
# strong one: grid.y partitions output columns, so the arm should be
# byte-identical to the committed dumps, and the per-case byte diff is the
# evidence.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=$HOME/mpk-qwen35/mirage-m4i5
ACC=$T/demo/qwen3_5/accept
export MPK_ACCEPT_DIR=$ACC
export PYTHONPATH=$T/python
PY=$HOME/mpk-qwen35/venv-rm/bin/python
M=/var/tmp/m4i5_ac3
K="${K:-4}"
BSLIST="${BSLIST:-16 1 2 4 8}"
DUMPS=$M/dumps_k${K}
mkdir -p "$DUMPS" "$M/logs"

echo "########## M4-I5 AC-3 arm MPK_MOE_N_SPLITS=$K  gpu=$GPU  $(date -Is) ##########"
used_on_pinned () {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v g="$GPU" '{gsub(/ /,"",$1)} $1+0==g+0 {gsub(/ /,"",$2); print $2+0}'
}
drain () {
  local i used
  for i in $(seq 1 90); do
    used=$(used_on_pinned); [ "${used:-9999}" -lt 500 ] && return 0; sleep 5
  done
  echo "    ABORT: device $GPU held ${used}MiB after 450s"; exit 97
}

for BS in $BSLIST; do
  [ -f "$DUMPS/tokens_bs${BS}.json" ] && { echo "##### bs$BS cached"; continue; }
  KDIR=$M/kernel_k${K}_bs${BS}
  echo "##### AC-3 wave bs=$BS $(date -Is)"
  drain
  B4=$(used_on_pinned)
  ( cd "$ACC" && MPK_MOE_N_SPLITS=$K timeout 3600 "$PY" -u mpk_engine_run.py \
      --batch-size "$BS" --out-dir "$DUMPS" --kernel-dir "$KDIR" \
      --max-seq-length 132 ) > "$M/logs/k${K}_bs${BS}.log" 2>&1
  echo "##### rc=$? bs=$BS before=${B4}MiB $(date -Is)"; tail -3 "$M/logs/k${K}_bs${BS}.log"
  printf '{"tag":"ac3_k%s_bs%s","pinned_device":"%s","mib_before":%s}\n' \
      "$K" "$BS" "$GPU" "${B4:-null}" > "$M/logs/audit_k${K}_bs${BS}.json"
  # audit the split that was actually compiled in
  grep -o "moe_fp8_blockscale_task_impl<bfloat16[^(]*" "$KDIR/test_rank0.cu" \
      2>/dev/null | head -2
done

echo "##### AC-3 GATE arm k=$K $(date -Is)"
"$PY" -u "$ACC/harness/run_ac3.py" --engine-dump-dir "$DUMPS" \
    --batch-sizes 1,2,4,8,16 --output-json "$M/run_report_k${K}.json" \
    > "$M/logs/k${K}_gate.log" 2>&1
echo "##### run_ac3 rc=$?"; tail -30 "$M/logs/k${K}_gate.log"

echo "##### PER-CASE BYTE DIFF vs committed results/dumps_final"
BD=$HOME/mpk-qwen35/m3i2a/bytediff.py
if [ -f "$BD" ]; then
  "$PY" -u "$BD" "$ACC/results/dumps_final" "$DUMPS" 1,2,4,8,16 \
      > "$M/bytediff_k${K}.json" 2> "$M/logs/k${K}_bytediff.err"
  echo "##### bytediff rc=$?"; tail -3 "$M/logs/k${K}_bytediff.err"
  "$PY" - "$M/bytediff_k${K}.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("identical:", d.get("identical"), " missing:", d.get("missing"))
print("counts:", json.dumps(d.get("counts")))
bad = {k: v for k, v in d.get("per_case", {}).items() if v != "identical"}
print("CHANGED:", json.dumps(bad, indent=1) if bad else "none")
EOF
else
  echo "##### bytediff.py not present at $BD -- skipped"
fi
echo "AC3_DONE $(date -Is)"
