#!/usr/bin/env bash
set -uo pipefail
MIRAGE=$HOME/mpk-qwen35/mirage-i5c-run
PY=$HOME/mpk-qwen35/venv-rm/bin/python
export PYTHONPATH=$MIRAGE/python
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=4
M=$HOME/mpk-qwen35/m3i5c

echo "=== P3: CI smoke (Qwen3-8B) $(date -Is) ==="
export PATH=/usr/local/cuda-12.8/bin:$PATH
cd $MIRAGE/demo/qwen3
$PY -u demo.py --use-mirage --max-new-tokens 50 \
    --max-num-batched-tokens 8 --max-num-batched-requests 1 \
    > $M/logs/s5_p3.log 2>&1
echo "P3 rc=$? $(date -Is)"
tail -20 $M/logs/s5_p3.log

echo "=== P4: profiled router cost, bs1 + bs16, 3 reps $(date -Is) ==="
cd $MIRAGE/demo/qwen3_5/accept
ids_for_bs() {
  case "$1" in
    1)  echo "p06-poem" ;;
    16) echo "p06-poem,p01-history,p04-chinese,p09-translate,p07-format,p05-cuda,p08-science,p10-logic,p03-python,p02-math" ;;
  esac
}
mkdir -p $M/p4_prof
for BS in 1 16; do
  KDIR=$M/kernels/p4_kernel_bs${BS}_prof
  for R in 0 1 2; do
    RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
    RAW=""; [ "$R" -eq 0 ] && RAW="--save-raw"
    echo "--- P4 profiled bs=$BS rep=$R ${RK:-(compile)} $(date -Is) ---"
    $PY -u opt/profile_wave.py --batch-size "$BS" \
        --prompt-ids "$(ids_for_bs "$BS")" --out-dir $M/p4_prof \
        --kernel-dir "$KDIR" --rep "$R" --slots 48000000 $RK $RAW \
        > $M/logs/s5_p4_bs${BS}_rep${R}.log 2>&1
    echo "rc=$? $(grep -h "wall=" $M/logs/s5_p4_bs${BS}_rep${R}.log | tail -1)"
  done
  echo "--- P4 parse bs=$BS $(date -Is) ---"
  $PY -u opt/parse_profile.py --raw $M/p4_prof/raw_bs${BS}_rep0.npz \
      --meta $M/p4_prof/meta_bs${BS}_rep0.json --names $M/p4_prof/task_names.json \
      --out-prefix $M/p4_prof/bs${BS} > $M/logs/s5_p4_parse_bs${BS}.log 2>&1
  echo "rc=$?"
done
echo "=== S5 DONE $(date -Is) ==="
