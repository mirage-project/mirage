#!/usr/bin/env bash
# M3-I6a c2 closure: the COMPLETE three-way pass-size sweep (4 / 2 / 1) at ALL
# five batch sizes, one protocol, arms interleaved.
#
# Why a re-run and not just a pass=1 arm bolted on: the c2 gap is that the
# selection rested on partial evidence, and appending a third arm measured later,
# on a different tree, without the drain gate would swap one comparability
# problem for another.  So all three arms run back-to-back per (bs, rep) inside
# ONE GPU claim, at integrated HEAD, with the same drain gate and the same
# per-rep device audit.  MPK_ATTN_Q_PASS makes all three arms available from one
# tree, so nothing is swapped between arms except that one value.
#
#   geometry A -- AC-3 geometry: the 10 pinned reference prompts, msl=132
#   geometry B -- matched 256/1024: synth 256-token prompt, msl=353 (96 decode
#                 steps).  The most PREFILL-heavy geometry: 256*bs/16 prefill
#                 iterations against 96 decode steps, i.e. where pass=1's doubled
#                 pass count should hurt most.
#   geometry C -- deep context: msl=897 (640 decode steps, ctx 257->896).  The
#                 most DECODE-heavy geometry, where pass=1's marginally lower
#                 per-KV-token cost should help most.
# B and C therefore BRACKET the tradeoff that decides 2 vs 1, rather than
# sampling one side of it.
#
# Scratch lives on /var/tmp (md0, ~148G) because /raid is at 9G box-wide.
set -uo pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
export HF_HOME=$HOME/mpk-qwen35/hf
export PYTHONUNBUFFERED=1

# The guard exec's us with the device it actually won.  Everything below -- the
# drain gate and every audit record -- derives from THIS value, never from the
# candidate list we asked for (M3-I7's lesson).
GPU="${CUDA_VISIBLE_DEVICES:?the guard must pin CUDA_VISIBLE_DEVICES}"

T=$HOME/mpk-qwen35/mirage
ACC=$T/demo/qwen3_5/accept
OPT=$ACC/opt
export MPK_ACCEPT_DIR=$ACC
PY=$HOME/mpk-qwen35/venv-mpk/bin/python
M=/var/tmp/m3i6a_sweep
SEED_BASE=20260725
ARMS="${ARMS:-4 2 1}"
REPS="${REPS:-0 1 2}"
mkdir -p "$M/logs" "$M/audit"

echo "########## M3-I6a three-way pass-size sweep  gpu=$GPU  $(date -Is) ##########"
echo "tree: $(git -C "$T" log --oneline -1)"
echo "arms: $ARMS   reps: $REPS   scratch: $M"
df -BG /var/tmp | tail -1
AVAIL=$(df -BG --output=avail /var/tmp | tail -1 | tr -dc '0-9')
[ "${AVAIL:-0}" -lt 20 ] && { echo "REFUSING: /var/tmp headroom ${AVAIL}G < 20G" >&2; exit 96; }

smi () { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits; }
used_on_pinned () {
  smi | awk -F',' -v g="$GPU" '{gsub(/ /,"",$1)} $1+0==g+0 {gsub(/ /,"",$2); print $2+0}'
}

drain () {   # the pinned device must be genuinely free before a rep starts
  local i used
  for i in $(seq 1 90); do
    used=$(used_on_pinned)
    [ "${used:-9999}" -lt 500 ] && return 0
    sleep 5
  done
  # A co-tenant arriving AFTER the claim invalidates every remaining rep on this
  # device, so do not soldier on: release the claim and let retry.sh re-probe
  # (possibly winning a different GPU).  Every completed rep is on disk and the
  # runners skip what already exists, so nothing is lost by bailing out here.
  echo "    ABORT: device $GPU held ${used}MiB by a co-tenant after 450s -- "
  echo "    releasing the claim so the guard can re-probe. Completed reps are kept."
  exit 97
}

# geometry A: mpk_engine_run.py does not record GPU state, so the audit is a
# sidecar written here -- same content, same per-rep granularity as
# profile_wave.py's own meta.gpu_before/gpu_after.
runA () {
  local BS="$1" QP="$2" REP="$3"
  local TAG=A_qp${QP}_bs${BS}_rep${REP}
  local AUD=$M/audit/${TAG}.json
  [ -f "$M/dumpsA_qp${QP}/timings_bs${BS}_rep${REP}" ] && { echo "  [$TAG] cached"; return; }
  drain
  local B4; B4=$(used_on_pinned)
  ( cd "$ACC" && MPK_ATTN_Q_PASS=$QP timeout 2400 $PY -u mpk_engine_run.py \
      --batch-size "$BS" --out-dir "$M/dumpsA_qp${QP}" \
      --kernel-dir "$M/kernel_A_qp${QP}_bs${BS}" --max-seq-length 132 \
      --dump-name "bs${BS}_rep${REP}" ) > "$M/logs/${TAG}.log" 2>&1
  local RC=$?
  local AF; AF=$(used_on_pinned)
  printf '{"tag":"%s","pinned_device":"%s","mib_before":%s,"mib_after":%s,"rc":%s}\n' \
      "$TAG" "$GPU" "${B4:-null}" "${AF:-null}" "$RC" > "$AUD"
  echo "  [$TAG] rc=$RC before=${B4}MiB $(grep -c 'wall=' "$M/logs/${TAG}.log" 2>/dev/null) waves"
}

# geometries B and C: profile_wave.py records gpu_before/gpu_after in its own
# meta, so the audit is already per-rep and self-describing.
runBC () {
  local GEOM="$1" BS="$2" QP="$3" REP="$4" MSL="$5"
  local TAG=${GEOM}_qp${QP}_bs${BS}_rep${REP}
  local OD=$M/noprof${GEOM}_qp${QP}
  [ -f "$OD/meta_bs${BS}_rep${REP}_qp${QP}.json" ] && { echo "  [$TAG] cached"; return; }
  local KDIR=$M/kernel_${GEOM}_qp${QP}_bs${BS}_noprof
  local SEED=$((SEED_BASE + BS*1000 + REP))
  local RK=""; [ -f "$KDIR/task_graph_rank0.json" ] && RK="--reuse-kernel"
  mkdir -p "$OD"
  drain
  MPK_ATTN_Q_PASS=$QP timeout 3000 $PY -u "$OPT/profile_wave.py" \
      --batch-size "$BS" --max-seq-length "$MSL" --max-new-tokens 96 --mbt 16 \
      --page-size 256 --synthetic-prompt-len 256 --synthetic-seed "$SEED" \
      --out-dir "$OD" --kernel-dir "$KDIR" --rep "$REP" --no-profiler $RK \
      > "$M/logs/${TAG}.log" 2>&1
  echo "  [$TAG] rc=$? $(grep -h 'wall=' "$M/logs/${TAG}.log" | tail -1)"
  local f=$OD/meta_bs${BS}_rep${REP}.json
  [ -f "$f" ] && mv "$f" "$OD/meta_bs${BS}_rep${REP}_qp${QP}.json"
}

echo; echo "--- geometry A: AC-3 (reference prompts, msl=132), arms interleaved ---"
for BS in 1 2 4 8 16; do for REP in $REPS; do for QP in $ARMS; do
  runA "$BS" "$QP" "$REP"
done; done; done

echo; echo "--- geometry B: matched 256/1024 (msl=353), arms interleaved ---"
for BS in 1 2 4 8 16; do for REP in $REPS; do for QP in $ARMS; do
  runBC B "$BS" "$QP" "$REP" 353
done; done; done

echo; echo "--- geometry C: deep context (msl=897, 640 decode steps), arms interleaved ---"
for BS in 1 2 4 8 16; do for REP in $REPS; do for QP in $ARMS; do
  runBC C "$BS" "$QP" "$REP" 897
done; done; done

echo; echo "PHASE6_DONE $(date -Is)"
df -BG /var/tmp | tail -1
