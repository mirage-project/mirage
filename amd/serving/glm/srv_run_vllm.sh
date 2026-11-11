#!/bin/bash
# One vLLM ROCm server run for the batch-size sweep, the counterpart of srv_run_glm.sh (SGLang): GLM-5.3-Flash, TP 4, no EP, no
# speculative decoding, no prefix caching, decode graphs for 1 2 4 8 16 32 64, greedy NTOK-token requests (bs_client_oai.py:
# the same prompts and the same printed line), a torch-profiler trace per batch size (vLLM's /start_profile, /stop_profile).
# Runs inside the vLLM image (vllm/vllm-openai-rocm:v0.32.0; its entrypoint is `vllm serve`: start the job with --entrypoint bash).
#   srv_run_vllm.sh NAME      env: BSLIST ("1 2 4 8 16 32 64") NTOK (256) WARM (32) PROF (1) MODEL (/dev/shm/glm53f) MEMF (0.82)
#                             KVD (fp8: --kv-cache-dtype) AITER (1: VLLM_ROCM_USE_AITER) MAXLEN (4096) EXTRA (more serve args)
#                             STARTMAX (2400 s) TP (4) MAXSEQ (96)   (Kimi K3: MODEL=/mpk-amd/k3 TP=8 MAXSEQ=64 MEMF=0.85)
# Out /mpk-amd/glm/srv/NAME/: log.txt, server.log, answers_bs<BS>.json, prof/ (traces of every batch size: prof/bs<BS>/)
N=$1
S=$(cd "$(dirname "$0")" && pwd); P=/mpk-amd/glm/srv/$N; rm -rf $P; mkdir -p $P/prof; LOG=$P/log.txt; : > $LOG
cd /tmp
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_ROCM_USE_AITER=${AITER:-1}
MODEL=${MODEL:-/dev/shm/glm53f}; MEMF=${MEMF:-0.82}; BSLIST=${BSLIST:-1 2 4 8 16 32 64}; NTOK=${NTOK:-256}; PROF=${PROF:-1}
WARM=${WARM:-32}; KVD=${KVD:-fp8}; MAXLEN=${MAXLEN:-4096}; PORT=30001; TP=${TP:-4}; MAXSEQ=${MAXSEQ:-96}
echo "$N vllm $(python3 -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -1) model $MODEL memf $MEMF kv $KVD aiter $VLLM_ROCM_USE_AITER tp $TP maxseq $MAXSEQ bs '$BSLIST' ntok $NTOK gpus $ROCR_VISIBLE_DEVICES $(date -u +%T)" >> $LOG
vllm serve $MODEL --served-model-name glm --trust-remote-code --tensor-parallel-size $TP --max-num-seqs $MAXSEQ \
  --gpu-memory-utilization $MEMF --kv-cache-dtype $KVD --max-model-len $MAXLEN --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]}' \
  --profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$P/prof\", \"torch_profiler_with_stack\": false}" \
  --host 127.0.0.1 --port $PORT $EXTRA > $P/server.log 2>&1 &
SP=$!
W0=$(date +%s)
until curl -s -m 3 127.0.0.1:$PORT/health > /dev/null 2>&1; do
  kill -0 $SP 2>/dev/null || { echo "SERVER EXITED before ready $(date -u +%T)" >> $LOG; tail -40 $P/server.log >> $LOG; echo SRV-RUN-DONE >> $LOG; exit 1; }
  [ $(( $(date +%s) - W0 )) -gt ${STARTMAX:-2400} ] && { echo "SERVER NOT READY after ${STARTMAX:-2400} s $(date -u +%T)" >> $LOG
    grep -a -E "Error|Traceback" $P/server.log | tail -12 >> $LOG; kill $SP; sleep 8; pkill -f "vllm serve"; echo SRV-RUN-DONE >> $LOG; exit 1; }
  sleep 5
done
echo "server up $(date -u +%T) after $(( $(date +%s) - W0 )) s" >> $LOG
grep -a -i -E "cudagraph_mode|cuda graph|Capturing|attention backend|Using .* backend" $P/server.log | head -6 | cut -c1-240 >> $LOG
for B in $BSLIST; do
  [ "$WARM" -gt 0 ] && python3 $S/bs_client_oai.py $P $B $WARM $PORT > /dev/null 2>&1
  python3 $S/bs_client_oai.py $P $B $NTOK $PORT >> $LOG 2>&1
  [ "$PROF" -gt 0 ] || continue
  n0=$(find $P/prof -type f | wc -l)
  PROFILE=1 python3 $S/bs_client_oai.py $P $B $NTOK $PORT >> $LOG 2>&1
  mkdir -p $P/prof/bs$B
  for i in $(seq 60); do   # (the new rank traces written out: their sizes stop changing)
    new=$(find $P/prof -maxdepth 1 -type f -newer $LOG.bs 2>/dev/null | wc -l)
    [ "$(find $P/prof -maxdepth 1 -type f | wc -l)" -gt "$n0" ] && { a=$(du -b $P/prof/* 2>/dev/null | md5sum); sleep 10; [ "$a" = "$(du -b $P/prof/* 2>/dev/null | md5sum)" ] && break; }
    sleep 5
  done
  find $P/prof -maxdepth 1 -type f -exec mv {} $P/prof/bs$B/ \;
  echo "trace bs $B: $(ls $P/prof/bs$B | head -4 | tr '\n' ' ') $(date -u +%T)" >> $LOG
done
kill $SP; sleep 10; pkill -f "vllm serve"
echo SRV-RUN-DONE >> $LOG
