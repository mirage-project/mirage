#!/bin/bash
# One SGLang ROCm server run for the batch-size sweep (plain decode, no speculative decoding): GLM-5.3 (zai-org/GLM-5.3, FP8,
# GlmMoeDsaForCausalLM), TP 8, no EP - the SGLang cookbook's MI355X FP8 command without MTP and without --enable-amd-monokernel
# (v0.5.23 images only): tilelang DSA, fp8 KV cache, AITER, aiter MoE runner. Per batch size in BSLIST: bs_client.py once to warm
# up (WARM tokens), once for speed, once with a PROF-step profiler trace (trace_bs<BS>/); then GSM8K (GSM8K questions, unset:
# skipped). A copy of amd/serving/glm/srv_run_glm.sh (GLM-5.3-Flash, TP 4).
#   srv_run_glm53.sh NAME   env: BSLIST ("1 2 4 8 16 32 64") NTOK (256) WARM (32) PROF (20) GSM8K GSM8K_PAR (64) MODEL
#                           (/dev/shm/glm53) MEMF (0.85) TP (8) CGBS (unset: SGLang's decode graph sizes) CHUNK (8192) EXTRA
#                           STARTMAX (1800 s)
#   MPK_G53=run (+ MPK_G53_TOKENS, MPK_G53_DLIB, G53_TREE: the tree whose amd/serving/glm53/sglang_glm53.py is hooked): our layer
#   kernels (sglang_patch_glm53.py patches the container's deepseek_v2.py first; without MPK_G53=run it is restored)
# Out /mpk-amd/glm53/srv/NAME/: log.txt, server.log, answers_bs<BS>.json, trace_bs<BS>/*TP-*.trace.json.gz
N=$1
S=$(cd "$(dirname "$0")" && pwd); P=/mpk-amd/glm53/srv/$N; rm -rf $P; mkdir -p $P; LOG=$P/log.txt; : > $LOG
cd /tmp
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false SGLANG_USE_AITER=1 AITER_ONLINE_TUNE=0
MODEL=${MODEL:-/dev/shm/glm53}; MEMF=${MEMF:-0.85}; BSLIST=${BSLIST:-1 2 4 8 16 32 64}; NTOK=${NTOK:-256}; PROF=${PROF:-20}
WARM=${WARM:-32}; TP=${TP:-8}
echo "$N model $MODEL tp $TP memf $MEMF bs '$BSLIST' ntok $NTOK warm $WARM cgbs '${CGBS:-default}' hooks: $(env | grep '^MPK_G53\|^G53_' | tr '\n' ' ') gpus $ROCR_VISIBLE_DEVICES $(date -u +%T)" >> $LOG
G53_TREE=${G53_TREE:-/mpk-amd/glm53/g53t6}
if [ "$MPK_G53" = run ]; then python3 $G53_TREE/amd/serving/glm53/sglang_patch_glm53.py >> $LOG 2>&1 || { echo "PATCH FAILED" >> $LOG; echo SRV-RUN-DONE >> $LOG; exit 1; }
else python3 $G53_TREE/amd/serving/glm53/sglang_patch_glm53.py --restore >> $LOG 2>&1; fi
python -c 'import sglang; print("sglang", sglang.__version__)' >> $LOG 2>&1
python -m sglang.launch_server --model-path $MODEL --trust-remote-code --tp-size $TP --ep-size 1 \
  --page-size 64 --mem-fraction-static $MEMF --max-running-requests 96 --cuda-graph-max-bs-decode 96 \
  ${CGBS:+--cuda-graph-bs-decode $CGBS} --chunked-prefill-size ${CHUNK:-8192} --max-prefill-tokens 8192 \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang --kv-cache-dtype fp8_e4m3 --moe-runner-backend aiter \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --watchdog-timeout 1800 --dist-timeout 600 --port 30000 --host 127.0.0.1 --disable-radix-cache --skip-server-warmup \
  $EXTRA > $P/server.log 2>&1 &
SP=$!
W0=$(date +%s)
until curl -s -m 3 127.0.0.1:30000/health > /dev/null 2>&1; do
  kill -0 $SP 2>/dev/null || { echo "SERVER EXITED before ready $(date -u +%T)" >> $LOG; tail -40 $P/server.log >> $LOG; echo SRV-RUN-DONE >> $LOG; exit 1; }
  [ $(( $(date +%s) - W0 )) -gt ${STARTMAX:-1800} ] && { echo "SERVER NOT READY after ${STARTMAX:-1800} s $(date -u +%T)" >> $LOG
    grep -a -E "Memory Fault|Traceback|Error" $P/server.log | tail -12 >> $LOG; kill $SP; sleep 8; pkill -f "sglang::"; echo SRV-RUN-DONE >> $LOG; exit 1; }
  sleep 5
done
echo "server up $(date -u +%T) after $(( $(date +%s) - W0 )) s" >> $LOG
grep -a -o "Capture cuda graph bs [^)]*\|cuda_graph_bs[^,]*,[^]]*\]" $P/server.log | head -2 >> $LOG
for B in $BSLIST; do
  [ "$WARM" -gt 0 ] && python3 $S/bs_client.py $P $B $WARM > /dev/null 2>&1
  python3 $S/bs_client.py $P $B $NTOK >> $LOG 2>&1
  [ "$PROF" -gt 0 ] || continue
  PROFILE=$PROF python3 $S/bs_client.py $P $B $NTOK >> $LOG 2>&1
  for i in $(seq 60); do
    n=$(ls $P/trace_bs$B/*TP-*.trace.json* 2>/dev/null | wc -l)
    [ "$n" -ge $TP ] && [ -z "$(find $P/trace_bs$B -name '*TP-*.trace.json*' -size 0)" ] && { a=$(du -b $P/trace_bs$B/* | md5sum); sleep 10; [ "$a" = "$(du -b $P/trace_bs$B/* | md5sum)" ] && break; }
    sleep 5
  done
  echo "trace bs $B: $(ls $P/trace_bs$B/*TP-0*.trace.json* 2>/dev/null | head -1) $(date -u +%T)" >> $LOG
done
if [ -n "$GSM8K" ]; then
  for GP in ${GSM8K_PAR:-64}; do
    python3 -m sglang.test.few_shot_gsm8k --num-questions $GSM8K --parallel $GP --port 30000 --max-new-tokens 512 \
      --data-path /mpk-amd/gsm8k_test.jsonl > $P/gsm8k_p$GP.txt 2>&1
    echo "GSM8K ($GSM8K questions, parallel $GP): $(grep -E 'Accuracy|Invalid|Latency|Output throughput' $P/gsm8k_p$GP.txt | tr '\n' ' ')" >> $LOG
  done
fi
grep -a "\[mpk-g53" $P/server.log | cut -c1-400 | tail -30 >> $LOG
kill $SP; sleep 8; pkill -f "sglang::" ; sleep 3
echo SRV-RUN-DONE >> $LOG
