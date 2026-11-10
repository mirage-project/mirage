#!/bin/bash
# One SGLang ROCm server run for the batch-size sweep (plain decode, no speculative decoding): GLM-5.3-Flash, TP 4, no EP
# (user 11-07), the SGLang cookbook's MI355X FP8 "high throughput" command (no speculative decoding) without --ep-size:
# AITER, fp8 KV cache, page 64, tilelang DSA, aiter MoE runner. (A copy of amd/serving/srv_run_bs.sh, the K3 script.) Then per batch size in BSLIST: bs_client.py (BS requests at once, NTOK new tokens
# each, end-of-sequence ignored) once to warm up (WARM new tokens: a size's first run pays first-use costs), once for speed and
# once with a PROF-step profiler trace (trace_bs<BS>/), and GSM8K with
# GSM8K questions, GSM8K_PAR at once (unset: skipped).
#   srv_run_bs.sh NAME      env: BSLIST ("1 2 4 8 16 32 64") NTOK (256) WARM (32) PROF (20) GSM8K GSM8K_PAR (64, or a list) MODEL MEMF
#                           CGBS (unset: SGLang's decode graph sizes; "1 2 4 8 16 32 64": only those, a batch padded to the
#                           next one, e.g. our per-size libraries); CHUNK (unset: SGLang's prefill chunk, 16384):
#                           --chunked-prefill-size (our libraries' own memory: a 16384-token prefill ran out, srv13);
#                           EXTRA: more launch_server arguments (e.g. --disable-cuda-graph);
#                           MPK_K3_*, MPK_AMD_* (our hooks) passed through
#   MPK_GLM=run (+ MPK_GLM_TOKENS, MPK_GLM_DLIB, GLM_TREE: the tree whose amd/serving/glm/sglang_glm.py is hooked): our kernels
#   (amd/serving/glm/sglang_patch_glm.py patches the container's glm5_next.py first; without MPK_GLM=run it is restored)
# Out /mpk-amd/glm/srv/NAME/: log.txt, server.log, answers_bs<BS>.json, trace_bs<BS>/*TP-*.trace.json.gz
N=$1
S=$(cd "$(dirname "$0")" && pwd); P=/mpk-amd/glm/srv/$N; rm -rf $P; mkdir -p $P; LOG=$P/log.txt; : > $LOG
cd /tmp   # (not a directory where `import sglang` finds the source checkout)
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false SGLANG_USE_AITER=1 AITER_ONLINE_TUNE=0
MODEL=${MODEL:-/dev/shm/glm53f}; MEMF=${MEMF:-0.82}; BSLIST=${BSLIST:-1 2 4 8 16 32 64}; NTOK=${NTOK:-256}; PROF=${PROF:-20}; WARM=${WARM:-32}
echo "$N model $MODEL memf $MEMF bs '$BSLIST' ntok $NTOK warm $WARM cgbs '${CGBS:-default}' hooks: $(env | grep '^MPK_GLM\|^GLM_' | tr '\n' ' ') gpus $ROCR_VISIBLE_DEVICES $(date -u +%T)" >> $LOG
GLM_TREE=${GLM_TREE:-/mpk-amd/glm/mirage_glm5}
if [ "$MPK_GLM" = run ]; then python3 $GLM_TREE/amd/serving/glm/sglang_patch_glm.py >> $LOG 2>&1 || { echo "PATCH FAILED" >> $LOG; echo SRV-RUN-DONE >> $LOG; exit 1; }
else python3 $GLM_TREE/amd/serving/glm/sglang_patch_glm.py --restore >> $LOG 2>&1; fi
python -c 'import sglang; print("sglang", sglang.__version__)' >> $LOG 2>&1
python -m sglang.launch_server --model-path $MODEL --trust-remote-code --quantization fp8 --tp-size 4 \
  --page-size 64 --mem-fraction-static $MEMF --max-running-requests 96 --cuda-graph-max-bs-decode 96 \
  ${CGBS:+--cuda-graph-bs-decode $CGBS} --min-free-slots-delay 1 --chunked-prefill-size ${CHUNK:-8192} --max-prefill-tokens 8192 \
  --prefill-decode-interval 0 --dsa-prefill-backend tilelang --dsa-decode-backend tilelang --linear-attn-verify-backend triton \
  --kv-cache-dtype fp8_e4m3 --moe-runner-backend aiter --reasoning-parser glm45 --tool-call-parser glm47 \
  --watchdog-timeout 1800 --dist-timeout 600 --port 30000 --host 127.0.0.1 --disable-radix-cache --skip-server-warmup \
  $EXTRA > $P/server.log 2>&1 &
SP=$!
W0=$(date +%s)   # (STARTMAX s, default 1500: a server not ready by then is killed - glmsrv4: faulted ranks, the others waited forever)
until curl -s -m 3 127.0.0.1:30000/health > /dev/null 2>&1; do
  kill -0 $SP 2>/dev/null || { echo "SERVER EXITED before ready $(date -u +%T)" >> $LOG; tail -40 $P/server.log >> $LOG; echo SRV-RUN-DONE >> $LOG; exit 1; }
  [ $(( $(date +%s) - W0 )) -gt ${STARTMAX:-1500} ] && { echo "SERVER NOT READY after ${STARTMAX:-1500} s $(date -u +%T)" >> $LOG
    grep -a -E "Memory Fault|Traceback|mpk-glm" $P/server.log | tail -12 >> $LOG; kill $SP; sleep 8; pkill -f "sglang::"; echo SRV-RUN-DONE >> $LOG; exit 1; }
  sleep 5
done
echo "server up $(date -u +%T)" >> $LOG
grep -a -o "Capture cuda graph bs [^)]*\|cuda_graph_bs[^,]*,[^]]*\]" $P/server.log | head -2 >> $LOG
for B in $BSLIST; do
  [ "$WARM" -gt 0 ] && python3 $S/bs_client.py $P $B $WARM > /dev/null 2>&1
  python3 $S/bs_client.py $P $B $NTOK >> $LOG 2>&1
  [ "$PROF" -gt 0 ] || continue   # (PROF=0: no profiled pass)
  # (PDELAY: s after the requests start before the profiler starts, bs_client.py DELAY, default 2; 10-22 srv24 / 25 / 27: at bs 1
  # the 256-token request ended before the profiler's steps began, the profiler then recorded the idle server for 5 min until
  # bs 2's steps came and its export grew each scheduler to ~300 GB host memory, OOM-killed)
  # (only at B <= PDELAY_BS, default 0: the other sizes' profiled steps where SGLang alone's traces have theirs)
  DELAY=$([ "$B" -le "${PDELAY_BS:-0}" ] && echo ${PDELAY:-2} || echo 2) PROFILE=$PROF python3 $S/bs_client.py $P $B $NTOK >> $LOG 2>&1
  # the trace written out: rank files whose sizes stop changing
  for i in $(seq 60); do
    n=$(ls $P/trace_bs$B/*TP-*.trace.json* 2>/dev/null | wc -l)
    [ "$n" -ge 4 ] && [ -z "$(find $P/trace_bs$B -name '*TP-*.trace.json*' -size 0)" ] && { a=$(du -b $P/trace_bs$B/* | md5sum); sleep 10; [ "$a" = "$(du -b $P/trace_bs$B/* | md5sum)" ] && break; }
    sleep 5
  done
  echo "trace bs $B: $(ls $P/trace_bs$B/*TP-1*.trace.json* 2>/dev/null | head -1) $(date -u +%T)" >> $LOG
done
if [ -n "$GSM8K" ]; then   # (GSM8K_PAR: one value or a list, a run each, in order)
  for GP in ${GSM8K_PAR:-64}; do
    python3 -m sglang.test.few_shot_gsm8k --num-questions $GSM8K --parallel $GP --port 30000 --max-new-tokens 512 \
      --data-path /mpk-amd/gsm8k_test.jsonl > $P/gsm8k_p$GP.txt 2>&1
    echo "GSM8K ($GSM8K questions, parallel $GP): $(grep -E 'Accuracy|Invalid|Latency|Output throughput' $P/gsm8k_p$GP.txt | tr '\n' ' ')" >> $LOG
    curl -s -m 3 127.0.0.1:30000/health > /dev/null 2>&1 || { echo "server down after GSM8K parallel $GP" >> $LOG; break; }
  done
fi
grep -a "\[mpk-glm" $P/server.log > $P/glm_lines.txt; cut -c1-400 $P/glm_lines.txt | tail -40 >> $LOG
kill $SP; sleep 8; pkill -f "sglang::" ; sleep 3
echo SRV-RUN-DONE >> $LOG
