#!/bin/bash
# One SGLang ROCm server run on the MI355X box (inside the SGLang image, through the GPU lease): full K3 + DSpark, TP 8, the
# B300 run's flags (mirage_kda scripts/b300_session/sgl_mpk_server.sh: DSPARK, 8 draft tokens per verify, max 8 requests, graphs
# bs 1..8, ReplaySSM spec rings) plus the MI35x recipe's (test_kimi_k3_perf_mi35x.py: triton attention, bf16, AITER env,
# multithread load). Then the B300 client (8 prompts one at a time, NTOK new tokens, greedy) and a torch-profiler trace of 2
# generations of 8 tokens.
#   srv_run.sh NAME [EXTRA server flags]     env: MODEL DRAFT MEMF NTOK GSM8K; MPK_K3_* (our hooks) passed through
# Out /mpk-amd/srv/NAME/: log.txt, server.log, answers.json, *TP-0*.trace.json*
N=$1; shift; EXTRA="$*"
cd /tmp   # (not a directory where `import sglang` finds the source checkout)
S=/mpk-amd/mirage_stage/amd/serving; P=/mpk-amd/srv/$N; rm -rf $P; mkdir -p $P; LOG=$P/log.txt; : > $LOG
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1
MODEL=${MODEL:-/mpk-amd/k3}; DRAFT=${DRAFT:-/mpk-amd/k3_dspark}; MEMF=${MEMF:-0.85}
# ATTN: --attention-backend (default triton); REPL: the ReplaySSM flag (default on; REPL= off)
ATTN=${ATTN:-triton}; REPL=${REPL---enable-linear-replayssm-spec}
echo "$N model $MODEL draft $DRAFT memf $MEMF attn $ATTN repl '$REPL' extra '$EXTRA' hooks: $(env | grep '^MPK_K3' | tr '\n' ' ') gpus $ROCR_VISIBLE_DEVICES $(date -u +%T)" >> $LOG
grep -c "mpk_amd" $(python -c 'import sglang,os;print(os.path.join(os.path.dirname(sglang.__file__),"srt/models/kimi_k3.py"))') \
  | sed 's/^/hook lines in kimi_k3.py: /' >> $LOG
cd /tmp
SGLANG_TORCH_PROFILER_DIR=$P python -m sglang.launch_server --model-path $MODEL --trust-remote-code --tp 8 --dtype bfloat16 \
  --attention-backend $ATTN --speculative-algorithm DSPARK --speculative-draft-model-path $DRAFT \
  --port 30000 --host 127.0.0.1 --mem-fraction-static $MEMF --max-running-requests 8 --cuda-graph-max-bs-decode 8 \
  $REPL --watchdog-timeout 3600 --skip-server-warmup \
  --model-loader-extra-config '{"enable_multithread_load": true}' $EXTRA > $P/server.log 2>&1 &
SP=$!
until curl -s -m 3 127.0.0.1:30000/health > /dev/null 2>&1; do
  kill -0 $SP 2>/dev/null || { echo "SERVER EXITED before ready $(date -u +%T)" >> $LOG; tail -40 $P/server.log >> $LOG; echo SRV-RUN-DONE >> $LOG; exit 1; }
  sleep 5
done
echo "server up $(date -u +%T)" >> $LOG
python3 $S/e2e_client.py $P ${NTOK:-128} >> $LOG 2>&1
# GSM8K=N: SGLang's 5-shot GSM8K, N questions, one at a time (our kernels run on batch-1 target-verify steps), greedy (as
# the B300 runs: mirage_kda scripts/b300_session/sgl_mpk_server.sh); data /mpk-amd/gsm8k_test.jsonl
if [ -n "$GSM8K" ]; then
  python3 -m sglang.test.few_shot_gsm8k --num-questions $GSM8K --parallel 1 --port 30000 --max-new-tokens 512 \
    --data-path /mpk-amd/gsm8k_test.jsonl > $P/gsm8k.txt 2>&1
  echo "GSM8K ($GSM8K questions, parallel 1): $(grep -E 'Accuracy|Invalid|Latency|Output throughput' $P/gsm8k.txt | tr '\n' ' ')" >> $LOG
fi
gen() { curl -s -m 900 http://127.0.0.1:30000/generate -H "Content-Type: application/json" -d "{\"text\": \"$2\", \"sampling_params\": {\"max_new_tokens\": $1, \"temperature\": 0}}"; }
curl -s -m 30 -X POST http://127.0.0.1:30000/start_profile -H "Content-Type: application/json" -d "{\"output_dir\": \"$P\", \"activities\": [\"CPU\", \"GPU\"], \"num_steps\": 30, \"record_shapes\": false}" > /dev/null
gen 8 "The history of the Roman Empire begins with" > /dev/null; gen 8 "The history of the Roman Empire begins with" > /dev/null
sleep 3; curl -s -m 60 -X POST http://127.0.0.1:30000/stop_profile > /dev/null
# every rank's trace written out: 8 non-empty files whose sizes stop changing for 30 s (a server killed earlier leaves them
# cut off; the files exist with 0 bytes while the ranks still write them: srv20 / srv20b were killed then)
for i in $(seq 240); do
  [ "$(ls $P/*TP-*.trace.json* 2>/dev/null | wc -l)" = 8 ] && [ -z "$(find $P -name '*TP-*.trace.json*' -size 0)" ] && { a=$(du -b $P/*TP-*.trace.json* | md5sum); sleep 30; [ "$a" = "$(du -b $P/*TP-*.trace.json* | md5sum)" ] && break; }
  kill -0 $SP 2>/dev/null || break; sleep 5
done
echo "trace: $(ls $P/*TP-0*.trace.json* 2>/dev/null | head -1) $(date -u +%T)" >> $LOG
grep -a "\[mpk_amd" $P/server.log > $P/mpk_lines.txt; cut -c1-400 $P/mpk_lines.txt | tail -60 >> $LOG
kill $SP; sleep 8; pkill -f "sglang::" ; sleep 3
echo SRV-RUN-DONE >> $LOG
