#!/bin/bash
# Runs one job on a shared box: waits for an idle GPU, runs the job there alone, and starts over if anyone else's process
# appears on that GPU meanwhile (the timings would be wrong).
#
#   nohup bash gpu_runner.sh NAME NGPU 'COMMAND' > ~/mpk-amd/logs/runner_NAME.log 2>&1 &
#
# Every CHECK_S (180) seconds: a GPU is idle when no process uses it (rocm-smi --showpidgpus) and its memory in use is 0%.
# When NGPU GPUs are idle, COMMAND runs in our container mpk-amd-NAME (the SGLang ROCm image, ~/mpk-amd mounted at /mpk-amd)
# with HIP_VISIBLE_DEVICES = those GPUs. Every WATCH_S (5) seconds while it runs: a process on one of those GPUs that is not
# in our container -> the container is stopped, the attempt is logged as overlapped, and the wait starts again. The exit
# code of a clean attempt ends the runner; the last line of the log is "RUNNER-DONE NAME rc=<code> gpus=<list>" or
# "RUNNER-GAVE-UP" after MAX_TRIES overlapped attempts.
set -u
NAME=$1
NGPU=$2
CMD=$3
IMAGE=${IMAGE:-lmsysorg/sglang-rocm:v0.5.21-rocm10-mi35x-20261009}
CHECK_S=${CHECK_S:-180}
WATCH_S=${WATCH_S:-5}
MAX_TRIES=${MAX_TRIES:-40}
C=mpk-amd-$NAME

log() { echo "$(date -u +%H:%M:%S) $*"; }

# "pid gpu" lines for every process using a GPU
pid_gpus() {
  rocm-smi --showpidgpus 2>/dev/null | awk '/^PID [0-9]+ is using [1-9]/ {pid = $2; getline; for (i = 1; i <= NF; ++i) print pid, $i}'
}

idle_gpus() {
  local busy mem
  busy=$(pid_gpus | awk '{print $2}' | sort -u)
  rocm-smi --showmemuse --json 2>/dev/null | python3 -c '
import json, sys
busy = set(sys.argv[1].split())
d = json.load(sys.stdin)
for card, v in sorted(d.items(), key=lambda kv: int(kv[0][4:])):
    g = card[4:]
    if g not in busy and v.get("GPU Memory Allocated (VRAM%)") == "0":
        print(g)' "$busy"
}

tries=0
while :; do
  idle=$(idle_gpus | tr '\n' ' ')
  n=$(echo $idle | wc -w)
  if [ "$n" -lt "$NGPU" ]; then
    log "waiting: idle GPUs [${idle% }], need $NGPU; next check in ${CHECK_S}s"
    sleep "$CHECK_S"
    continue
  fi
  gpus=$(echo $idle | tr ' ' '\n' | head -n "$NGPU" | paste -sd,)
  tries=$((tries + 1))
  log "attempt $tries on GPUs $gpus"
  echo running > "$HOME/mpk-amd/logs/$NAME.rc"
  docker run -d --rm --name "$C" --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
    --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES="$gpus" -v "$HOME/mpk-amd:/mpk-amd" "$IMAGE" \
    bash -c "$CMD; echo \$? > /mpk-amd/logs/$NAME.rc" > /dev/null
  overlapped=""
  seen=""
  prev_others=""
  while [ "$(docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null)" = "true" ]; do
    # the GPUs' processes first, then our container's: a process of ours that starts between the two reads is then in both
    # (read the other way round, a job that starts short processes, e.g. one benchmark binary after another, looked like
    # another user's process: 10-11). A process counts as another user's only when two checks in a row see it.
    gp=$(pid_gpus)
    ours=" $(docker top "$C" -eo pid 2>/dev/null | tail -n +2 | tr '\n' ' ') "
    others=$(echo "$gp" | awk -v g=",$gpus," 'index(g, "," $2 ",") {print $1}' | while read -r p; do
      [ -z "$p" ] && continue
      case "$ours" in *" $p "*) ;; *) echo "$p" ;; esac
    done | sort -u | tr '\n' ' ')
    if [ -z "$seen" ]; then
      seen=$(echo "$gp" | while read -r p g; do case "$ours" in *" $p "*) echo "$g" ;; esac; done | sort -u | paste -sd,)
      [ -n "$seen" ] && log "our processes are on GPUs $seen (asked for $gpus)"
    fi
    both=""
    for p in $others; do case " $prev_others " in *" $p "*) both="$both $p" ;; esac; done
    if [ -n "$both" ]; then
      # the job may have ended already (its exit code written, the container still closing): then its results stand (10-11:
      # a finished job was taken as overlapped and run again)
      if [ "$(cat "$HOME/mpk-amd/logs/$NAME.rc" 2>/dev/null)" != "running" ]; then
        break
      fi
      overlapped=$both
      docker stop -t 5 "$C" > /dev/null 2>&1
      break
    fi
    prev_others=$others
    sleep "$WATCH_S"
  done
  if [ -n "$overlapped" ]; then
    log "OVERLAPPED: other processes [$overlapped] on GPUs $gpus; job stopped"
    if [ "$tries" -ge "$MAX_TRIES" ]; then
      log "RUNNER-GAVE-UP $NAME after $tries attempts"
      exit 1
    fi
    sleep "$CHECK_S"
    continue
  fi
  # --rm: the exit code is gone with the container; the job wrote it to /mpk-amd/logs/<NAME>.rc
  rc=$(cat "$HOME/mpk-amd/logs/$NAME.rc")
  log "RUNNER-DONE $NAME rc=$rc gpus=$gpus"
  exit 0
done
