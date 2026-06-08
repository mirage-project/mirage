#!/usr/bin/env bash
# Online-serving CI: launch the OpenAI-compatible HTTP server, wait for the
# persistent kernel to compile, then drive single / concurrent / streaming
# requests against it via demo/qwen3/demo_online.py.  Tears the server down on
# exit.  Any failed mode (empty/errored response) makes demo_online.py exit
# non-zero, which fails this script and the CI step.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIRAGE_HOME="${MIRAGE_HOME:-$ROOT}"

PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
MAX_NUM_BATCHED_REQUESTS="${MAX_NUM_BATCHED_REQUESTS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8}"
CONCURRENT="${CONCURRENT:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"   # per-request timeout (seconds)
READY_TIMEOUT="${READY_TIMEOUT:-1200}"      # wait for compile + startup (seconds)

DEMO="$ROOT/demo/qwen3/demo_online.py"
LOG_DIR="${MIRAGE_HOME}/logs"
SERVER_LOG="${LOG_DIR}/online_server.log"
mkdir -p "$LOG_DIR"

# Poll the server's /docs endpoint until it answers 200, the server process
# dies, or the timeout elapses.  The server does not accept connections until
# its FastAPI lifespan finishes compiling the persistent kernel, so connection
# errors are expected (and retried) during startup.
wait_for_ready() {
    local timeout="$1"
    local deadline
    deadline=$(( $(date +%s) + timeout ))
    while (( $(date +%s) < deadline )); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server process (pid=${SERVER_PID}) exited before becoming ready"
            return 1
        fi
        if python -c "
import sys, urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:${PORT}/docs', timeout=5) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    echo "Server at 127.0.0.1:${PORT} not ready after ${timeout}s"
    return 1
}

echo "MIRAGE_HOME=${MIRAGE_HOME}"
echo "Launching online server: port=${PORT} model=${MODEL} batched_requests=${MAX_NUM_BATCHED_REQUESTS} batched_tokens=${MAX_NUM_BATCHED_TOKENS}"

python -m mirage.engine.launch_server \
    --host 127.0.0.1 \
    --port "$PORT" \
    --model "$MODEL" \
    --max-num-batched-requests "$MAX_NUM_BATCHED_REQUESTS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    rc=$?
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping server (pid=${SERVER_PID})..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 0.5
        done
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
    if [[ "$rc" -ne 0 ]]; then
        echo "=== server log (last 100 lines) ==="
        tail -n 100 "$SERVER_LOG" 2>/dev/null || true
        echo "=== end server log ==="
    fi
}
trap cleanup EXIT

# First wait absorbs the kernel compile; later waits return immediately when
# the server is up, or fast-fail if it has died.
echo "=== single request ==="
wait_for_ready "$READY_TIMEOUT"
python "$DEMO" --port "$PORT" --timeout "$REQUEST_TIMEOUT"

echo "=== ${CONCURRENT} concurrent requests ==="
wait_for_ready "$READY_TIMEOUT"
python "$DEMO" --port "$PORT" --timeout "$REQUEST_TIMEOUT" --concurrent "$CONCURRENT"

echo "=== streaming request ==="
wait_for_ready "$READY_TIMEOUT"
python "$DEMO" --port "$PORT" --timeout "$REQUEST_TIMEOUT" --stream

echo "Online-serving checks passed."
