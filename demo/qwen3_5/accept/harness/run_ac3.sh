#!/bin/bash
# Thin entry point for the AC-3 harness — forwards straight to run_ac3.py so it can be
# invoked the same way regardless of caller cwd (matches accept.sh's expectation of a
# reviewable, callable workspace script). See run_ac3.py --help for the full contract.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/run_ac3.py" "$@"
