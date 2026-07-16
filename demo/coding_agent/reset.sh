#!/usr/bin/env bash
# Reset the coding-agent demo workspace back to its seed files (main.py, notes.txt).
# Reuses tool_runtime.reset_workspace() so the seed content has a single source.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -c "import sys; sys.path.insert(0, '$HERE'); import tool_runtime as R; \
R.reset_workspace(); print('workspace reset:', sorted(p.name for p in R.WORKSPACE.iterdir()))"
