"""Guard: `profiler_persistent.event_name_list` must mirror `enum TaskType`.

The MPK profiler stores the raw `TaskType` enum value in each event tag, so a
task added to `include/mirage/persistent_kernel/runtime_header.h` without a
matching entry here shows up in traces as `UNKNOWN_<id>` (perfetto/CSV) or used
to crash the Perfetto export outright. Worse, a *stale* entry mislabels real
events: before this test existed, id 298 was labelled `TASK_SM100_TASK_END`
while the header had moved that placeholder to 299 and given 298 to
`TASK_DFLASH_KV_STORE_SM100`, and id 105 was labelled `TASK_SILU_MUL_LINEAR`
instead of `TASK_SILU_MUL_LINEAR_WITH_RESIDUAL`.

Pure CPU / pure-parse test: no torch CUDA, no GPU, no build artifacts.

Run standalone:  python tests/runtime_python/test_profiler_task_names.py
Run under pytest: pytest tests/runtime_python/test_profiler_task_names.py
"""

import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HEADER = os.path.join(
    REPO_ROOT, "include", "mirage", "persistent_kernel", "runtime_header.h"
)
PROFILER_PY = os.path.join(
    REPO_ROOT, "python", "mirage", "mpk", "profiler_persistent.py"
)

_ENTRY_RE = re.compile(r"^\s*(TASK_[A-Z0-9_]+)\s*=\s*(\d+)\s*,?\s*(?://.*)?$")


def parse_task_type_enum(path=HEADER):
    """Return {id: name} for `enum TaskType` in runtime_header.h."""
    with open(path) as f:
        lines = f.read().split("\n")
    start = None
    for i, line in enumerate(lines):
        if re.match(r"\s*enum\s+TaskType", line):
            start = i
            break
    assert start is not None, f"`enum TaskType` not found in {path}"

    out = {}
    depth = 0
    for i in range(start, len(lines)):
        line = lines[i]
        depth += line.count("{") - line.count("}")
        m = _ENTRY_RE.match(line)
        if m:
            name, value = m.group(1), int(m.group(2))
            assert value not in out, f"duplicate TaskType id {value} in {path}"
            out[value] = name
        if depth == 0 and i > start:
            break
    assert out, f"parsed no TaskType entries from {path}"
    return out


def _load_event_name_list():
    sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
    try:
        # Import the module by path so this test needs neither an installed
        # `mirage` nor `torch`/`tg4perfetto` (profiler_persistent imports both
        # at module scope).
        import ast

        with open(PROFILER_PY) as f:
            tree = ast.parse(f.read(), PROFILER_PY)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "event_name_list"
                for t in node.targets
            ):
                return ast.literal_eval(node.value)
    finally:
        sys.path.pop(0)
    raise AssertionError(f"`event_name_list` not found in {PROFILER_PY}")


def test_profiler_map_matches_runtime_header():
    enum_map = parse_task_type_enum()
    name_map = _load_event_name_list()

    missing = sorted(set(enum_map) - set(name_map))
    extra = sorted(set(name_map) - set(enum_map))
    wrong = sorted(
        (i, name_map[i], enum_map[i])
        for i in set(enum_map) & set(name_map)
        if name_map[i] != enum_map[i]
    )

    msgs = []
    if missing:
        msgs.append(
            "ids in runtime_header.h but NOT in event_name_list: "
            + ", ".join(f"{i} ({enum_map[i]})" for i in missing)
        )
    if extra:
        msgs.append(
            "ids in event_name_list but NOT in runtime_header.h: "
            + ", ".join(f"{i} ({name_map[i]})" for i in extra)
        )
    if wrong:
        msgs.append(
            "mislabelled ids (map -> header): "
            + ", ".join(f"{i}: {got} -> {want}" for i, got, want in wrong)
        )
    assert not msgs, (
        "python/mirage/mpk/profiler_persistent.py's event_name_list is out of "
        "sync with include/mirage/persistent_kernel/runtime_header.h:\n  "
        + "\n  ".join(msgs)
    )


if __name__ == "__main__":
    test_profiler_map_matches_runtime_header()
    n = len(parse_task_type_enum())
    print(f"OK: event_name_list matches enum TaskType exactly ({n} entries)")
