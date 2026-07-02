r"""Sandboxed tool runtime for the large coding-agent demo.

Executes the registry tools INSIDE one workspace folder and enforces every
safety rule independently of the grammar — so the sandbox holds even if the
structural tag were bypassed:

  * paths must resolve inside the workspace (no absolute paths, no ``..``);
  * only a whitelist of source/text extensions may be read/written;
  * ``.py`` writes/edits must pass ``ast.parse`` before touching disk;
  * ``run_python`` / ``run_pytest`` use ``subprocess`` with **no shell**, a fixed
    cwd, captured output and a timeout;
  * ``git_status`` / ``git_diff`` are read-only.

``reset_workspace()`` restores the seed project and (re)inits a git repo with an
initial commit, so ``git_status`` / ``git_diff`` show the agent's own changes.
The grammar constrains the *shape* of each call; this module owns *safety*.
"""

import ast
import json
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
WORKSPACE = (HERE / "workspace").resolve()
SEED = HERE / "project_seed"

TEXT_EXTS = (".py", ".txt", ".md", ".json", ".toml", ".cfg", ".ini")
CMD_TIMEOUT_S = 30
MAX_OUTPUT = 4000          # truncate tool output fed back to the model
# Tests are the reward signal — the agent may READ them but never modify them,
# so it can't "pass" by weakening the suite (enforced, not just prompt-asked).
TEST_RE = re.compile(r"^(test_.*|.*_test|conftest)\.py$")


class ToolError(Exception):
    """A call violated a sandbox rule; the message is surfaced to the model."""


# ── Path safety ─────────────────────────────────────────────────────────────

def _resolve(path: str) -> Path:
    """Validate a workspace-relative path and return an absolute path guaranteed
    to live inside the workspace."""
    if not isinstance(path, str) or not path.strip():
        raise ToolError("path must be a non-empty string")
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        raise ToolError(f"illegal path {path!r}: must be workspace-relative, no '..'")
    resolved = (WORKSPACE / p).resolve()
    if not (resolved == WORKSPACE or resolved.is_relative_to(WORKSPACE)):
        raise ToolError(f"path escapes workspace: {path!r}")
    return resolved


def _check_ext(p: Path) -> None:
    if p.suffix not in TEXT_EXTS:
        raise ToolError(f"unsupported extension {p.suffix!r}; allowed: {TEXT_EXTS}")


def _guard_not_test(p: Path) -> None:
    """Refuse writes/edits to test files — the agent must fix the code, not the
    tests, so it can't pass by weakening the suite."""
    if TEST_RE.match(p.name):
        raise ToolError(f"editing test files is not allowed: {_rel(p)}")


def _validate_python(source: str, name: str) -> None:
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise ToolError(f"{name}: python syntax error: {e}") from e


def _rel(p: Path) -> str:
    return str(p.relative_to(WORKSPACE))


def _visible(p: Path) -> bool:
    """Hide the .git bookkeeping dir from the file-listing/search tools."""
    return ".git" not in p.relative_to(WORKSPACE).parts


def _clip(s: str) -> str:
    return s if len(s) <= MAX_OUTPUT else s[:MAX_OUTPUT] + "\n... (truncated)"


# ── Tools ───────────────────────────────────────────────────────────────────

def tool_list_dir(args: dict) -> str:
    root = _resolve(args["path"])
    if not root.is_dir():
        raise ToolError(f"not a directory: {args['path']!r}")
    names = sorted(_rel(p) for p in root.rglob("*") if p.is_file() and _visible(p))
    return "\n".join(names) if names else "(empty)"


def tool_read_file(args: dict) -> str:
    p = _resolve(args["path"])
    _check_ext(p)
    if not p.is_file():
        raise ToolError(f"no such file: {args['path']!r}")
    return _clip(p.read_text())


def tool_write_file(args: dict) -> str:
    p = _resolve(args["path"])
    _check_ext(p)
    _guard_not_test(p)
    content = args["content"]
    if not isinstance(content, str):
        raise ToolError("content must be a string")
    if p.suffix == ".py":
        _validate_python(content, p.name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {_rel(p)} ({len(content)} bytes)"


def _tolerant_replace(text: str, old: str, new: str):
    """Replace `old` with `new`. Exact-unique first; then a per-line match that
    ignores trailing whitespace and surrounding blank lines (models rarely
    reproduce a big block byte-for-byte). Returns (updated, note) or (None, why)."""
    n = text.count(old)
    if n == 1:
        return text.replace(old, new, 1), None
    if n > 1:
        return None, f"is ambiguous: {n} exact occurrences"
    fl = text.split("\n")
    on = [ln.rstrip() for ln in old.strip("\n").split("\n")]
    fn = [ln.rstrip() for ln in fl]
    hits = ([i for i in range(len(fn) - len(on) + 1) if fn[i:i + len(on)] == on]
            if on else [])
    if len(hits) == 1:
        i = hits[0]
        return "\n".join(fl[:i] + new.split("\n") + fl[i + len(on):]), \
            "matched ignoring trailing whitespace"
    if len(hits) > 1:
        return None, f"is ambiguous: {len(hits)} whitespace-insensitive matches"
    return None, "not found"


def tool_edit_file(args: dict) -> str:
    p = _resolve(args["path"])
    _check_ext(p)
    _guard_not_test(p)
    if not p.is_file():
        raise ToolError(f"no such file: {args['path']!r}")
    old, new = args["old"], args["new"]
    if not isinstance(old, str) or not isinstance(new, str):
        raise ToolError("old and new must be strings")
    text = p.read_text()
    updated, note = _tolerant_replace(text, old, new)
    if updated is None:
        raise ToolError(
            f"`old` {note} in {_rel(p)}. Do NOT retry the same edit — either copy "
            f"`old` verbatim from the file below, or (to replace most/all of it) "
            f"call write_file with the full new contents.\n--- {_rel(p)} ---\n"
            + _clip(text))
    if p.suffix == ".py":
        _validate_python(updated, p.name)
    p.write_text(updated)
    return f"edited {_rel(p)}" + (f" ({note})" if note else "")


def tool_find_files(args: dict) -> str:
    pattern = args["pattern"]
    if not isinstance(pattern, str) or "/" in pattern and ".." in pattern:
        raise ToolError("invalid pattern")
    hits = sorted(_rel(p) for p in WORKSPACE.rglob(pattern)
                  if p.is_file() and _visible(p))
    return "\n".join(hits) if hits else "(no matches)"


def tool_grep(args: dict) -> str:
    try:
        rx = re.compile(args["pattern"])
    except re.error as e:
        raise ToolError(f"bad regex: {e}") from e
    out = []
    for p in sorted(WORKSPACE.rglob("*")):
        if not p.is_file() or p.suffix not in TEXT_EXTS or not _visible(p):
            continue
        try:
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if rx.search(line):
                    out.append(f"{_rel(p)}:{i}: {line.strip()}")
        except (UnicodeDecodeError, OSError):
            continue
    return _clip("\n".join(out)) if out else "(no matches)"


def _run(cmd, label: str) -> str:
    try:
        proc = subprocess.run(cmd, cwd=str(WORKSPACE), capture_output=True,
                              text=True, timeout=CMD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise ToolError(f"{label} timed out after {CMD_TIMEOUT_S}s")
    except FileNotFoundError as e:
        raise ToolError(f"{label}: {e}") from e
    body = proc.stdout + (f"\n[stderr]\n{proc.stderr}" if proc.stderr else "")
    return _clip(f"exit={proc.returncode}\n{body}".rstrip())


def tool_run_python(args: dict) -> str:
    p = _resolve(args["path"])
    if p.suffix != ".py":
        raise ToolError("run_python only runs .py files")
    if not p.is_file():
        raise ToolError(f"no such file: {args['path']!r}")
    return _run(["python", _rel(p)], "run_python")


def tool_run_pytest(args: dict) -> str:
    # -s (no capture) lets the perf test's [PERF] speedup line show in the output.
    return _run(["python", "-m", "pytest", "-q", "-s"], "run_pytest")


def tool_git_status(args: dict) -> str:
    return _run(["git", "status", "--short"], "git_status") or "(clean)"


def tool_git_diff(args: dict) -> str:
    return _run(["git", "--no-pager", "diff"], "git_diff") or "(no diff)"


def tool_finish(args: dict) -> str:
    """End the session — but only if the suite is green. Runs pytest and raises
    ToolError on any failure, so a premature finish is rejected and the agent
    must keep fixing. Requires a non-empty summary of the changes made."""
    summary = args.get("summary", "")
    if not isinstance(summary, str) or not summary.strip():
        raise ToolError("finish requires a non-empty 'summary' of your changes")
    result = tool_run_pytest({})                 # "exit=<code>\n<pytest output>"
    if not result.startswith("exit=0"):
        raise ToolError("cannot finish — the test suite is not green:\n" + result)
    return f"all tests pass — task complete. {summary}"


_TOOLS = {
    "list_dir": tool_list_dir, "read_file": tool_read_file,
    "write_file": tool_write_file, "edit_file": tool_edit_file,
    "find_files": tool_find_files, "grep": tool_grep,
    "run_python": tool_run_python, "run_pytest": tool_run_pytest,
    "git_status": tool_git_status, "git_diff": tool_git_diff,
    "finish": tool_finish,
}
TOOL_NAMES = set(_TOOLS)


# ── Parse + dispatch ────────────────────────────────────────────────────────

_CALL_RE = re.compile(r"<function=(\w+)>(.*?)</function>", re.S)


def parse_tool_calls(text: str):
    """Extract ``(name, args_dict)`` pairs from model output (grammar-guaranteed
    shape, so this is a formality). Malformed JSON / unknown tool raises."""
    calls = []
    for name, raw in _CALL_RE.findall(text):
        raw = raw.strip() or "{}"
        try:
            args = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ToolError(f"{name}: bad JSON args {raw!r}: {e}") from e
        if name not in TOOL_NAMES:
            raise ToolError(f"unknown tool {name!r}")
        calls.append((name, args))
    return calls


def dispatch(name: str, args: dict) -> str:
    """Execute one registry tool, converting any ToolError into an ``ERROR: ...``
    string so the agent loop can feed it back instead of crashing."""
    if name not in _TOOLS:
        return f"ERROR: unknown tool {name!r}"
    try:
        return _TOOLS[name](args)
    except ToolError as e:
        return f"ERROR: {e}"
    except KeyError as e:
        return f"ERROR: missing argument {e}"
    except Exception as e:  # pragma: no cover - unexpected
        return f"ERROR: {type(e).__name__}: {e}"


def tests_pass() -> bool:
    """True iff the workspace pytest suite is fully green (used to auto-detect
    task completion)."""
    try:
        proc = subprocess.run(["python", "-m", "pytest", "-q"], cwd=str(WORKSPACE),
                              capture_output=True, text=True, timeout=CMD_TIMEOUT_S)
        return proc.returncode == 0
    except Exception:
        return False


# ── Workspace maintenance ───────────────────────────────────────────────────

def reset_workspace() -> None:
    """Restore the workspace to the seed project and (re)init a git repo with an
    initial commit, so git_status/git_diff reflect the agent's changes."""
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    shutil.copytree(SEED, WORKSPACE)
    env = dict(GIT_AUTHOR_NAME="demo", GIT_AUTHOR_EMAIL="demo@example.com",
               GIT_COMMITTER_NAME="demo", GIT_COMMITTER_EMAIL="demo@example.com")
    import os
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "commit", "-q", "-m", "seed"]):
        subprocess.run(cmd, cwd=str(WORKSPACE), env={**os.environ, **env},
                       capture_output=True, text=True)
