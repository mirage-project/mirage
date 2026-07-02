r"""
Sandboxed tool runtime for the constrained coding-agent demo.

Executes the five tool calls (list/read/write/edit/bash) INSIDE one workspace
folder and enforces every safety rule independently of the grammar — so the
sandbox holds even if the structural tag is bypassed entirely:

  * filenames must match ^[A-Za-z0-9_\-]+\.(txt|py)$ — no absolute paths, no
    ``..``, no path separators, no traversal;
  * the resolved path must stay inside the workspace;
  * only .txt / .py extensions are supported;
  * .py writes/edits must pass ``ast.parse`` before touching disk;
  * bash only runs ``python <filename.py>`` for a file in the workspace, via a
    ``subprocess`` with no shell, a fixed cwd, captured output and a timeout.

The workspace is resolved relative to this file (``demo/coding_agent/workspace``),
so the enforcement does not depend on the process's current directory.
"""

import ast
import json
import re
import subprocess
from pathlib import Path

WORKSPACE = (Path(__file__).parent / "workspace").resolve()
FILENAME_RE = re.compile(r"^[A-Za-z0-9_\-]+\.(txt|py)$")
BASH_RE = re.compile(r"^python ([A-Za-z0-9_\-]+\.py)$")
SUPPORTED_EXTS = (".txt", ".py")
BASH_TIMEOUT_S = 10

# Initial demo files; also used to (re)seed the workspace for a clean run.
SEED_FILES = {
    "main.py": 'def main():\n    print("old message")\n\n'
               'if __name__ == "__main__":\n    main()\n',
    "notes.txt": "This file contains the message that main.py should print.\n",
}


class ToolError(Exception):
    """A call violated a sandbox rule; the message is surfaced to the model."""


# ── Path safety ─────────────────────────────────────────────────────────────

def resolve_workspace_path(filename: str) -> Path:
    """Validate ``filename`` and return an absolute path guaranteed to live
    inside the workspace. Rejects anything that is not a bare ``NAME.txt`` /
    ``NAME.py`` (which already excludes absolute paths, ``..`` and separators),
    then re-checks containment after resolving as defense in depth."""
    if not isinstance(filename, str) or not FILENAME_RE.match(filename):
        raise ToolError(
            f"illegal filename {filename!r}: must match {FILENAME_RE.pattern}")
    path = (WORKSPACE / filename).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ToolError(f"path escapes workspace: {filename!r}")
    return path


def _check_ext(path: Path) -> None:
    if path.suffix not in SUPPORTED_EXTS:
        raise ToolError(
            f"unsupported extension {path.suffix!r}; only {SUPPORTED_EXTS}")


def _validate_python(source: str, name: str) -> None:
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise ToolError(f"{name}: python syntax error: {e}") from e


# ── Tools ───────────────────────────────────────────────────────────────────

def tool_list(args: dict) -> str:
    """`ls` the workspace. The only valid payload is ``{}``."""
    if args != {}:
        raise ToolError("list takes no arguments; payload must be {}")
    names = sorted(p.name for p in WORKSPACE.iterdir())
    return "\n".join(names) if names else "(empty)"


def tool_read(args: dict) -> str:
    """Read a .txt/.py file from the workspace."""
    path = resolve_workspace_path(args["filename"])
    _check_ext(path)
    if not path.is_file():
        raise ToolError(f"no such file: {path.name}")
    return path.read_text()


def tool_write(args: dict) -> str:
    """Create or overwrite a .txt/.py file. .py content must parse."""
    path = resolve_workspace_path(args["filename"])
    _check_ext(path)
    content = args["content"]
    if not isinstance(content, str):
        raise ToolError("content must be a string")
    if path.suffix == ".py":
        _validate_python(content, path.name)
    path.write_text(content)
    return f"wrote {path.name} ({len(content)} bytes)"


def tool_edit(args: dict) -> str:
    """Replace the exact ``old`` text with ``new`` (must occur exactly once).
    For .py files the result must still parse."""
    path = resolve_workspace_path(args["filename"])
    _check_ext(path)
    if not path.is_file():
        raise ToolError(f"no such file: {path.name}")
    old, new = args["old"], args["new"]
    if not isinstance(old, str) or not isinstance(new, str):
        raise ToolError("old and new must be strings")
    text = path.read_text()
    count = text.count(old)
    if count == 0:
        raise ToolError(f"`old` text not found in {path.name}")
    if count > 1:
        raise ToolError(f"`old` text is ambiguous: found {count}x in {path.name}")
    updated = text.replace(old, new, 1)
    if path.suffix == ".py":
        _validate_python(updated, path.name)
    path.write_text(updated)
    return f"edited {path.name}"


def tool_bash(args: dict) -> str:
    """Run exactly ``python <filename.py>`` in the workspace (no shell)."""
    command = args["command"]
    m = BASH_RE.match(command) if isinstance(command, str) else None
    if not m:
        raise ToolError("bash only allows `python <filename.py>`")
    path = resolve_workspace_path(m.group(1))  # re-validate inside workspace
    if path.suffix != ".py":
        raise ToolError("bash can only run .py files")
    if not path.is_file():
        raise ToolError(f"no such file: {path.name}")
    proc = subprocess.run(
        ["python", path.name], cwd=str(WORKSPACE),
        capture_output=True, text=True, timeout=BASH_TIMEOUT_S)
    body = proc.stdout + (f"\n[stderr]\n{proc.stderr}" if proc.stderr else "")
    return f"exit={proc.returncode}\n{body}".rstrip()


_TOOLS = {"list": tool_list, "read": tool_read, "write": tool_write,
          "edit": tool_edit, "bash": tool_bash}


# ── Parse + dispatch ────────────────────────────────────────────────────────

_CALL_RE = re.compile(r"<function=(\w+)>(.*?)</function>", re.S)


def parse_tool_calls(text: str):
    """Extract ``(name, args_dict)`` pairs from model output. Malformed JSON or
    an unknown tool raises ToolError so the caller can report it."""
    calls = []
    for name, raw in _CALL_RE.findall(text):
        raw = raw.strip() or "{}"
        try:
            args = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ToolError(f"{name}: bad JSON args {raw!r}: {e}") from e
        if name not in _TOOLS:
            raise ToolError(f"unknown tool {name!r}")
        calls.append((name, args))
    return calls


def dispatch(name: str, args: dict) -> str:
    """Execute one tool call, converting any ToolError into an ``ERROR: ...``
    string so the agent loop can feed it back to the model instead of crashing."""
    if name not in _TOOLS:
        return f"ERROR: unknown tool {name!r}"
    try:
        return _TOOLS[name](args)
    except ToolError as e:
        return f"ERROR: {e}"
    except KeyError as e:
        return f"ERROR: missing argument {e}"
    except subprocess.TimeoutExpired:
        return f"ERROR: `{name}` timed out after {BASH_TIMEOUT_S}s"
    except Exception as e:  # pragma: no cover - unexpected
        return f"ERROR: {type(e).__name__}: {e}"


# ── Workspace maintenance ───────────────────────────────────────────────────

def reset_workspace() -> None:
    """Restore the workspace to exactly the seed files, for a reproducible run."""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    for p in WORKSPACE.iterdir():
        if p.is_file():
            p.unlink()
    for name, content in SEED_FILES.items():
        (WORKSPACE / name).write_text(content)
