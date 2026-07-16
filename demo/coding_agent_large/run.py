"""Large constrained coding-agent demo — XGrammar structural tags + an 11-tool
sandboxed runtime, driven through the MPK megakernel.

The model REASONS in free text and emits one or more tool calls per turn as
``<function=NAME>{json args}</function>`` (the ``triggered_tags`` grammar). Each
turn's calls are executed inside a locked-down workspace and the results are fed
back, looping until the model calls ``finish`` (or pytest is green). The grammar
constrains the *shape* of every call — that it names a real tool and its args fit
the tool's JSON schema; agent_runtime.py independently enforces all filesystem
and execution safety.

Tools live as JSON files under ``tools/`` (drop in a new file to add a tool — no
code changes) and are loaded with ``mirage.mpk.create_tools``.

Run it two ways:

    # Live: constrained decoding through the MPK megakernel + Qwen3-8B (default; --model to change)
    CUDA_VISIBLE_DEVICES=0 python demo/coding_agent_large/run.py

    # Scripted: no GPU/model — drives the SAME sandbox through a fixed trajectory
    # that solves the task, showing the flow + safety:
    python demo/coding_agent_large/run.py --scripted
"""

import argparse
import json
import pathlib
import sys
import textwrap

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))          # for `import agent_runtime`
sys.path.insert(0, "python")           # for `import mirage` when run from repo root

import agent_runtime as R              # noqa: E402

TOOLS_DIR = HERE / "tools"
MAX_TURNS = 60          # runaway backstop; the real stop is pass-or-context-limit

TASK = ("The workspace has a slow, pure-Python `batched_topk_filter` in "
        "topk_filter.py (keep each row's top-k scores, set the rest to -inf). "
        "Optimize it with numpy so the whole test suite — correctness AND speed — "
        "passes under `python -m pytest`, then call finish. Do not edit the tests.")

SYSTEM = textwrap.dedent("""\
    You are an autonomous coding agent working inside a sandboxed workspace.
    Think briefly, then ACT by emitting one or more tool calls, each as
    <function=NAME>{json args}</function>. You will see every tool's result
    before your next turn. You MUST follow the guidelines.

    Guidelines:
      - Explore the files before editing; run the tests to see what fails.
      - There may be MORE THAN ONE problem — fix every failing test.
      - edit_file is for SMALL snippets (`old` must match once). To rewrite a
        whole function or file, use write_file with the full new contents — do
        not try to edit_file the entire file.
      - If a call returns an ERROR, change your approach rather than repeating
        the same call.
      - ALWAYS call run_pytest immediately after a write_file or edit_file — your
        very next action must be run_pytest to check the change. Never skip it.
      - Only after a run_pytest shows ALL tests passing, call finish with a
        summary of your changes (that ends the session). Never call finish
        without a preceding run_pytest that passed.""")


# ── display (Codex / Claude Code style) ─────────────────────────────────────

_TTY = sys.stdout.isatty()
def _sgr(code: str) -> str:
    return code if _TTY else ""
DIM, BOLD, GREEN, RED, RESET = (_sgr(c) for c in
    # DIM is an explicit dark gray (256-color) rather than the faint attribute,
    # so results/chrome read as a darker gray on truecolor terminals.
    ("\033[38;5;239m", "\033[1m", "\033[32m", "\033[31m", "\033[0m"))


def _fmt_args(args: dict, maxv: int = 56) -> str:
    """Render call args compactly: a single arg as just its value, else key=value."""
    def show(x):
        s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
        s = s.replace("\n", "⏎")
        return s if len(s) <= maxv else s[:maxv - 1] + "…"
    if not args:
        return ""
    if len(args) == 1:
        return show(next(iter(args.values())))
    return ", ".join(f"{k}={show(v)}" for k, v in args.items())


def _print_result(result: str, max_lines: int = 6) -> None:
    """Print a tool result branched under a ⎿ connector, long output truncated."""
    color = RED if result.startswith("ERROR") else DIM
    rows = result.splitlines() or ["(ok)"]
    for i, ln in enumerate(rows[:max_lines]):
        print(f"  {color}{'⎿' if i == 0 else ' '}{RESET} {DIM}{ln}{RESET}")
    if len(rows) > max_lines:
        print(f"    {DIM}… +{len(rows) - max_lines} lines{RESET}")


def _print_reasoning(text: str) -> None:
    """Show the model's free-text reasoning (before the first tool call) as prose."""
    pre = text.split("<function=")[0].strip()
    if pre:
        print()
        for para in pre.splitlines():
            print(textwrap.fill(para, width=88) if para.strip() else "")


def _print_final_pytest(reason: str = None, turns: int = None) -> bool:
    """Run the suite one last time and print its full results + the [PERF]
    performance comparison, with the summary/perf lines highlighted."""
    out = R.dispatch("run_pytest", {})
    ok = out.startswith("exit=0")
    mark = f"{GREEN}✔{RESET}" if ok else f"{RED}■{RESET}"
    header = reason or ("all tests pass" if ok else "tests still failing")
    suffix = f"  {DIM}({turns} turns){RESET}" if turns else ""
    print(f"\n{mark} {BOLD}{header}{RESET}{suffix}")
    print(f"\n{BOLD}Final test results{RESET}")
    for ln in out.splitlines():
        if ln.startswith("exit="):
            continue
        if ln.startswith("[PERF]"):
            c = BOLD + GREEN
        elif "passed" in ln and "failed" not in ln and "error" not in ln.lower():
            c = GREEN
        elif "failed" in ln or "error" in ln.lower():
            c = RED
        else:
            c = DIM
        print(f"  {c}{ln}{RESET}")
    return ok


def execute_turn(text: str, verbose: bool = True):
    """Run every tool call in a turn's text. Returns (feedback, done)."""
    try:
        calls = R.parse_tool_calls(text)
    except R.ToolError as e:
        return f"ERROR: {e}", False
    if not calls:
        return "(no tool call emitted — reply with at least one <function=...> call)", False
    lines, done = [], False
    for name, args in calls:
        result = R.dispatch(name, args)      # finish is a real tool: it runs pytest
        ok = not result.startswith("ERROR")
        if verbose:                          # ⏺ tool(args)  then  ⎿ result
            print()                          # line break before each tool call
            dot = (GREEN if ok else RED) + "⏺" + RESET
            print(f"{dot} {BOLD}{name}{RESET}({DIM}{_fmt_args(args)}{RESET})")
            _print_result(result)
        lines.append(f"[{name}]\n{result}")
        if name == "finish" and ok:
            done = True                      # finish succeeded → tests verified green
            break
    return "\n".join(lines), done


# ── live model driver ───────────────────────────────────────────────────────

def run_live(model: str = "Qwen/Qwen3-14B") -> None:
    import threading
    import time

    from mirage.engine import ModelRunner, RunnerConfig
    from mirage.mpk import create_tools, structured_tools as T

    # Always start from the fresh slow seed so the agent has a real task; its
    # optimized result persists in the workspace *after* the run (nothing resets
    # at the end) — reusing a prior run's output would pre-solve the task.
    R.reset_workspace()
    tools = create_tools(str(TOOLS_DIR))
    print(f"{DIM}building {model} megakernel ({len(tools)} tools)…{RESET}")

    CTX = 6144           # per-request context budget (prompt + generation)
    MIN_GEN = 256        # room a turn needs to reason + emit its calls
    # Qwen3-8B is the validated single-GPU config for this ModelRunner +
    # constrained-decoding path (all demo/qwen3 demos use it). Larger dense
    # variants like Qwen3-14B are registered but NOT validated here — their
    # shapes hit an SM100 TMA-alignment bug (a split-K linear dim that isn't
    # 16B-aligned). Qwen3-30B-A3B is smarter but needs the separate multi-GPU
    # demo path, not this engine. max_num_pages just needs to cover CTX.
    cfg = RunnerConfig(model=model, vocab_size=153600,
                       enable_constrained_decoding=True, do_sample=True,
                       max_num_batched_requests=1, max_num_batched_tokens=8,
                       max_seq_length=CTX, max_num_pages=32, page_size=4096)
    runner = ModelRunner(cfg)
    rt, tok = runner.runtime, runner.tokenizer
    threading.Thread(target=runner, daemon=True).start()
    time.sleep(2)

    # After compilation: show the prompt (a clean divider from the build noise),
    # then start the agent loop.
    print(f"\n{DIM}model: {model}  ·  {len(tools)} tools: "
          f"{', '.join(sorted(t['name'] for t in tools))}{RESET}")
    print(f"\n{BOLD}>{RESET} " + textwrap.fill(TASK, width=86, subsequent_indent="  "))
    print(f"\n{DIM}{'─' * 72}{RESET}")

    conversation = [
        {"role": "system", "content": SYSTEM + "\n\n" + T.tools_system_prompt(tools)},
        {"role": "user", "content": TASK}]

    # Keep going until the tests pass or the conversation no longer fits the
    # context window; MAX_TURNS is only a runaway backstop.
    reason, turn = None, 0
    while True:
        prompt = tok.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt")[0]
        used = prompt.shape[0]
        if used + MIN_GEN >= CTX:
            reason = f"context limit — prompt is {used}/{CTX} tokens"
            break
        if turn >= MAX_TURNS:
            reason = f"backstop ({MAX_TURNS} turns)"
            break

        rid = turn
        rt.set_request_grammar(rid, triggered_tags=tools)
        rt.submit(rid, prompt)
        try:
            row, final = rt.wait_for_request(rid, timeout=300)
        except TimeoutError:
            reason = "request timed out"
            break
        text = tok.decode(rt.read_tokens_at_row(row, final)[prompt.shape[0]:],
                          skip_special_tokens=True).strip()

        print(f"\n{DIM}─── turn {turn} · {used}/{CTX} tokens ───{RESET}")
        _print_reasoning(text)
        conversation.append({"role": "assistant", "content": text})

        feedback, finished = execute_turn(text)
        if finished:   # the finish tool itself re-ran pytest and it was green
            conversation.append({"role": "user", "content": "Tool results:\n" + feedback})
            reason = "model called finish (verified green by re-running pytest)"
            break
        # A rejected finish already returned its ERROR in feedback. If the tests
        # are green but the model hasn't ended yet, nudge it to finish.
        if R.tests_pass():
            feedback += ('\n[note] all tests pass. Call '
                         '<function=finish>{"summary": "..."}</function> with a '
                         'summary of the changes you made to end the session.')
        conversation.append({"role": "user", "content": "Tool results:\n" + feedback})
        turn += 1

    _print_final_pytest(reason, turn + 1)
    print(f"\n{DIM}edited file: {R.WORKSPACE / 'topk_filter.py'}{RESET}")
    rt.shutdown()


# ── scripted driver (no GPU/model) ──────────────────────────────────────────

# The numpy solution the scripted run writes (what a good agent would produce).
SOLUTION = '''"""Batched top-k filter: keep each row's top-k scores, set the rest to -inf."""

import numpy as np


def batched_topk_filter(scores, k):
    scores = np.asarray(scores, dtype=float)
    n, v = scores.shape
    if k >= v:
        return scores.copy()
    kth = np.partition(scores, v - k, axis=1)[:, v - k]   # k-th largest per row
    return np.where(scores >= kth[:, None], scores, -np.inf)
'''


def _call(name: str, **args) -> str:
    return f"<function={name}>{json.dumps(args)}</function>"


def run_scripted() -> None:
    R.reset_workspace()
    print(f"{DIM}scripted (no model)  ·  {len(R.TOOL_NAMES)} tools  ·  "
          f"fixed solving trajectory{RESET}")
    print(f"\n{BOLD}>{RESET} " + textwrap.fill(TASK, width=86, subsequent_indent="  "))
    print(f"\n{DIM}{'─' * 72}{RESET}")

    trajectory = [
        _call("list_dir", path="."),
        _call("read_file", path="topk_filter.py"),
        _call("read_file", path="test_topk_filter.py"),
        _call("run_pytest"),                                 # 3 pass, 2 fail
        _call("write_file", path="topk_filter.py", content=SOLUTION),
        _call("run_pytest"),                                 # 5 pass
        _call("git_diff"),
        _call("finish", summary="Vectorized batched_topk_filter with numpy "
              "(np.partition + np.where); correctness and speed tests pass."),
    ]

    for turn, call in enumerate(trajectory):
        print(f"\n{DIM}─── step {turn} ───{RESET}")
        _, done = execute_turn(call)
        if done:
            break

    print(f"\n{DIM}─── safety: the runtime rejects these even if the grammar were "
          f"bypassed ───{RESET}")
    for label, call in [
        ("absolute path",   _call("read_file", path="/etc/passwd")),
        ("path traversal",  _call("read_file", path="../../secret.txt")),
        ("bad extension",   _call("write_file", path="evil.sh", content="rm -rf /")),
        ("broken python",   _call("write_file", path="broken.py", content="def (")),
        ("edit the tests",  _call("edit_file", path="test_topk_filter.py",
                                  old="assert", new="pass  # assert")),
    ]:
        print(f"\n{BOLD}{label}{RESET}")
        execute_turn(call)

    _print_final_pytest()
    print(f"\n{DIM}edited file: {R.WORKSPACE / 'topk_filter.py'}{RESET}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scripted", action="store_true",
                   help="run without a model; drive the sandbox directly")
    p.add_argument("--model", default="Qwen/Qwen3-14B",
                   help="HF model id (Qwen3 family). Qwen3-8B is the validated "
                        "single-GPU config; larger dense variants may hit an "
                        "SM100 TMA-alignment bug.")
    args = p.parse_args()
    if args.scripted:
        run_scripted()
    else:
        run_live(args.model)


if __name__ == "__main__":
    main()
