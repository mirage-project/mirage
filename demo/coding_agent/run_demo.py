"""
Constrained coding-agent demo — XGrammar structural tags + a sandboxed runtime.

The model is constrained to emit ONLY the five tool calls
(list/read/write/edit/bash), tool-only, newline-separated. Each turn the calls
are executed inside a locked-down workspace and the results are fed back,
looping until the model runs ``python main.py``. The grammar constrains the
*shape* of every call; tool_runtime.py independently enforces all filesystem
and execution safety.

The demo folder is ``demo/coding_agent/`` and the sandbox is
``demo/coding_agent/workspace/`` (resolved relative to this file).

Run it two ways:

    # Live: constrained decoding through the MPK megakernel + Qwen3-8B
    CUDA_VISIBLE_DEVICES=1 python demo/coding_agent/run_demo.py

    # Scripted: no GPU/model — drives the SAME sandbox runtime through a fixed
    # happy path plus adversarial calls, to show the safety enforcement:
    python demo/coding_agent/run_demo.py --scripted
"""

import argparse
import json
import pathlib
import sys
import textwrap

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

import tool_runtime as R                         # noqa: E402
from structural_tags import build_structural_tag  # noqa: E402

MAX_TURNS = 6

TASK = ('Update main.py so it prints "new message" instead of "old message", '
        "then run it with `python main.py` to confirm the change.")

SYSTEM = textwrap.dedent("""\
    You are a coding agent working inside a locked sandbox folder. Respond with
    exactly ONE tool call per message — no prose — as
    <function=NAME>{json args}</function>. You will be shown its result before
    your next step. The available tools are:

      <function=list>{}</function>
      <function=read>{"filename":"main.py"}</function>
      <function=write>{"filename":"main.py","content":"..."}</function>
      <function=edit>{"filename":"main.py","old":"...","new":"..."}</function>
      <function=bash>{"command":"python main.py"}</function>

    Filenames are bare names ending in .txt or .py inside the workspace; bash may
    only run `python <file>.py`. List the files, read what you need, make the
    change, then run `python main.py`.""")


# ── formatting helpers ──────────────────────────────────────────────────────

def _indent(text: str, n: int = 3) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.splitlines()) or (pad + "(empty)")


def _compact(args: dict, width: int = 88) -> str:
    s = json.dumps(args, ensure_ascii=False)
    return s if len(s) <= width else s[:width - 1] + "…"


def _run_tool_call(text: str):
    """Execute the single tool call in ``text`` (the grammar allows exactly one).
    Print it, and return (result_str, ran_main)."""
    try:
        calls = R.parse_tool_calls(text)
    except R.ToolError as e:
        print(_indent(f"(unparseable) ERROR: {e}"))
        return f"ERROR: {e}", False
    if not calls:
        print(_indent("(no tool call emitted)"))
        return "(no tool call emitted)", False

    name, args = calls[0]
    result = R.dispatch(name, args)
    print(f"   $ {name}({_compact(args)})")
    print(_indent(result, 5))
    ran_main = (name == "bash" and isinstance(args, dict)
                and args.get("command") == "python main.py"
                and not result.startswith("ERROR"))
    return f"[{name}] {result}", ran_main


# ── live model driver ───────────────────────────────────────────────────────

def run_live() -> None:
    import threading
    import time

    from mirage.engine import ModelRunner, RunnerConfig

    R.reset_workspace()
    stag = build_structural_tag()

    cfg = RunnerConfig(
        model="Qwen/Qwen3-8B", vocab_size=153600,
        enable_constrained_decoding=True, do_sample=True,
        max_num_batched_requests=1, max_num_batched_tokens=8,
        max_seq_length=1024, max_num_pages=64, page_size=4096)
    runner = ModelRunner(cfg)              # xgrammar auto-initialized from the flag
    rt, tok = runner.runtime, runner.tokenizer
    threading.Thread(target=runner, daemon=True).start()
    time.sleep(2)

    conversation = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": TASK}]
    print("TASK:", TASK, "\n")

    for turn in range(MAX_TURNS):
        prompt = tok.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt")[0]
        rid = turn
        rt.set_request_grammar(rid, structural_tag=stag)
        rt.submit(rid, prompt)
        row, final = rt.wait_for_request(rid, timeout=180)
        text = tok.decode(rt.read_tokens_at_row(row, final)[prompt.shape[0]:],
                          skip_special_tokens=True).strip()
        # grammar for this rid is auto-released when the turn completes

        print(f"── turn {turn} ── model:")
        print(_indent(text))
        conversation.append({"role": "assistant", "content": text})

        result, ran_main = _run_tool_call(text)
        conversation.append({"role": "user", "content": "Tool result:\n" + result})
        if ran_main:
            print("\n✓ model ran `python main.py` — task complete.")
            break
    else:
        print("\n(reached MAX_TURNS without running `python main.py`)")

    print("\nfinal main.py:\n" + _indent((R.WORKSPACE / "main.py").read_text()))
    rt.shutdown()


# ── scripted driver (no GPU/model) ──────────────────────────────────────────

def run_scripted() -> None:
    R.reset_workspace()
    print("TASK:", TASK)
    print("(scripted: driving the real sandbox runtime directly, no model)\n")

    print("=== happy path: what a well-behaved agent emits ===")
    for call in [
        '<function=list>{}</function>',
        '<function=read>{"filename":"main.py"}</function>',
        '<function=read>{"filename":"notes.txt"}</function>',
        '<function=edit>{"filename":"main.py","old":"old message","new":"new message"}</function>',
        '<function=bash>{"command":"python main.py"}</function>',
    ]:
        _run_tool_call(call)

    print("\n=== safety: the runtime rejects these even though the grammar is bypassed ===")
    for label, call in [
        ("absolute path",   '<function=read>{"filename":"/etc/passwd"}</function>'),
        ("path traversal",  '<function=read>{"filename":"../../secret.txt"}</function>'),
        ("unsupported ext", '<function=read>{"filename":"main.sh"}</function>'),
        ("shell chaining",  '<function=bash>{"command":"python main.py && rm -rf /"}</function>'),
        ("non-python bash", '<function=bash>{"command":"ls"}</function>'),
        ("cat via bash",    '<function=bash>{"command":"cat main.py"}</function>'),
        ("bad python write",'<function=write>{"filename":"broken.py","content":"def ("}</function>'),
    ]:
        print(f"\n[{label}]")
        _run_tool_call(call)

    print("\nfinal main.py:\n" + _indent((R.WORKSPACE / "main.py").read_text()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scripted", action="store_true",
                   help="run without a model; drive the sandbox runtime directly")
    args = p.parse_args()
    (run_scripted if args.scripted else run_live)()


if __name__ == "__main__":
    main()
