"""
Agentic coding assistant (à la Claude Code / Codex) driven by a JSON tool
registry + XGrammar structural tags. The model reasons in FREE text and, the
moment it emits ``<function=``, xgrammar switches it to CONSTRAINED — the rest
must be a schema-valid call to one of the registered tools.

Tools live as JSON files under ``tool_registry/``; add a tool by dropping in a
new file (registry→structural-tag helpers live in
``mirage.mpk.structured_tools``). Run:

    CUDA_VISIBLE_DEVICES=1 python demo/constrained_decoding/agent_tools.py
"""

import json
import pathlib
import re
import sys
import threading
import time

sys.path.insert(0, "python")
import torch  # noqa: E402,F401

from mirage.mpk import create_tools, structured_tools as T  # noqa: E402

HERE = pathlib.Path(__file__).parent
USER = ("Add a `greet(name)` function to utils.py that prints a greeting, then "
        "run the test suite with pytest.")


def main():
    from mirage.engine import ModelRunner, RunnerConfig

    # tool JSON files → registry → (optional) routing
    tools = create_tools(HERE / "tool_registry")
    tools = T.select_tools(tools, USER)          # placeholder: returns all
    print(f"registry: {len(tools)} tools ->", [t["name"] for t in tools])

    cfg = RunnerConfig(model="Qwen/Qwen3-8B", vocab_size=153600,
                       enable_constrained_decoding=True, do_sample=True,
                       max_num_batched_requests=1, max_num_batched_tokens=8,
                       max_seq_length=512, max_num_pages=32, page_size=4096)
    runner = ModelRunner(cfg)              # xgrammar auto-initialized from the flag
    rt, tok = runner.runtime, runner.tokenizer
    rt.set_request_grammar(0, triggered_tags=tools)   # free text until <function=, then constrained

    prompt = tok.apply_chat_template(
        [{"role": "system", "content": T.tools_system_prompt(tools)},
         {"role": "user", "content": USER}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_tensors="pt")[0]

    threading.Thread(target=runner, daemon=True).start()
    time.sleep(2)
    rt.submit(0, prompt)
    row, final = rt.wait_for_request(0, timeout=180)
    out = tok.decode(rt.read_tokens_at_row(row, final)[prompt.shape[0]:],
                     skip_special_tokens=True)

    print("\n===== GENERATED =====\n" + out)
    print("\n===== PARSED TOOL CALLS =====")
    names = {t["name"] for t in tools}
    for name, args in re.findall(r"<function=(\w+)>(.*?)</function>", out, re.S):
        parsed = json.loads(args)
        print(f"  {name}({json.dumps(parsed)})  known_tool={name in names}")
    rt.shutdown()                          # grammar auto-released on completion


if __name__ == "__main__":
    main()
