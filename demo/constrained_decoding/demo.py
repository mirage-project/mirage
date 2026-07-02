"""
Constrained decoding (xgrammar) demo for MPK.

Two modes:

  --simulate   (default) A self-contained decode loop that drives the real
               StructuredGenerationManager over synthetic logits, masking each
               step exactly as the GPU apply_token_bitmask task does, and shows
               that every generated token is grammar-valid.  Runs anywhere with
               xgrammar installed — no model, no megakernel.

  --model NAME The real online_pinned serving path (requires the model + a
               megakernel built with apply_token_bitmask_layer wired in before
               the argmax/sampling layer).  Shown as the usage template; see the
               NOTE in run_online_pinned().

Examples:
  python demo.py                       # simulated JSON-grammar generation
  python demo.py --grammar email       # simulated regex (email) generation
  python demo.py --model Qwen/Qwen3-8B # real serving (needs model + wiring)
"""

import argparse
import importlib.util
import pathlib
import re

import torch

HERE = pathlib.Path(__file__).parent
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
STRUCTURED_PY = REPO_ROOT / "python" / "mirage" / "mpk" / "structured.py"


def _load_structured():
    spec = importlib.util.spec_from_file_location("mpk_structured", STRUCTURED_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.StructuredGenerationManager


# ── Simulated decode loop ──────────────────────────────────────────────────
#
# A tiny character-level "model": the vocab is single characters and the
# "logits" are random, so the ONLY thing keeping the output well-formed is the
# grammar mask.  This drives the actual StructuredGenerationManager and applies
# the mask the same way apply_token_bitmask_sm100.cuh does (allowed → keep,
# disallowed → -inf), then greedily samples.

def _char_vocab():
    # printable chars sufficient for small JSON / email grammars
    chars = list(' \t\n{}[]":,.@-_0123456789'
                 'abcdefghijklmnopqrstuvwxyz'
                 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                 'truefalsn')  # ensure true/false/null letters present
    # de-dup, stable order
    seen, vocab = set(), []
    for c in chars:
        if c not in seen:
            seen.add(c)
            vocab.append(c)
    return vocab


def run_simulate(grammar_kind: str, max_tokens: int, seed: int):
    import xgrammar as xgr

    StructuredGenerationManager = _load_structured()
    torch.manual_seed(seed)

    vocab = _char_vocab()
    V = len(vocab)
    words = (V + 31) // 32
    ti = xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.RAW, vocab_size=V)

    if grammar_kind == "email":
        spec = dict(regex=r"[a-z0-9_]+@[a-z]+\.[a-z]+")
        desc = "regex e-mail"
    else:
        spec = dict(ebnf=r'''root   ::= "{" ws "\"id\"" ws ":" ws number ws "}"
number ::= [0-9]+
ws     ::= [ ]*''')
        desc = "EBNF json-ish object"

    # Mock pinned buffers (CPU tensors == page-locked memory in the real path).
    n_req, max_seq = 1, max_tokens + 8
    token_bitmask = torch.full((n_req, words), -1, dtype=torch.int32)
    mask_seq = torch.zeros(n_req, dtype=torch.int32)
    flag = torch.zeros(1, dtype=torch.int32)
    tokens = torch.zeros(n_req, max_seq, dtype=torch.int64)
    pinned_step = torch.zeros(n_req, dtype=torch.int32)
    row_map = {0: 0}

    mgr = StructuredGenerationManager(
        token_bitmask=token_bitmask, mask_seq=mask_seq, constrained_flag=flag,
        tokens=tokens, pinned_step=pinned_step,
        find_row_for_rid=lambda rid: row_map.get(rid, -1),
    )
    mgr.init_xgrammar(tokenizer_info=ti, vocab_size=V)
    mgr.set_request_grammar(0, **spec)

    print(f"grammar: {desc}")
    print(f"vocab size: {V}, constrained_flag={flag[0].item()}")

    row = 0
    out_ids = []
    for s in range(max_tokens):
        pinned_step[row] = s
        mgr.tick()  # host: accept prev token, fill mask for step s, publish seq

        # device side: a random "logit" vector, masked exactly like the kernel.
        logits = torch.randn(V)
        if flag[0].item() == 1 and mask_seq[row].item() >= s:
            row_mask = token_bitmask[row]
            allowed = torch.tensor(
                [bool((row_mask[i // 32].item() >> (i % 32)) & 1) for i in range(V)]
            )
            logits = torch.where(allowed, logits, torch.full_like(logits, float("-inf")))

        tok = int(torch.argmax(logits).item())
        tokens[row, s] = tok
        out_ids.append(tok)

        # stop if the grammar is satisfied (the matcher accepts this token first
        # inside the next tick; check terminal by peeking a fork)
        text = "".join(vocab[t] for t in out_ids)
        if mgr._matchers.get(row) is None:
            break
        # crude termination: a fresh matcher replay tells us if we're complete
        if _is_complete(mgr, spec, out_ids):
            break

    text = "".join(vocab[t] for t in out_ids)
    print(f"generated ({len(out_ids)} tokens): {text!r}")
    if grammar_kind != "email":
        import json
        try:
            json.loads(text)
            print("✓ output parses as JSON")
        except Exception as e:
            print(f"(note) JSON parse: {e}")
    return text


def _is_complete(mgr, spec, out_ids) -> bool:
    """Replay the token sequence on a fresh matcher and report completion."""
    import xgrammar as xgr
    if "regex" in spec:
        cg = mgr._compiler.compile_regex(spec["regex"])
    else:
        cg = mgr._compiler.compile_grammar(spec["ebnf"])
    m = xgr.GrammarMatcher(cg)
    for t in out_ids:
        if not m.accept_token(t):
            return True  # rejected → can't continue; treat as done
    return m.is_terminated()


# ── Real online_pinned serving (usage template) ────────────────────────────

def run_online_pinned(model: str, vocab_size: int, sample: bool, tool_call: bool):
    """The real serving path. The Qwen3 builder inserts
    ``apply_token_bitmask_layer`` before argmax when
    ``enable_constrained_decoding=True``.  ``vocab_size`` must be the *padded*
    logit width the masking task sees (153600 for Qwen3).

    With ``tool_call`` a structural tag is used: the model generates FREE
    (unconstrained) text and, the moment it emits the trigger ``<function=``,
    xgrammar switches it to CONSTRAINED — the rest must be a schema-valid call to
    one of the tools loaded from ``tool_registry/`` (the ONLY tools the model may
    call). Otherwise a bounded-integer JSON schema is enforced.
    """
    import json
    import threading
    from mirage.engine import ModelRunner, RunnerConfig

    cfg = RunnerConfig(model=model, vocab_size=vocab_size,
                       enable_constrained_decoding=True, do_sample=sample,
                       max_num_batched_requests=1, max_seq_length=512)
    runner = ModelRunner(cfg)              # xgrammar auto-initialized from the flag
    rt, tok = runner.runtime, runner.tokenizer

    if tool_call:
        from mirage.mpk import create_tools, structured_tools as T
        tools = create_tools(HERE / "tool_registry")   # only the registry tools
        names = {t["name"] for t in tools}
        rt.set_request_grammar(0, triggered_tags=tools)
        print(f"registry: {len(tools)} tools ->", sorted(names))
        msgs = [{"role": "system", "content": T.tools_system_prompt(tools)},
                {"role": "user", "content": "Show me the git status of the repo."}]
        prompt = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False, return_tensors="pt")[0]
    else:
        schema = ('{"type":"object","properties":{"id":'
                  '{"type":"integer","minimum":0,"maximum":999999}},"required":["id"]}')
        rt.set_request_grammar(0, json_schema=schema, any_whitespace=False)
        prompt = tok("Give me a JSON object: ", return_tensors="pt").input_ids[0]

    threading.Thread(target=runner, daemon=True).start()  # drives mpk()
    rt.submit(0, prompt)
    row, final = rt.wait_for_request(0, timeout=120)
    out = tok.decode(rt.read_tokens_at_row(row, final)[prompt.shape[0]:],
                     skip_special_tokens=True)
    print("OUTPUT:", repr(out))
    if tool_call:
        calls = re.findall(r"<function=(\w+)>(.*?)</function>", out, re.S)
        print("free text before call:",
              repr(out[:out.index("<function=")]) if calls else "(none)")
        for name, args in calls:
            print(f"tool call: {name}({json.loads(args)})  known_tool={name in names}")
        if not calls:
            print("tool call: (no call emitted)")
    rt.shutdown()                          # grammar auto-released on completion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None,
                   help="HF model name → real online_pinned path (needs wiring)")
    p.add_argument("--vocab-size", type=int, default=None,
                   help="model logit dim (required with --model)")
    p.add_argument("--grammar", choices=["json", "email"], default="json",
                   help="simulated-mode grammar")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample", action="store_true",
                   help="sample (Gumbel-max) instead of greedy argmax")
    p.add_argument("--structural-tag", action="store_true",
                   help="tool-call structural tag: free text until <function=, "
                        "then schema-constrained (unconstrained→constrained switch)")
    args = p.parse_args()

    if args.model is not None:
        assert args.vocab_size is not None, "--vocab-size is required with --model"
        run_online_pinned(args.model, args.vocab_size, args.sample,
                          args.structural_tag)
    else:
        run_simulate(args.grammar, args.max_tokens, args.seed)


if __name__ == "__main__":
    main()
