"""
Tool registry → XGrammar structural tags (composable API), scalable to hundreds
of tools.

Reusable helpers for turning a registry of tool definitions into an xgrammar
``StructuralTag`` for constrained tool-calling. The friendly path is
:func:`create_tools` + ``OnlinePinnedRuntime.set_request_grammar``::

    from mirage.mpk import create_tools
    tools = create_tools("path/to/tool_registry")     # dir of *.json …
    tools = create_tools([json_str1, json_str2])      # … or a list of JSON strings
    rt.set_request_grammar(rid, triggered_tags=tools) # free text + calls
    #   ...tags_with_separator=tools  (tool-only) | dispatch=tools (many tools)
    #   ...add model="qwen_3" to emit the tools in the model's native format

The lower-level flow (if you need the ``StructuralTag`` object directly):
    tool JSON files on disk / JSON strings
      → create_tools()               (→ list of tool dicts)
      → select_tools()               (optional routing to a relevant subset)
      → build_*_structural_tag()     (tools → xgrammar StructuralTag)
      → GrammarCompiler.compile_structural_tag(tag)  (constrained decoding)

Each tool is a dict ``{name, description, parameters(JSON Schema)}``. Drop a new
JSON file anywhere under the registry folder to add a tool — no code changes.

xgrammar is imported lazily inside the builders so ``import mirage.mpk`` and
``create_tools`` work without it (only the ``build_*``/``tool_to_tag`` helpers
require xgrammar).

Notes vs. the deprecated API (StructuralTagItem / from_legacy_structural_tag):
  * We build typed pydantic Format objects (xgrammar.structural_tag.*), not raw
    dicts. If you do use a dict, it must be the full StructuralTag shape
    ``{"type": "structural_tag", "format": {...}}`` — a bare ``{"format": ...}``
    is rejected by the compiler.
  * build_model_structural_tag() takes a *template name* (e.g. "qwen_3",
    "llama", "deepseek_v3_1"), NOT a HuggingFace model id, and emits that
    model's native tool-call format (Qwen: ``<tool_call>{"name":...}</tool_call>``).
"""

from __future__ import annotations

import json
import pathlib
from typing import Iterable, List, Optional, Union

TRIGGER = "<function="  # custom tool-call prefix for the triggered/dispatch tags


# ── Registry ────────────────────────────────────────────────────────────────

def create_tools(source: Union[str, pathlib.Path, Iterable]) -> List[dict]:
    """Build a tool list from a registry directory **or** an iterable of JSON.

    ``source`` is either:
      * a directory path (str / ``Path``) — load every ``*.json`` under it
        recursively (same as :func:`load_tools`); or
      * an iterable of JSON strings and/or already-parsed tool dicts.

    Each tool must be ``{"name", "parameters"(JSON Schema)[, "description"]}``.
    """
    if isinstance(source, (str, pathlib.Path)):
        p = pathlib.Path(source)
        assert p.is_dir(), (
            f"create_tools: {source!r} is not a directory; pass a registry dir "
            f"or a list of JSON strings/dicts")
        return load_tools(p)
    tools = []
    for item in source:
        t = json.loads(item) if isinstance(item, str) else dict(item)
        assert "name" in t and "parameters" in t, f"bad tool def: {t}"
        tools.append(t)
    return tools


def load_tools(tool_registry_dir: Union[str, pathlib.Path]) -> List[dict]:
    """Load every ``*.json`` tool definition under a registry folder (recursive)."""
    root = pathlib.Path(tool_registry_dir)
    tools = [json.loads(p.read_text()) for p in sorted(root.rglob("*.json"))]
    for t in tools:
        assert "name" in t and "parameters" in t, f"bad tool def: {t}"
    return tools


def select_tools(tools: List[dict], user_message: str,
                 max_tools: Optional[int] = None) -> List[dict]:
    """Routing hook: pick the relevant tools for a request. Placeholder returns
    all tools (optionally the first ``max_tools``); replace the body with
    embedding/keyword retrieval so hundreds of tools aren't all exposed per call.
    """
    # e.g. rank by name/description overlap with user_message, then truncate.
    return tools if max_tools is None else tools[:max_tools]


# ── OpenAI-style view + prompt text ─────────────────────────────────────────

def to_openai_tools(tools: List[dict]) -> List[dict]:
    return [{"type": "function", "function": {
        "name": t["name"], "description": t.get("description", ""),
        "parameters": t["parameters"], "strict": True}} for t in tools]


def tools_system_prompt(tools: List[dict], trigger: str = TRIGGER) -> str:
    """The grammar constrains *structure*; the model still needs to know the tools
    exist. Render their names/descriptions for the system prompt."""
    lines = [f"- {t['name']}: {t.get('description', '')}" for t in tools]
    return ("You may call tools. Available tools:\n" + "\n".join(lines) +
            f"\nInvoke a tool as {trigger}NAME>{{json args}}</function>.")


# ── Structural-tag builders (custom <function=NAME> format) ─────────────────

def tool_to_tag(tool: dict, trigger: str = TRIGGER):
    """One tool → a TagFormat: begin <function=NAME>, JSON-schema body, end."""
    import xgrammar.structural_tag as st
    return st.TagFormat(
        begin=f"{trigger}{tool['name']}>",
        content=st.JSONSchemaFormat(json_schema=tool["parameters"]),
        end="</function>")


def build_triggered_tags_structural_tag(
        tools: List[dict], trigger: str = TRIGGER,
        at_least_one: bool = False, stop_after_first: bool = False):
    """Free text is allowed until ``trigger`` appears, then one of the tools'
    schemas is enforced. Interleaves prose and (multiple) tool calls."""
    import xgrammar.structural_tag as st
    return st.StructuralTag(format=st.TriggeredTagsFormat(
        triggers=[trigger], tags=[tool_to_tag(t, trigger) for t in tools],
        at_least_one=at_least_one, stop_after_first=stop_after_first))


def build_tags_with_separator_structural_tag(
        tools: List[dict], separator: str = "\n", trigger: str = TRIGGER,
        at_least_one: bool = True, stop_after_first: bool = False):
    """Tool-call-only output (no free text): one or more calls joined by
    ``separator``."""
    import xgrammar.structural_tag as st
    return st.StructuralTag(format=st.TagsWithSeparatorFormat(
        tags=[tool_to_tag(t, trigger) for t in tools], separator=separator,
        at_least_one=at_least_one, stop_after_first=stop_after_first))


def build_dispatch_structural_tag(
        tools: List[dict], trigger: str = TRIGGER, loop: bool = True):
    """Prefix-dispatch form — efficient when there are many distinct tool
    prefixes. Each rule maps ``<function=NAME>`` to (schema, "</function>")."""
    import xgrammar.structural_tag as st
    rules = [[f"{trigger}{t['name']}>", st.SequenceFormat(elements=[
                st.JSONSchemaFormat(json_schema=t["parameters"]),
                st.ConstStringFormat(value="</function>")])]
             for t in tools]
    return st.StructuralTag(format=st.DispatchFormat(rules=rules, loop=loop))


def build_model_structural_tag(
        tools: List[dict], model: str = "qwen_3", tool_choice: str = "auto",
        reasoning: bool = True):
    """Use the model's NATIVE tool-call format via get_model_structural_tag.
    ``model`` is a format name (qwen_3, qwen_3_coder, llama, deepseek_v3_1,
    kimi, harmony, …), NOT a HuggingFace model id."""
    import xgrammar as xgr
    return xgr.get_model_structural_tag(
        model=model, tools=to_openai_tools(tools),
        tool_choice=tool_choice, reasoning=reasoning)
