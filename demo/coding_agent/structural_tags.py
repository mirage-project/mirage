"""
XGrammar structural tag for the constrained coding-agent demo.

Constrains the model to emit exactly ONE of the five allowed tool calls
(list/read/write/edit/bash) per turn, tool-only (no prose), using the composable
``tags_with_separator`` format with ``stop_after_first=True``. Each turn ends
after one call so the runtime can execute it and feed the result back before the
model's next step. Built with the modern ``xgrammar.StructuralTag.from_json(...)``
API — NOT ``StructuralTagItem`` and NOT ``from_legacy_structural_tag``.

The grammar only constrains the *shape* of each call: which tools exist, that a
filename looks like ``NAME.txt`` / ``NAME.py``, and that ``bash`` is exactly
``python NAME.py``. All filesystem/execution *safety* is enforced independently
by tool_runtime.py, so the sandbox never depends on the grammar.

The ``.py`` write/edit content is a plain JSON string here (validated at runtime
with ``ast.parse``). The tag structure is kept modular so a real Python grammar
can later replace the ``content`` schema of the write/edit tags.
"""

import xgrammar as xgr

# Shared with tool_runtime.py on purpose: the grammar rejects malformed calls
# early, and the runtime re-validates against the same contract (defense in
# depth). A bare filename ending in .txt/.py — no separators, no traversal.
FILENAME_PATTERN = r"^[A-Za-z0-9_\-]+\.(txt|py)$"
BASH_PATTERN = r"^python [A-Za-z0-9_\-]+\.py$"

TOOL_NAMES = ("list", "read", "write", "edit", "bash")

_FILENAME_SCHEMA = {"type": "string", "pattern": FILENAME_PATTERN}


def _tag(begin: str, properties: dict, required: list) -> dict:
    """One ``<function=NAME> {json-schema args} </function>`` tag definition."""
    return {
        "type": "tag",
        "begin": begin,
        "content": {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
        "end": "</function>",
    }


def build_structural_tag() -> "xgr.StructuralTag":
    """Return the StructuralTag constraining output to exactly one of the five
    sandbox tool calls per turn."""
    return xgr.StructuralTag.from_json({
        "format": {
            "type": "tags_with_separator",
            "separator": "\n",
            "at_least_one": True,
            "stop_after_first": True,
            "tags": [
                _tag("<function=list>", {}, []),
                _tag("<function=read>",
                     {"filename": _FILENAME_SCHEMA}, ["filename"]),
                _tag("<function=write>",
                     {"filename": _FILENAME_SCHEMA, "content": {"type": "string"}},
                     ["filename", "content"]),
                _tag("<function=edit>",
                     {"filename": _FILENAME_SCHEMA,
                      "old": {"type": "string"}, "new": {"type": "string"}},
                     ["filename", "old", "new"]),
                _tag("<function=bash>",
                     {"command": {"type": "string", "pattern": BASH_PATTERN}},
                     ["command"]),
            ],
        }
    })


if __name__ == "__main__":
    # Smoke test: build + compile the tag against a small char vocab.
    tag = build_structural_tag()
    vocab = list(dict.fromkeys(
        ' \t\n{}[]":,._/0123456789'
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ<>=()|-'))
    ti = xgr.TokenizerInfo(vocab, vocab_type=xgr.VocabType.RAW, vocab_size=len(vocab))
    xgr.GrammarCompiler(ti).compile_structural_tag(tag)
    print("structural tag compiled OK; tools:", ", ".join(TOOL_NAMES))
