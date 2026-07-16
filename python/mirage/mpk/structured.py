"""
Constrained decoding for MPK via xgrammar.

Owns the xgrammar state (compiler, per-request ``GrammarMatcher`` objects) and,
each decode step, advances the matchers and fills the pinned bitmask the
``apply_token_bitmask`` GPU task reads. Driven by ``OnlinePinnedRuntime``, which
calls :meth:`tick` from its drain loop.

Pinned buffers (indexed by buffer row): ``token_bitmask`` [n_req, ceil(vocab/32)]
(bit j set ⇒ token j allowed), ``mask_seq`` [n_req] (the GPU waits until this
reaches the row's decode step), ``constrained_flag`` [1] (global on/off).
"""

import threading
from typing import Dict


class StructuredGenerationManager:
    """xgrammar state + per-step bitmask production.

    token_bitmask / mask_seq / constrained_flag are the pinned buffers; tokens
    and pinned_step are the (device/pinned) token buffer and per-row decode step;
    find_row_for_rid maps rid -> buffer row (-1 if unassigned); prompt_lengths is
    the per-row prompt length (the prefill/decode boundary).
    """

    def __init__(self, token_bitmask, mask_seq, constrained_flag, tokens,
                 pinned_step, find_row_for_rid, prompt_lengths=None,
                 active_rows=None):
        self._token_bitmask = token_bitmask
        self._mask_seq = mask_seq
        self._constrained_flag = constrained_flag
        self._tokens = tokens
        self._pinned_step = pinned_step
        self._find_row_for_rid = find_row_for_rid
        self._prompt_lengths = prompt_lengths
        self._active_rows = active_rows           # callable -> list of active rows

        self._xgr = None
        self._compiler = None
        self._real_vocab = None                   # ids >= this are padding/reserved
        self._allow_all = None                    # mask row: real allowed, padding not
        self._cache: Dict[tuple, object] = {}     # spec -> CompiledGrammar
        self._matchers: Dict[int, object] = {}    # row -> GrammarMatcher
        self._pending: Dict[int, object] = {}     # rid -> GrammarMatcher (row TBD)
        self._prompt_len: Dict[int, int] = {}     # row -> prompt length
        self._last_step: Dict[int, int] = {}      # row -> last published step
        self._row_rid: Dict[int, int] = {}        # row -> rid whose matcher it holds
        self._active_rids: set = set()            # rids with a live grammar; the
        #                                           global flag is on iff non-empty
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._token_bitmask is not None

    def init_xgrammar(self, tokenizer=None, vocab_size=None,
                      tokenizer_info=None) -> None:
        """Build the compiler. ``vocab_size`` must equal the model's (padded)
        logit dim — it sizes the bitmask. Pass a HF ``tokenizer`` or a pre-built
        ``tokenizer_info``."""
        import xgrammar as xgr

        assert self.available, "kernel was not built with constrained-decoding buffers"
        if tokenizer_info is None:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                tokenizer, vocab_size=vocab_size)
        vocab_size = vocab_size or tokenizer_info.vocab_size
        assert (vocab_size + 31) // 32 == self._token_bitmask.shape[1], (
            f"vocab_size={vocab_size} does not match bitmask width "
            f"{self._token_bitmask.shape[1]} (must equal the model's logit dim)")
        self._xgr = xgr
        self._compiler = xgr.GrammarCompiler(tokenizer_info)

        # Padding hardening. Token ids >= real_vocab are embedding padding /
        # reserved slots (≈0 logits, not decodable) that must never be sampled.
        # len(tokenizer) is a safe lower bound — all real/chat-special tokens are
        # below it. Precompute a mask row with those bits cleared, used to AND
        # every published mask and as the unconstrained fallback, so padding is
        # disallowed everywhere and the argmax can't be lured onto it.
        self._real_vocab = None
        if tokenizer is not None:
            try:
                n = len(tokenizer)
                if 0 < n < vocab_size:
                    self._real_vocab = n
            except Exception:
                pass
        self._allow_all = self._token_bitmask.new_full(
            (self._token_bitmask.shape[1],), -1)
        if self._real_vocab is not None:
            w, b = divmod(self._real_vocab, 32)
            self._allow_all[w:] = 0
            if b:
                self._allow_all[w] = (1 << b) - 1

    def _compile(self, json_schema=None, ebnf=None, regex=None,
                 structural_tag=None, any_whitespace=True):
        assert self._compiler is not None, "call init_xgrammar() first"
        c = self._compiler
        if json_schema is not None:
            # any_whitespace=False forbids arbitrary inter-token whitespace
            # (keeps canonical separators), avoiding runaway spaces/newlines.
            key = ("schema", json_schema, any_whitespace)
            build = lambda: c.compile_json_schema(json_schema, any_whitespace=any_whitespace)
        elif ebnf is not None:
            key, build = ("ebnf", ebnf), lambda: c.compile_grammar(ebnf)
        elif regex is not None:
            key, build = ("regex", regex), lambda: c.compile_regex(regex)
        elif structural_tag is not None:  # StructuralTag object / JSON str / dict
            key, build = ("stag", repr(structural_tag)), lambda: c.compile_structural_tag(structural_tag)
        else:
            key, build = ("json",), c.compile_builtin_json_grammar
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def set_request_grammar(self, rid: int, *, json_schema=None, ebnf=None,
                            regex=None, structural_tag=None,
                            triggered_tags=None, tags_with_separator=None,
                            dispatch=None, model=None,
                            any_whitespace=True) -> None:
        """Attach a grammar to a request. Provide exactly one grammar source:

          * a raw source — ``json_schema`` / ``ebnf`` / ``regex`` /
            ``structural_tag`` (none of these ⇒ builtin JSON); or
          * a tool list (from :func:`~mirage.mpk.create_tools`) via one of
            ``triggered_tags`` (free text + calls), ``tags_with_separator``
            (tool-only), or ``dispatch`` (many tools / loop). Add ``model=``
            (e.g. ``"qwen_3"``) to emit the tools in that model's *native*
            tool-call format instead of the chosen wrapper.

        ``any_whitespace=False`` (JSON schema only) forbids arbitrary whitespace.
        Binds to the row lazily and turns on the global flag.
        """
        tool_args = [t for t in (triggered_tags, tags_with_separator, dispatch)
                     if t is not None]
        assert len(tool_args) <= 1, (
            "pass tools via only one of triggered_tags= / tags_with_separator= / "
            "dispatch=")
        assert model is None or tool_args, (
            "model= requires a tool list (triggered_tags= / tags_with_separator= "
            "/ dispatch=)")
        if tool_args:
            assert json_schema is ebnf is regex is structural_tag is None, (
                "pass a tool list OR a raw grammar source, not both")
            from . import structured_tools as T
            tools = tool_args[0]
            if model is not None:
                structural_tag = T.build_model_structural_tag(tools, model=model)
            elif triggered_tags is not None:
                structural_tag = T.build_triggered_tags_structural_tag(tools)
            elif tags_with_separator is not None:
                structural_tag = T.build_tags_with_separator_structural_tag(tools)
            else:
                structural_tag = T.build_dispatch_structural_tag(tools)
        matcher = self._xgr.GrammarMatcher(self._compile(
            json_schema=json_schema, ebnf=ebnf, regex=regex,
            structural_tag=structural_tag, any_whitespace=any_whitespace))
        with self._lock:
            self._pending[rid] = matcher
            self._active_rids.add(rid)
        self.set_constrained(True)

    def set_constrained(self, on: bool) -> None:
        if self._constrained_flag is not None:
            self._constrained_flag[0] = 1 if on else 0

    def tick(self) -> None:
        """Bind newly-admitted grammars and, while the flag is on, publish a mask
        for EVERY active row each step (all-ones for rows without a live matcher).

        The device masking task spin-waits on ``mask_seq[row] >= step`` for every
        active request when the flag is on, so an active row that is unconstrained
        (mixed batch), whose grammar has terminated, or is a leftover of a
        just-completed constrained request must still get a (permissive) mask
        published or the kernel hangs. When the flag is off the task early-returns
        without reading mask_seq, so there is nothing to publish.

        pinned_step[row] is the row's decode step; the newest generated token sits
        at tokens[row, step]. During prefill (step < prompt_len) argmax outputs
        are not written to tokens, so we publish the fresh mask and accept
        nothing; in decode we accept tokens[row, step] before filling the next.
        """
        if self._compiler is None:
            return
        # Held for the whole body so a concurrent release()/set_request_grammar()
        # (drain_completions runs on both the drain thread and wait_for_request's
        # thread) can't mutate the matcher dicts mid-iteration.
        with self._lock:
            if self._pending:
                for rid in list(self._pending):
                    row = self._find_row_for_rid(rid)
                    if row >= 0:
                        self._matchers[row] = self._pending.pop(rid)
                        self._row_rid[row] = rid
                        self._prompt_len[row] = (int(self._prompt_lengths[row].item())
                                                 if self._prompt_lengths is not None else 0)
                        self._last_step[row] = -1
            if self._constrained_flag is None or int(self._constrained_flag[0]) == 0:
                return
            rows = (self._active_rows() if self._active_rows is not None
                    else list(self._matchers))
            for row in rows:
                step = int(self._pinned_step[row].item())
                if step == self._last_step.get(row, -1):
                    continue
                matcher = self._matchers.get(row)
                if matcher is not None and matcher.is_terminated():
                    self._matchers.pop(row, None)
                    self._row_rid.pop(row, None)
                    matcher = None
                if matcher is not None and step >= self._prompt_len.get(row, 0) \
                        and self._last_step.get(row, -1) >= 0:   # decode, not first
                    tok = int(self._tokens[row, step].item())
                    if self._real_vocab is not None and tok >= self._real_vocab:
                        # Padding/reserved id (e.g. 153599) — not a grammar token;
                        # skip accept_token so xgrammar doesn't spam a rejection
                        # warning, and treat it as the recovery trigger below.
                        accepted = False
                    else:
                        try:
                            accepted = matcher.accept_token(tok)
                        except Exception:
                            accepted = False
                    if not accepted:
                        # The model sampled a grammar-invalid token. Under correct
                        # masking it can only sample allowed tokens, so this means a
                        # degenerate all-disallowed step where the argmax fell back
                        # to a padding id — which would otherwise spin forever,
                        # re-rejecting the same token each step. Stop constraining
                        # this row so the request can finish (it decodes freely
                        # under the padding-disallowed fallback mask below).
                        self._matchers.pop(row, None)
                        self._row_rid.pop(row, None)
                        matcher = None
                if matcher is not None:
                    matcher.fill_next_token_bitmask(self._token_bitmask, index=row)
                    self._token_bitmask[row] &= self._allow_all   # hard-disallow padding
                else:
                    # no live grammar for this active row → allow every real token
                    self._token_bitmask[row].copy_(self._allow_all)
                self._mask_seq[row] = step      # data-before-flag: bitmask written first
                self._last_step[row] = step

    def release(self, rid: int, row: int = None) -> None:
        """Drop a request's grammar state; clear the global flag once no grammars
        remain.

        Idempotent and thread-safe. The runtime calls this automatically on
        completion (from ``drain_completions``) with the completion ring's
        authoritative ``row`` — required because the GPU resets pinned_rid_at_row
        to -1 at completion, so a rid-based ``find_row_for_rid`` lookup would miss
        the row and leave the flag stuck on. ``row`` is cleaned only if it still
        holds this rid's matcher (guards a row already recycled to a new request).
        Callers normally never invoke this; use it only to abort a grammar early.
        """
        with self._lock:
            self._pending.pop(rid, None)
            self._active_rids.discard(rid)
            if row is None:
                row = self._find_row_for_rid(rid)
            if row is not None and row >= 0 and self._row_rid.get(row, rid) == rid:
                for d in (self._matchers, self._prompt_len, self._last_step,
                          self._row_rid):
                    d.pop(row, None)
            active = bool(self._active_rids)
        if not active:
            self.set_constrained(False)

    def reset(self) -> None:
        with self._lock:
            for d in (self._matchers, self._pending, self._prompt_len,
                      self._last_step, self._row_rid):
                d.clear()
            self._active_rids.clear()
        if self._mask_seq is not None:
            self._mask_seq.zero_()
        self.set_constrained(False)
