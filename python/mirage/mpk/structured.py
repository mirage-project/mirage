"""
Structured generation (constrained decoding) for MPK via xgrammar.

This module owns all xgrammar state — the grammar compiler, the per-request
``GrammarMatcher`` objects, and the per-step bitmask fill — and writes into the
pinned CPU↔GPU buffers that the ``apply_token_bitmask`` MPK task reads:

  * ``pinned_token_bitmask``    [n_req, ceil(vocab/32)] int32 — bit j set ⇒ token
                                j allowed, indexed by buffer row.
  * ``pinned_mask_seq``         [n_req] int32 — published step a row's mask is
                                valid for; the GPU task waits (ld.acquire.sys)
                                until this catches up to the row's decode step.
  * ``pinned_constrained_flag`` [1] int32 — global runtime toggle. 0 ⇒ the GPU
                                masking task is a no-op (unconstrained decode
                                costs nothing); 1 ⇒ masks are applied.

The grammar *advance* + bitmask *fill* run on the CPU each decode step,
overlapped with the GPU forward pass, so the mask is ready by the time the
tail masking task runs.  :class:`StructuredGenerationManager` is driven by
:class:`~mirage.mpk.online_pinned_runtime.OnlinePinnedRuntime`, which calls
:meth:`tick` from its background drain loop.
"""

import threading
from typing import Dict


class StructuredGenerationManager:
    """Encapsulates xgrammar grammar state and per-step bitmask production.

    Parameters
    ----------
    token_bitmask    : pinned int32 [n_req, ceil(vocab/32)]
    mask_seq         : pinned int32 [n_req]
    constrained_flag : pinned int32 [1]
    tokens           : the (device) token buffer [n_req, max_seq_len]
    pinned_step      : pinned int32 [n_req], the GPU-published decode step/row
    find_row_for_rid : callable rid -> buffer row (or -1 if not yet assigned)
    """

    def __init__(self, token_bitmask, mask_seq, constrained_flag, tokens,
                 pinned_step, find_row_for_rid, prompt_lengths=None):
        self._token_bitmask = token_bitmask
        self._mask_seq = mask_seq
        self._constrained_flag = constrained_flag
        self._tokens = tokens
        self._pinned_step = pinned_step
        self._find_row_for_rid = find_row_for_rid
        self._prompt_lengths = prompt_lengths   # [n_req] int32, indexed by row

        self._xgr = None                       # xgrammar module (lazy import)
        self._compiler = None
        self._tokenizer_info = None
        self._vocab_size = None
        self._compiled_cache: Dict[tuple, object] = {}   # spec-key -> CompiledGrammar
        self._matchers: Dict[int, object] = {}           # buffer_row -> GrammarMatcher
        self._pending: Dict[int, object] = {}            # rid -> GrammarMatcher (row TBD)
        self._prompt_len: Dict[int, int] = {}            # row -> prompt length (decode boundary)
        self._last_step: Dict[int, int] = {}             # row -> last step a mask was published
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """True if the kernel was built with the constrained-decoding buffers."""
        return self._token_bitmask is not None

    # ── Setup ─────────────────────────────────────────────────────────────

    def init_xgrammar(self, tokenizer=None, vocab_size: int = None,
                      tokenizer_info=None) -> None:
        """Build the xgrammar compiler + tokenizer info.

        Pass either a HuggingFace ``tokenizer`` (the common case) or a
        pre-built ``tokenizer_info`` (e.g. for testing). ``vocab_size`` MUST
        equal the model's logit dimension (padding included) — it sizes the
        bitmask the GPU masking task reads.
        """
        import xgrammar as xgr  # lazy: only needed when constraining

        assert self.available, (
            "constrained decoding requires the pinned bitmask buffers; ensure "
            "the kernel was built in online_pinned mode with constrained "
            "decoding meta tensors"
        )
        if tokenizer_info is None:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                tokenizer, vocab_size=vocab_size
            )
        if vocab_size is None:
            vocab_size = tokenizer_info.vocab_size
        expected_words = self._token_bitmask.shape[1]
        got_words = (vocab_size + 31) // 32
        assert got_words == expected_words, (
            f"vocab_size={vocab_size} (→{got_words} mask words) does not match "
            f"the allocated bitmask width {expected_words}; RunnerConfig."
            f"vocab_size must equal the model's logit dim"
        )
        self._xgr = xgr
        self._vocab_size = vocab_size
        self._tokenizer_info = tokenizer_info
        self._compiler = xgr.GrammarCompiler(self._tokenizer_info)

    def _compile(self, json_schema=None, ebnf=None, regex=None):
        assert self._compiler is not None, "call init_xgrammar() first"
        if json_schema is not None:
            key = ("schema", json_schema)
        elif ebnf is not None:
            key = ("ebnf", ebnf)
        elif regex is not None:
            key = ("regex", regex)
        else:
            key = ("json",)
        cached = self._compiled_cache.get(key)
        if cached is not None:
            return cached
        c = self._compiler
        if json_schema is not None:
            compiled = c.compile_json_schema(json_schema)
        elif ebnf is not None:
            compiled = c.compile_grammar(ebnf)
        elif regex is not None:
            compiled = c.compile_regex(regex)
        else:
            compiled = c.compile_builtin_json_grammar()
        self._compiled_cache[key] = compiled
        return compiled

    def set_request_grammar(self, rid: int, *, json_schema=None, ebnf=None,
                            regex=None) -> None:
        """Attach a grammar to a request (call once, near submit()).

        The matcher binds to the request's buffer row lazily once the GPU
        assigns one, and is advanced per decode step by :meth:`tick`.  Turns
        on the global constrained flag.
        """
        compiled = self._compile(json_schema=json_schema, ebnf=ebnf, regex=regex)
        matcher = self._xgr.GrammarMatcher(compiled)
        with self._lock:
            self._pending[rid] = matcher
        self.set_constrained(True)

    def set_constrained(self, on: bool) -> None:
        """Flip the global runtime flag the GPU masking task reads."""
        if self._constrained_flag is not None:
            self._constrained_flag[0] = 1 if on else 0

    # ── Per-step driver ───────────────────────────────────────────────────

    def tick(self) -> None:
        """Advance every active matcher and publish its mask for the current
        step.  Cheap no-op when no grammars are active.

        Contract (validated against a live online_pinned Qwen3 decode):
          * pinned_step[row] == config.step[row]; the most recently generated
            token lives at tokens[row, pinned_step] (cf. the EOS check on
            tokens[row, step+num_tokens]).
          * PREFILL (step < prompt_length[row]): the model is consuming the
            prompt; its argmax outputs are NOT written to tokens (the kernel
            only writes when step+j+1 >= prompt_len). We must NOT accept these
            "tokens" — doing so feeds garbage to the matcher and terminates it,
            which then stops publishing masks and deadlocks the kernel. Instead
            we keep publishing the *fresh*-grammar mask: it constrains the first
            generated token, which is produced inside the final prefill chunk.
          * DECODE (step >= prompt_length[row]): accept the just-generated token
            tokens[row, step], then fill the mask for the next token.
          * The first published mask for a row is always the fresh grammar (no
            accept), regardless of phase.
        """
        if self._compiler is None:
            return
        # Bind pending grammars to rows the GPU has now assigned, recording the
        # row's prompt length (the prefill/decode boundary).
        if self._pending:
            with self._lock:
                for rid in list(self._pending):
                    row = self._find_row_for_rid(rid)
                    if row >= 0:
                        self._matchers[row] = self._pending.pop(rid)
                        self._prompt_len[row] = (
                            int(self._prompt_lengths[row].item())
                            if self._prompt_lengths is not None else 0)
                        self._last_step[row] = -1
        for row, matcher in list(self._matchers.items()):
            if matcher.is_terminated():
                self._matchers.pop(row, None)  # done; don't feed it more tokens
                continue
            step = int(self._pinned_step[row].item())
            if step == self._last_step.get(row, -1):
                continue  # mask for this step already published
            plen = self._prompt_len.get(row, 0)
            first = self._last_step.get(row, -1) < 0
            # Accept only during decode, and never on the very first mask.
            if step >= plen and not first:
                tok = int(self._tokens[row, step].item())
                try:
                    matcher.accept_token(tok)
                except Exception:
                    pass
            # Fill this row's bitmask directly into the pinned buffer, then
            # publish the seq (data-before-flag: the pinned writes complete
            # before the seq store the GPU acquires on).
            matcher.fill_next_token_bitmask(self._token_bitmask, index=row)
            self._mask_seq[row] = step
            self._last_step[row] = step
            if matcher.is_terminated():
                self._matchers.pop(row, None)

    # ── Teardown ──────────────────────────────────────────────────────────

    def release(self, rid: int) -> None:
        """Drop a request's matcher (call on completion)."""
        with self._lock:
            self._pending.pop(rid, None)
        row = self._find_row_for_rid(rid)
        if row >= 0:
            self._matchers.pop(row, None)
            self._prompt_len.pop(row, None)
            self._last_step.pop(row, None)
        if not self._matchers and not self._pending:
            self.set_constrained(False)

    def reset(self) -> None:
        """Clear all grammar state for a new session."""
        with self._lock:
            self._matchers.clear()
            self._pending.clear()
            self._prompt_len.clear()
            self._last_step.clear()
        if self._mask_seq is not None:
            self._mask_seq.zero_()
        self.set_constrained(False)
