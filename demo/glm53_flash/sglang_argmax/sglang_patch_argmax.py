"""Hook flash_argmax_hook.py (next to this file) into the installed SGLang (0.5.22; two files). A backup <file>.flash_am_orig
is made once and every patch starts from it:
  srt/layers/logits_processor.py         LogitsProcessor._get_logits: maybe_defer() right after the lm_head GEMM
  srt/model_executor/model_runner.py     ModelRunner.sample: before_sample() before the logits bias and the sampler
Inert unless FLASH_ARGMAX=1 (unset: the hook module is not even imported; the patched lines cost one environ lookup). The
patched lines import flash_argmax_hook from FLASH_ARGMAX_DIR, else from this file's directory at patch time (the SGLang
runs strip PYTHONPATH: sgl_chain.py).

    python sglang_patch_argmax.py             -> "ARGMAX-PATCH applied ..." (or "ARGMAX-PATCH NOT applied: ..." + exit 1)
    python sglang_patch_argmax.py --restore   -> "ARGMAX-PATCH restored ..."
    --sglang-dir DIR: the sglang package directory to patch (default: the imported sglang's)
Every anchor must be found exactly once in its file's backup, else nothing is written (the message names the file and the
anchor). A file that is neither its backup nor the patch of its backup (SGLang reinstalled or changed after the backup):
nothing is written either (remove the stale .flash_am_orig after checking).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUFFIX = ".flash_am_orig"
TAG = "flash_argmax (sglang_patch_argmax.py)"


def _import_lines(indent: str) -> str:
    """The lines that bind _fa to the hook module (imported once per process; later calls: a sys.modules lookup)."""
    return (f"{indent}import sys as _fa_sys\n"
            f"{indent}_fa = _fa_sys.modules.get(\"flash_argmax_hook\")\n"
            f"{indent}if _fa is None:\n"
            f"{indent}    _fa_sys.path.insert(0, _fa_os.environ.get(\"FLASH_ARGMAX_DIR\") or {HERE!r})\n"
            f"{indent}    import flash_argmax_hook as _fa\n")


# (path under the sglang package, anchor, replacement)
EDITS = [
    ("srt/layers/logits_processor.py", """        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)

        used_tp_lm_head_all_to_all = False
""", f"""        import os as _fa_os
        if _fa_os.environ.get("FLASH_ARGMAX", "") == "1":   # {TAG}; unset: not imported
{_import_lines(" " * 12)}            _fa_r = _fa.maybe_defer(self, logits, logits_metadata)
            if _fa_r is not None:   # per-rank argmax in the graph; the logits are gathered only on demand
                return _fa_r

        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)

        used_tp_lm_head_all_to_all = False
"""),
    ("srt/model_executor/model_runner.py", """        logits_output.auxiliary_device_output = None
        observer = self.sampling_observer
""", f"""        logits_output.auxiliary_device_output = None
        observer = self.sampling_observer
        import os as _fa_os
        if _fa_os.environ.get("FLASH_ARGMAX", "") == "1":   # {TAG}; unset: not imported
{_import_lines(" " * 12)}            _fa_ids = _fa.before_sample(self, logits_output, forward_batch, observer)
            if _fa_ids is not None:   # greedy batch: the per-rank argmax's token ids
                self.ngram_embedding_manager.update_after_decode(
                    next_token_ids=_fa_ids,
                    forward_batch=forward_batch,
                )
                return _fa_ids
"""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--sglang-dir", default=None, help="the sglang package directory (default: the imported sglang's)")
    args = ap.parse_args()
    if args.sglang_dir:
        sdir, ver = os.path.abspath(args.sglang_dir), "?"
    else:
        import sglang
        sdir, ver = os.path.dirname(sglang.__file__), getattr(sglang, "__version__", "?")
    files = sorted({rel for rel, _, _ in EDITS})
    out = {}
    for rel in files:
        f = os.path.join(sdir, rel)
        orig = f + SUFFIX
        if not os.path.exists(f):
            print(f"ARGMAX-PATCH NOT applied: {f} not found (sglang {ver}); nothing changed")
            sys.exit(1)
        if args.restore and not os.path.exists(orig):
            print(f"ARGMAX-PATCH restore: no backup {orig}: {rel} left as it is")
            continue
        base =open(orig).read() if os.path.exists(orig) else open(f).read()
        patched = base
        for r, a, b in EDITS:
            if r != rel:
                continue
            n = patched.count(a)
            if n != 1:
                print(f"ARGMAX-PATCH NOT applied: anchor found {n} times (need 1) in {orig if os.path.exists(orig) else f} "
                      f"(sglang {ver}); nothing changed. Anchor:\n{a}")
                sys.exit(1)
            patched = patched.replace(a, b)
        cur = open(f).read()
        if cur not in (base, patched):
            print(f"ARGMAX-PATCH NOT {'restored' if args.restore else 'applied'}: {f} is neither its backup {orig} nor "
                  f"the patch of it (SGLang changed after the backup?); nothing changed")
            sys.exit(1)
        if TAG in base:
            print(f"ARGMAX-PATCH NOT applied: the backup {orig if os.path.exists(orig) else f} already holds the patch; "
                  f"nothing changed")
            sys.exit(1)
        out[f] = (orig, base, base if args.restore else patched, cur)
    for f, (orig, base, want, cur) in out.items():   # every file checked: now write
        if not os.path.exists(orig):
            with open(orig, "w") as fh:
                fh.write(base)
        if cur != want:
            with open(f, "w") as fh:
                fh.write(want)
    what = "restored" if args.restore else "applied"
    print(f"ARGMAX-PATCH {what}: {', '.join(files)} (sglang {ver}, {sdir}; hook dir {HERE})")


if __name__ == "__main__":
    main()
