"""Hook amd/serving/sglang_k3.py into the installed SGLang's kimi_k3.py (v0.5.21, the MI355X image). A backup
kimi_k3.py.mpk_orig is made once; every patch starts from it. The hooks do nothing unless MPK_K3_MOE / _KDA / _MLA are set.

    python amd/serving/sglang_patch.py             (in the server's environment; the container's site-packages)
    python amd/serving/sglang_patch.py --restore
"""
import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
IMPORT = f"import sys as _mpk_sys; _mpk_sys.path.insert(0, {HERE!r}); import sglang_k3 as _mpk_amd"

# (anchor, replacement): every anchor must be found exactly once
EDITS = [
    # the hooks' module, imported once with the model file
    ("""from sglang.srt.utils import is_hip, is_npu, make_pp_layers
""", f"""from sglang.srt.utils import is_hip, is_npu, make_pp_layers
{IMPORT}   # mpk_amd
"""),
    # the layer keeps the attention bank (aggregation 2's rows)
    ("""        self._declare_stages(config, layer_idx, attn_bank)
""", """        self._mpk_bank = attn_bank   # mpk_amd
        self._declare_stages(config, layer_idx, attn_bank)
"""),
    # after aggregation 1: ours for the attention part through aggregation 2, or SGLang's; the MoE: ours or SGLang's
    ("""        hidden_states = self.attn_boundary.prepare(
            hidden_states, forward_batch, capture_gathered=capture_gathered
        )
        hidden_states = self._run_self_attn(
            hidden_states, positions, forward_batch, zero_allocator
        )
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        if self._ffn_writes_stream:
            hidden_states = self.mlp(
                hidden_states,
                prefix_sum=residual_batch.written_residual(forward_batch),
            )
        else:
            hidden_states = self.mlp(hidden_states)
        return self.ffn_boundary.finish(hidden_states, forward_batch)
""", f"""        hidden_states = self.attn_boundary.prepare(
            hidden_states, forward_batch, capture_gathered=capture_gathered
        )
        _mpk_h = _mpk_amd.attn(self, hidden_states, forward_batch)   # mpk_amd
        if _mpk_h is not None:   # mpk_amd: ours ran the attention part through aggregation 2
            hidden_states = _mpk_h
        else:
            hidden_states = self._run_self_attn(
                hidden_states, positions, forward_batch, zero_allocator
            )
            hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
            hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        if self._ffn_writes_stream:
            _mpk_p = residual_batch.written_residual(forward_batch)   # mpk_amd
            _mpk_y = _mpk_amd.moe(self, hidden_states, _mpk_p, forward_batch)   # mpk_amd
            hidden_states = _mpk_y if _mpk_y is not None else self.mlp(
                hidden_states,
                prefix_sum=_mpk_p,
            )
        else:
            hidden_states = self.mlp(hidden_states)
        return self.ffn_boundary.finish(hidden_states, forward_batch)
"""),
    # after the weights are loaded (before SGLang's post-load processing and the graph capture)
    ("""        self.post_load_weights()
        return loaded_params
""", f"""        self.post_load_weights()
        _mpk_amd.prepare(self)   # mpk_amd
        return loaded_params
"""),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--restore", action="store_true")
    a = p.parse_args()
    import sglang
    f = os.path.join(os.path.dirname(sglang.__file__), "srt", "models", "kimi_k3.py")
    orig = f + ".mpk_orig"
    if not os.path.exists(orig):
        shutil.copy(f, orig)
    if a.restore:
        shutil.copy(orig, f)
        print("RESTORED", f)
        return
    s = open(orig).read()
    for anchor, new in EDITS:
        n = s.count(anchor)
        if n != 1:
            raise SystemExit(f"sglang_patch: anchor found {n} times in {orig} (SGLang version?):\n{anchor[:300]}")
        s = s.replace(anchor, new)
    open(f, "w").write(s)
    import py_compile
    py_compile.compile(f, doraise=True)
    print("PATCHED", f, f"({s.count('# mpk_amd')} hook lines; sglang {getattr(sglang, '__version__', '?')})")


if __name__ == "__main__":
    main()
