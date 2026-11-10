"""Hook amd/serving/glm/sglang_glm.py into the installed SGLang's glm5_next.py (v0.5.22, the MI355X ROCm image). A backup
glm5_next.py.mpk_orig is made once; every patch starts from it. The hooks do nothing unless MPK_GLM=run.

    python amd/serving/glm/sglang_patch_glm.py             (in the server's environment; the container's site-packages)
    python amd/serving/glm/sglang_patch_glm.py --restore
"""
import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
IMPORT = f"import sys as _mpk_sys; _mpk_sys.path.insert(0, {HERE!r}); import sglang_glm as _mpk_glm"

# (anchor, replacement): every anchor must be found exactly once
EDITS = [
    ("""from torch.nn import functional as F
""", f"""from torch.nn import functional as F
{IMPORT}   # mpk_glm
"""),
    # a decode step of a listed batch size: our kernels, every layer (Glm5NextModel.forward)
    ("""        total_num_layers = self.end_layer - self.start_layer
        if self.pp_group.is_first_rank:
""", """        if _mpk_glm.active(forward_batch):   # mpk_glm
            return _mpk_glm.forward(self, input_ids, forward_batch, input_embeds)   # mpk_glm
        total_num_layers = self.end_layer - self.start_layer
        if self.pp_group.is_first_rank:
"""),
    # after the weights are loaded and SGLang's post-load processing (before the KV pool and the graph capture)
    ("""        if run_post:
            DeepseekV2WeightLoaderMixin.post_load_weights(
                self, is_nextn=is_nextn, weight_names=weight_names
            )

    def post_load_weights(""", """        if run_post:
            DeepseekV2WeightLoaderMixin.post_load_weights(
                self, is_nextn=is_nextn, weight_names=weight_names
            )
        if not is_nextn:   # mpk_glm
            _mpk_glm.prepare(self)   # mpk_glm

    def post_load_weights("""),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--restore", action="store_true")
    a = p.parse_args()
    import sglang
    f = os.path.join(os.path.dirname(sglang.__file__), "srt", "models", "glm5_next.py")
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
            raise SystemExit(f"sglang_patch_glm: anchor found {n} times in {orig} (SGLang version?):\n{anchor[:300]}")
        s = s.replace(anchor, new)
    open(f, "w").write(s)
    import py_compile
    py_compile.compile(f, doraise=True)
    print("PATCHED", f, f"({s.count('# mpk_glm')} hook lines; sglang {getattr(sglang, '__version__', '?')})")


if __name__ == "__main__":
    main()
