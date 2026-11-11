"""Hook amd/serving/glm53/sglang_glm53.py into the installed SGLang's deepseek_v2.py (v0.5.22, the MI355X ROCm image). A backup
deepseek_v2.py.mpk_orig is made once; every patch starts from it. The hooks do nothing unless MPK_G53=run (and the model is
GlmMoeDsaForCausalLM).

    python amd/serving/glm53/sglang_patch_glm53.py             (in the server's environment; the container's site-packages)
    python amd/serving/glm53/sglang_patch_glm53.py --restore
"""
import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
IMPORT = f"import sys as _mpk_sys; _mpk_sys.path.insert(0, {HERE!r}); import sglang_glm53 as _mpk_g53"

# (anchor, replacement): every anchor must be found exactly once
EDITS = [
    ("""import torch.nn.functional as F
from torch import nn
""", f"""import torch.nn.functional as F
from torch import nn
{IMPORT}   # mpk_g53
"""),
    # a decode step we take: our kernels, every layer (DeepseekV2Model.forward)
    ("""    ) -> Union[torch.Tensor, PPProxyTensors]:
        total_num_layers = self.end_layer - self.start_layer
        dsa_forward_uses_topk = self._dsa_forward_uses_topk()
""", """    ) -> Union[torch.Tensor, PPProxyTensors]:
        if _mpk_g53.active(forward_batch):   # mpk_g53
            return _mpk_g53.forward(self, input_ids, positions, forward_batch, input_embeds)   # mpk_g53
        total_num_layers = self.end_layer - self.start_layer
        dsa_forward_uses_topk = self._dsa_forward_uses_topk()
"""),
    # after the weights are loaded and SGLang's post-load processing (before the KV pool and the graph capture)
    ("""    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        self.do_load_weights(weights, is_nextn)
""", """    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        self.do_load_weights(weights, is_nextn)
        if not is_nextn:   # mpk_g53
            _mpk_g53.prepare(self)   # mpk_g53
"""),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--restore", action="store_true")
    a = p.parse_args()
    import sglang
    f = os.path.join(os.path.dirname(sglang.__file__), "srt", "models", "deepseek_v2.py")
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
            raise SystemExit(f"PATCH FAILED: anchor found {n} times: {anchor[:80]!r}")
        s = s.replace(anchor, new)
    open(f, "w").write(s)
    print("PATCHED", f)


if __name__ == "__main__":
    main()
