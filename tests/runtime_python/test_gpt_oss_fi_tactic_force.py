# Copyright 2026 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""demo/gpt_oss_static/flashinfer_moe.py force_tactic / pinned_tactic: outside a tactic search the MoE op's lookup
answers the forced tactic (a hit, runner 0), other ops and searches keep the tuner's answer, None restores it; a pins
file applies only to its FlashInfer build and GPU, and the shipped plans/b200/fi_tactics.json is well formed.
FlashInfer's tuner lookup is stubbed (no MoE kernel: imports only, any GPU)."""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_force_and_pinned_tactic(monkeypatch, tmp_path):
    pytest.importorskip("flashinfer")
    monkeypatch.syspath_prepend(os.path.join(ROOT, "demo/gpt_oss_static"))
    from flashinfer.autotuner import AutoTuner
    monkeypatch.setattr(AutoTuner, "search_cache", lambda self, op, runners, *a, **k: (True, 0, [16, 377], "p"))
    monkeypatch.setattr(AutoTuner, "_gptoss_force", False, raising=False)   # removed after the test
    import flashinfer
    import flashinfer_moe as fm
    import torch
    t = AutoTuner.get()
    monkeypatch.setattr(t, "is_tuning_mode", False)
    fm.force_tactic(None)
    assert t.search_cache(fm.MOE_OP, [None], ()) == (True, 0, [16, 377], "p")
    fm.force_tactic((8, 12))
    assert t.search_cache(fm.MOE_OP, [None], ()) == (True, 0, [8, 12], "p")
    assert t.search_cache("flashinfer::other", [None], ())[2] == [16, 377]
    t.is_tuning_mode = True
    assert t.search_cache(fm.MOE_OP, [None], ())[2] == [16, 377]
    t.is_tuning_mode = False
    fm.force_tactic(None)
    assert t.search_cache(fm.MOE_OP, [None], ())[2] == [16, 377]

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA B200")
    f = tmp_path / "pins.json"
    f.write_text(json.dumps({"flashinfer": flashinfer.__version__, "gpu": "NVIDIA B200", "64": [16, 769]}))
    assert fm.pinned_tactic(str(f), 64) == [16, 769] and fm.pinned_tactic(str(f), 32) is None
    assert fm.pinned_tactic(str(tmp_path / "missing.json"), 64) is None
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA H100 80GB HBM3")
    assert fm.pinned_tactic(str(f), 64) is None   # another GPU: its own tactic ids
    f.write_text(json.dumps({"flashinfer": "0.0.0", "gpu": "NVIDIA B200", "64": [16, 769]}))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA B200")
    assert fm.pinned_tactic(str(f), 64) is None   # another FlashInfer build

    shipped = json.load(open(os.path.join(ROOT, "demo/gpt_oss_static/plans/b200/fi_tactics.json")))
    assert shipped["gpu"] == "NVIDIA B200" and shipped["flashinfer"]
    pins = {k: v for k, v in shipped.items() if k not in ("flashinfer", "gpu")}
    assert pins and all(k.isdigit() and len(v) == 2 and all(isinstance(x, int) for x in v) for k, v in pins.items())
