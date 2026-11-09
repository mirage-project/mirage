# On PYTHONPATH in the SGLang runs (no file in the SGLang venv changes): Python imports this at the start of every process, the
# TP workers included. It runs the system's sitecustomize (which this one shadows), then loads the static layers' hooks:
# dsv41_attn_hook.py when DSV41_STATIC_ATTN is set, dsv41_moe_hook.py when DSV41_STATIC_MOE is set.
import os
import runpy
import sys

_attn = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_moe = os.path.join(os.path.dirname(os.path.dirname(_attn)), "deepseek_v4_1_moe", "sglang")

if os.path.exists("/usr/lib/python3.12/sitecustomize.py"):
    runpy.run_path("/usr/lib/python3.12/sitecustomize.py")

# only the run's own processes: its main script and its multiprocessing children. Other Python processes (cargo's build scripts
# run `python -c`, and read its stdout) must stay plain.
_cmd = " ".join(getattr(sys, "orig_argv", sys.argv))
if any(k in _cmd for k in ("spawn_main", "sgl_generate", "sgl_gsm8k", "one_batch")):
    if os.environ.get("DSV41_STATIC_ATTN"):
        sys.path.insert(0, _attn)
        import dsv41_attn_hook  # noqa: F401,E402
    if os.environ.get("DSV41_STATIC_MOE"):
        sys.path.insert(0, _moe)
        import dsv41_moe_hook  # noqa: F401,E402
