# On PYTHONPATH in the SGLang runs (no file in the SGLang venv changes): Python imports this at the start of every process, the
# TP workers included. It runs the system's sitecustomize (which this one shadows), then loads dsv41_static_hook.py when
# DSV41_STATIC_ATTN or DSV41_STATIC_MOE is set.
import os
import runpy
import sys

_here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if os.path.exists("/usr/lib/python3.12/sitecustomize.py"):
    runpy.run_path("/usr/lib/python3.12/sitecustomize.py")

# only the run's own processes: its main script and its multiprocessing children. Other Python processes (cargo's build scripts
# run `python -c`, and read its stdout) must stay plain.
_cmd = " ".join(getattr(sys, "orig_argv", sys.argv))
if (os.environ.get("DSV41_STATIC_ATTN") or os.environ.get("DSV41_STATIC_MOE")) and any(k in _cmd for k in ("spawn_main", "sgl_generate", "sgl_gsm8k", "one_batch")):
    sys.path.insert(0, _here)
    import dsv41_static_hook  # noqa: F401,E402
