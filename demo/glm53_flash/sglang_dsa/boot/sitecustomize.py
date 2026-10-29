# On PYTHONPATH in the SGLang runs (sgl_chain.py; no file in the SGLang venv changes): Python imports this module at the
# start of every process, the TP workers included. It runs the system's sitecustomize (which this one shadows), then loads the
# static-DSA hook (dsa_static_hook.py) when DSA_STATIC_LAYERS is set, and the static-MoE hook (sglang_moe/moe_static_hook.py)
# when MOE_STATIC_LAYERS is set (the KDA hook comes through the venv's kda_static_boot.pth). The MoE hook first: the DSA hook's
# decoder-layer forward calls the MoE's forward and the FFN exit it patches.
import os
import runpy

_here = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))   # this checkout

if os.path.exists("/usr/lib/python3.12/sitecustomize.py"):
    runpy.run_path("/usr/lib/python3.12/sitecustomize.py")

if os.environ.get("MOE_STATIC_LAYERS"):
    import sys

    _root = os.environ.get("MOE_STATIC_ROOT", os.environ.get("DSA_STATIC_ROOT", _here))
    sys.path.insert(0, os.path.join(_root, "demo", "glm53_flash", "sglang_moe"))
    import moe_static_hook  # noqa: F401,E402

if os.environ.get("DSA_STATIC_LAYERS"):
    import sys

    _root = os.environ.get("DSA_STATIC_ROOT", _here)
    sys.path.insert(0, os.path.join(_root, "demo", "glm53_flash", "sglang_dsa"))
    import dsa_static_hook  # noqa: F401,E402
