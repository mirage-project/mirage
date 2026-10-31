# Imported at the start of every Python process of the SGLang venv (kda_static_boot.pth next to it in site-packages): loads
# the static-KDA hook (kda_static_hook.py) when KDA_STATIC_LAYERS is set, otherwise does nothing.
# Install: copy this file to the venv's site-packages and write a kda_static_boot.pth there holding the line
#   import kda_static_boot
# KDA_STATIC_ROOT names the mirage checkout (this file is not in it).
import os

if os.environ.get("KDA_STATIC_LAYERS"):
    import sys

    _root = os.environ["KDA_STATIC_ROOT"]
    sys.path.insert(0, os.path.join(_root, "demo", "glm53_flash", "sglang_kda"))
    import kda_static_hook  # noqa: F401,E402
