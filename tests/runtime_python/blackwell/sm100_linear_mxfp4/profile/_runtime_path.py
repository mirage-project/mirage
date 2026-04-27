from pathlib import Path
import sys


RUNTIME_DIR = Path(__file__).resolve().parent.parent
runtime_dir = str(RUNTIME_DIR)
if runtime_dir not in sys.path:
    sys.path.insert(0, runtime_dir)