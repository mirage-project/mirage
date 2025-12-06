import os
import ctypes
import z3

def preload_so(lib_path, name_hint):
    try:
        ctypes.CDLL(lib_path)
    except OSError as e:
        raise ImportError(f"Could not preload {name_hint} ({lib_path}): {e}")

_z3_libdir = os.path.join(os.path.dirname(z3.__file__), "lib")
_z3_so_path = os.path.join(_z3_libdir, "libz3.so")
preload_so(_z3_so_path, "libz3.so")

_this_dir = os.path.dirname(__file__)
_mirage_root = os.path.abspath(os.path.join(_this_dir, "..", ".."))
_subexpr_so_path = os.path.join(_mirage_root, "build", "abstract_subexpr", "release", "libabstract_subexpr.so")
_formal_verifier_so_path = os.path.join(_mirage_root, "build", "formal_verifier", "release", "libformal_verifier.so")
preload_so(_subexpr_so_path, "libabstract_subexpr.so")
preload_so(_formal_verifier_so_path, "libformal_verifier.so")

from .core import *
from .kernel import *
from .persistent_kernel import PersistentKernel
from .threadblock import *


class InputNotFoundError(Exception):
    """Raised when cannot find input tensors"""

    pass


def set_gpu_device_id(device_id: int):
    global_config.gpu_device_id = device_id
    core.set_gpu_device_id(device_id)


def bypass_compile_errors(value: bool = True):
    global_config.bypass_compile_errors = value


def new_kernel_graph():
    kgraph = core.CyKNGraph()
    return KNGraph(kgraph)


def new_threadblock_graph(
    grid_dim: tuple, block_dim: tuple, forloop_range: int, reduction_dimx: int
):
    bgraph = core.CyTBGraph(grid_dim, block_dim, forloop_range, reduction_dimx)
    return TBGraph(bgraph)


# Other Configurations
from .global_config import global_config

# Graph Datasets
from .graph_dataset import graph_dataset
from .version import __version__