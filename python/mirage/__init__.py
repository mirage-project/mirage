import os

try:
    from .core import *
except ImportError:
    _mirage_root = os.path.dirname(__file__)
    _z3_lib = os.path.join(_mirage_root, 'deps', 'z3', 'build')
    print(f"Z3 library path: {_z3_lib}")
    os.environ['LD_LIBRARY_PATH'] = f"{_z3_lib}:{os.environ.get('LD_LIBRARY_PATH','LD_LIBRARY_PATH')}"
    os.environ['Z3_DIR'] = _z3_lib
    
    from .core import *

class InputNotFoundError(Exception):
    """Raised when cannot find input tensors """
    pass

def new_graph():
    graph = core.PyGraph()
    return graph

# Current Version
__version__ = "0.1.0"
