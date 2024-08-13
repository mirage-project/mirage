import os

try:
    from .core import *
except ImportError:
    import z3
    _z3_lib = os.path.join(os.path.dirname(z3.__file__), 'lib')
    os.environ['LD_LIBRARY_PATH'] = f"{_z3_lib}:{os.environ.get('LD_LIBRARY_PATH','LD_LIBRARY_PATH')}"
    
    from .core import *

class InputNotFoundError(Exception):
    """Raised when cannot find input tensors """
    pass

def new_graph():
    graph = core.PyGraph()
    return graph

# Current Version
__version__ = "0.1.0"
