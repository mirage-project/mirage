from .core import *

class InputNotFoundError(Exception):
    """Raised when cannot find input tensors """
    pass

def new_graph():
    graph = core.PyGraph()
    return graph

# Current Version
__version__ = "0.1.0"
