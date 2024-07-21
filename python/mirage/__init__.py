from .core import *
from .wrapper import PyGraphWrapper

class InputNotFoundError(Exception):
    """Raised when cannot find input tensors """
    pass

def new_graph():
    graph = core.PyGraph()
    return PyGraphWrapper(graph)

# Current Version
__version__ = "0.1.0"
