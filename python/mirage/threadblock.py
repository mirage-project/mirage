from .core import *

class TBGraph:
    def __init__(self, graph):
        self.cygraph = graph

    def new_input(self, dtensor: PyDTensor, input_map: tuple, forloop_dim: int):
        return self.cygraph.new_input(dtensor, input_map, forloop_dim)

    def new_output(self, stensor: PySTensor, output_map: tuple, forloop_dim: int = -1):
        return self.cygraph.new_output(stensor, output_map, forloop_dim)

    def matmul(self, A: PySTensor, B: PySTensor):
        return self.cygraph.matmul(A, B)

    def exp(self, A: PySTensor):
        return self.cygraph.exp(A)

    def silu(self, A: PySTensor):
        return self.cygraph.silu(A)

    def square(self, A: PySTensor):
        return self.cygraph.square(A)

    def sqrt(self, A: PySTensor):
        return self.cygraph.sqrt(A)

    def add(self, A: PySTensor, B: PySTensor):
        return self.cygraph.add(A, B)

    def mul(self, A: PySTensor, B: PySTensor):
        return self.cygraph.mul(A, B)

    def div(self, A: PySTensor, B: PySTensor):
        return self.cygraph.div(A, B)

    def reduction(self, A: PySTensor, dim: int):
        return self.cygraph.reduction(A, dim)

    def concat(self, A: PySTensor, B: PySTensor, dim: int):
        return self.cygraph.concat(A, B, dim)

    def forloop_accum(self, A: PySTensor, acc: str = None):
        return self.cygraph.forloop_accum(A, acc)
