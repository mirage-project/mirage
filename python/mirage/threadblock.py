from .core import *

class TBGraph:
    def __init__(self, graph):
        self.cygraph = graph

    def new_input(self, dtensor: DTensor, input_map: tuple, forloop_dim: int):
        return self.cygraph.new_input(dtensor, input_map, forloop_dim)

    def new_output(self, stensor: STensor, output_map: tuple, forloop_dim: int = -1):
        return self.cygraph.new_output(stensor, output_map, forloop_dim)

    def matmul(self, A: STensor, B: STensor):
        return self.cygraph.matmul(A, B)
    
    def matmul_e4m3(self, A: STensor, B: STensor, sa: STensor = None, sb: STensor = None):
        # find scaling using amax if necessary
        if not sa:
          sa = self.cygraph.amax(A)
        if not sb:
          sb = self.cygraph.amax(B)
        assert all(i == 1 for i in sa.shape), \
              "A blockwise scale factor too complex"
        assert all(i == 1 for i in sb.shape), \
              "B blockwise scale factor too complex"
        # quantize
        Aq = self.cygraph.div(A, sa)
        Aq = self.cygraph.clamp(self.cygraph.mul_scalar(Aq, 448.0), -448.0, 448.0)
        Bq = self.cygraph.div(B, sb)
        Bq = self.cygraph.clamp(self.cygraph.mul_scalar(Bq, 448.0), -448.0, 448.0)
        A = self.cygraph.tofp8(Aq)
        B = self.cygraph.tofp8(Bq)
        
        # hopper matmul: A(fp8_e4m3) x B(fp8_e4m3) -> D(fp16)
        D = self.cygraph.matmul(A, B)
        
        # dequantize
        sd = self.cygraph.mul(sa, sb)
        sd = self.cygraph.mul_scalar(sd, 1/448.0)
        D = self.cygraph.mul(D, sd)
        return D

    def exp(self, A: STensor):
        return self.cygraph.exp(A)

    def silu(self, A: STensor):
        return self.cygraph.silu(A)

    def gelu(self, A: STensor):
        return self.cygraph.gelu(A)
    
    def relu(self, A: STensor):
        return self.cygraph.relu(A)
    
    def clamp(self, A: STensor, min_val: float, max_val: float):
        return self.cygraph.clamp(A, min_val, max_val)
    
    # def tofp8(self, A: STensor, scale: STensor):
    #     A_s = self.cygraph.div(A, scale) # scale dim should be all one
    #     return self.cygraph.clamp(A_s, -448.f, 448.f)
    
    # def fromfp8(self, A: STensor, scale: STensor):
    #     return self.cygraph.mul(A, scale)
      
    def square(self, A: STensor):
        return self.cygraph.square(A)

    def sqrt(self, A: STensor):
        return self.cygraph.sqrt(A)

    def add(self, A: STensor, B: STensor):
        return self.cygraph.add(A, B)

    def mul(self, A: STensor, B: STensor):
        return self.cygraph.mul(A, B)

    def div(self, A: STensor, B: STensor):
        return self.cygraph.div(A, B)

    def reduction(self, A: STensor, dim: int):
        return self.cygraph.reduction(A, dim)

    def rms_norm(self, A: STensor):
        return self.cygraph.rms_norm(A)

    def concat(self, A: STensor, B: STensor, dim: int):
        return self.cygraph.concat(A, B, dim)

    def forloop_accum(self, A: STensor, acc: str = None):
        return self.cygraph.forloop_accum(A, acc)
