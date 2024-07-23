import torch
from .core import optimize

class PyGraphWrapper:  
    def __init__(self, graph):  
        self.pygraph = graph
        
        self._is_compiled = False
        self.run = None
        self._cached_buffer_size = None
  
    def __getattr__(self, name):  
        return getattr(self.pygraph, name)
    
    def execute(self, **kwargs):
        buffer_size = self.compile(**kwargs)
        
        assert self.run is not None, "The graph is not compiled yet."
        
        # TODO: set buffer_size to a given value
        
        input_tensors = kwargs.get("inputs", [])
        output_nodes = kwargs.get("outputs", [])
        
        # TODO: dtype and device
        buffer_tensor = torch.empty(buffer_size, dtype=torch.uint8, device=input_tensors[0].device).contiguous()
        output_tensors = [
            torch.empty([node.dim(i) for i in range(node.num_dims)], 
                        dtype=torch.float16, device=input_tensors[0].device) for node in output_nodes]
        print("storage ptr of output[0]:", hex(output_tensors[0].storage().data_ptr()))
        print("Shapes of output tensors:", [tensor.shape for tensor in output_tensors])
        
        buffer_tensor_ptr = buffer_tensor.data_ptr()
        input_tensors_ptr = [tensor.data_ptr() for tensor in input_tensors]
        output_tensors_ptr = [tensor.data_ptr() for tensor in output_tensors]
        
        print("Input tensors ptr:", [hex(ptr) for ptr in input_tensors_ptr])
        print("Output tensors ptr:", [hex(ptr) for ptr in output_tensors_ptr])
        print("Buffer tensor ptr:", hex(buffer_tensor_ptr))
        print("Buffer size:", buffer_tensor.storage().size())
        
        self.run(input_tensors_ptr, output_tensors_ptr, buffer_tensor_ptr)
        
        return output_tensors
    
    def compile(self, **kwargs):
        if self._is_compiled:
            return self._cached_buffer_size
        
        # graphs = optimize(self.pygraph, **kwargs)
        # for i, graph in enumerate(graphs):
        #     graph.generate_cuda_program("generated_program_{}.cu".format(i))
        
        # TODO
        
        so_path = '/data2/sft/InternData/qinyanzhao/spiritedaway/mirage/build/test.cpython-38-x86_64-linux-gnu.so'
        
        print("Loading the shared object file from path: {}".format(so_path))
        import importlib.util
        spec = importlib.util.spec_from_file_location("__mirage_launcher", so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.run = getattr(mod, "launch")
        
        self._is_compiled = True
        self._cached_buffer_size = 268435456
        return self._cached_buffer_size