import json

class Operator:
    def __init__(self, name=None, fn=None, input_ops=[], output_ops=[], input_tensor_shapes=[], output_tensor_shapes=[], additional_params={}, kwargs={}):
        self.name = name
        self.fn = fn
        self.input_ops = input_ops
        self.output_ops = output_ops
        self.additional_params = additional_params
        self.kwargs = kwargs
        self.input_tensor_shapes = input_tensor_shapes
        self.output_tensor_shapes = output_tensor_shapes
    
    def __json__(self):
        op_dict = self.__dict__.copy()
        op_dict["input_ops"] = [op.name for op in op_dict["input_ops"]]
        op_dict["output_ops"] = [op.name for op in op_dict["output_ops"]]
        return op_dict
    
    def __repr__(self):
        return f"\nNAME: {self.name}\nTYPE: {self.fn}\nIN_OPS: {[op.name for op in self.input_ops]}\nOUT_OPS: {[op.name for op in self.output_ops]}\nIN_TENSORS: {self.input_tensor_shapes}\nOUT_TENSORS: {self.output_tensor_shapes}\n"
        
    def to_json(self):
        return json.dumps(self, default=lambda obj : obj.__json__())

    def from_json(self, op):
        if isinstance(op, str):
            op = json.loads(op)
        for key, value in op.items():
            setattr(self, key, value)
        return self
