class Operator:
    def __init__(self, name=None, fn=None, input_ops=[], output_ops=[], input_tensor_shapes=[], output_tensor_shapes=[], additional_params=[], kwargs={}):
        self.name = name
        self.fn = fn
        self.input_ops = input_ops
        self.output_ops = output_ops
        self.additional_params = additional_params
        self.kwargs = kwargs
        self.input_tensor_shapes = input_tensor_shapes
        self.output_tensor_shapes = output_tensor_shapes