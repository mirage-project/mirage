"""
Operator module for representing computational graph nodes. This defines
the IR for our implementation.

This module defines the Operator class, which represents a single operation
in a computational graph. Each operator contains information about its inputs,
outputs, tensor shapes, and can be serialized to/from JSON format.
"""

import json

class Operator:
    """
    Represents a computational graph operator with inputs, outputs, and metadata.
    
    An Operator encapsulates a single operation in a computational graph, storing
    references to connected operators, tensor shape information, and operation-specific
    parameters. Supports JSON serialization for graph persistence.
    
    Attributes:
        name (str): Unique identifier for the operator.
        fn (str): Function or operation type (e.g., 'conv', 'matmul', 'relu').
        input_ops (list[Operator]): List of operators providing input to this operator.
        output_ops (list[Operator]): List of operators consuming output from this operator.
        input_tensor_shapes (list): Shapes of input tensors.
        output_tensor_shapes (list): Shapes of output tensors.
        additional_params (dict): Operation-specific parameters.
        kwargs (dict): Additional keyword arguments for the operation.
    """
    def __init__(self, name=None, fn=None, input_ops=[], output_ops=[], input_tensor_shapes=[], output_tensor_shapes=[], additional_params={}, kwargs={}):
        """
        Initialize an Operator instance.
        
        Args:
            name (str, optional): Unique identifier for the operator.
            fn (str, optional): Operation type or function name.
            input_ops (list[Operator], optional): Operators providing input. Defaults to [].
            output_ops (list[Operator], optional): Operators consuming output. Defaults to [].
            input_tensor_shapes (list, optional): Shapes of input tensors. Defaults to [].
            output_tensor_shapes (list, optional): Shapes of output tensors. Defaults to [].
            additional_params (dict, optional): Operation-specific parameters. Defaults to {}.
            kwargs (dict, optional): Additional keyword arguments. Defaults to {}.
        
        Note:
            Using mutable default arguments is generally discouraged in Python. Consider
            using None as default and initializing with empty containers in the method body.
        """
        self.name = name
        self.fn = fn
        self.input_ops = input_ops
        self.output_ops = output_ops
        self.additional_params = additional_params
        self.kwargs = kwargs
        self.input_tensor_shapes = input_tensor_shapes
        self.output_tensor_shapes = output_tensor_shapes
    
    def __json__(self):
        """
        Convert operator to a JSON-serializable dictionary.
        
        Converts the operator's attributes to a dictionary format suitable for JSON
        serialization. Replaces operator object references in input_ops and output_ops
        with their string names to avoid circular references.
        
        Returns:
            dict: Dictionary representation with operator names instead of objects.
        """
        op_dict = self.__dict__.copy()
        op_dict["input_ops"] = [op.name for op in op_dict["input_ops"]]
        op_dict["output_ops"] = [op.name for op in op_dict["output_ops"]]
        return op_dict
    
    def __repr__(self):
        """
        Generate a human-readable string representation of the operator.
        
        Returns:
            str: Multi-line formatted string showing operator's key attributes including
                 name, type, connected operators, and tensor shapes.
        """
        return f"\nNAME: {self.name}\nTYPE: {self.fn}\nIN_OPS: {[op.name for op in self.input_ops]}\nOUT_OPS: {[op.name for op in self.output_ops]}\nIN_TENSORS: {self.input_tensor_shapes}\nOUT_TENSORS: {self.output_tensor_shapes}\n"
        
    def to_json(self):
        """
        Serialize the operator to a JSON string.
        
        Returns:
            str: JSON string representation of the operator using the __json__ method.
        """
        return json.dumps(self, default=lambda obj : obj.__json__())

    def from_json(self, op):
        """
        Deserialize operator attributes from JSON data.
        
        Populates the current operator instance with attributes from a JSON string
        or dictionary. Note that this method modifies the current instance in place
        and does not reconstruct operator object references (input_ops and output_ops
        will remain as lists of names).
        
        Args:
            op (str or dict): JSON string or dictionary containing operator attributes.
        
        Returns:
            Operator: Returns self for method chaining.
        """
        if isinstance(op, str):
            op = json.loads(op)
        for key, value in op.items():
            setattr(self, key, value)
        return self
