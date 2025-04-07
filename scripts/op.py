import json

class Operator:
    def __init__(self, name=None, fn=None, input_ops=[], output_ops=[], input_tensor_shapes=[], output_tensor_shapes=[]):
        self.name = name
        self.fn = fn
        self.input_ops = input_ops
        self.output_ops = output_ops
        self.input_tensor_shapes = input_tensor_shapes
        self.output_tensor_shapes = output_tensor_shapes
    
    def __json__(self):
        op_dict = self.__dict__.copy()
        op_dict["input_ops"] = [op.name for op in op_dict["input_ops"]]
        op_dict["output_ops"] = [op.name for op in op_dict["output_ops"]]
        return op_dict
    
    def to_json(self):
        return json.dumps(self, default=lambda obj : obj.__json__())
    
    def from_json(self, op):
        if isinstance(op, str):
            op = json.loads(op)
        for key, value in op.items():
            setattr(self, key, value)
        return self
    
class Graph:
    def __init__(self, graph_dict={}):
        self.graph_dict = graph_dict
    
    def __json__(self):
        op_dict = {}
        for op, _ in self.graph_dict.items():
            op_dict[op.name] = op
        out_dict = {"all_ops": op_dict, "graph": {}}
        for from_node, to_nodes in self.graph_dict.items():
            out_dict["graph"][from_node.name] = [to_node.name for to_node in to_nodes]
        return out_dict
    
    def to_json(self):
        return json.dumps(self, default=lambda obj : obj.__json__())

    def from_json(self, graph):
        if isinstance(graph, str):
            graph = json.loads(graph)
        all_ops = {}
        for op_name, op_json in graph["all_ops"].items():
            all_ops[op_name] = Operator().from_json(op_json)
        for _, op_node in all_ops.items():
            for i in range(len(op_node.input_ops)):
                if op_node.input_ops[i] in all_ops:
                    op_node.input_ops[i] = all_ops[op_node.input_ops[i]]
            for i in range(len(op_node.output_ops)):
                if op_node.output_ops[i] in all_ops:
                    op_node.output_ops[i] = all_ops[op_node.output_ops[i]]
            op_node.input_tensor_shapes = [(tuple(shape_id[0]), shape_id[1]) for shape_id in op_node.input_tensor_shapes]
            op_node.output_tensor_shapes = [(tuple(shape_id[0]), shape_id[1]) for shape_id in op_node.output_tensor_shapes]
        graph_dict = {}
        for from_name, to_names in graph["graph"].items():
            graph_dict[all_ops[from_name]] = [all_ops[to_name] for to_name in to_names]
        
        self.graph_dict = graph_dict
        return self
