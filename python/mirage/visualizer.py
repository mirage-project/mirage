import json
import graphviz as gv

colors_map = {
    "kernel": {
        "node": "#70a148",
        "bg": "#e0edd5",
        "edge": "#527536",
        "edge_label": "black",
        "io": "#527536",
    },
    "block": {
        "node": "#5a8fcb",
        "bg": "#dbe8f5",
        "edge": "#4273b1",
        "edge_label": "black",
        "io": "#527536",
    },
    "thread": {
        "node": "#f5c342",
        "bg": "#fdf2d0",
        "edge": "#b89230",
        "edge_label": "black",
        "io": "#4273b1",
    }
}

node_font_size = "30"
edge_font_size = "30"
tensor_node_font_size = "23"
graph_label_font_size = "32"

op_nodelabel_mapping = {
    "kn_unkown": "Unknown",
    "kn_input_op": "Input",
    "kn_output_op": "Output",
    "kn_matmul_op": "MatMul",
    "kn_exp_op": "Exp",
    "kn_square_op": "Square",
    "kn_sqrt_op": "Sqrt",
    "kn_mul_scalar_op": "MulScalar",
    "kn_silu_op": "SiLU",
    "kn_sigmoid_op": "Sigmoid",
    "kn_gelu_op": "GeLU",
    "kn_relu_op": "ReLU",
    "kn_clamp_op": "Clamp",
    "kn_log_op": "Log",
    "kn_add_op": "Add",
    "kn_mul_op": "Multiply",
    "kn_div_op": "Div",
    "kn_pow_op": "Pow",
    "kn_reduction_0_op": "Reduction 0",
    "kn_reduction_1_op": "Reduction 1",
    "kn_reduction_2_op": "Reduction 2",
    "kn_rms_norm_op": "RMS Norm",
    "kn_concat_first_op_id": "Concat First",
    "kn_concat_0_op": "Concat 0",
    "kn_concat_1_op": "Concat 1",
    "kn_concat_2_op": "Concat 2",
    "kn_concat_last_op_id": "Concat Last",
    "kn_split_first_op_id": "Split First",
    "kn_split_0_op": "Split 0",
    "kn_split_1_op": "Split 1",
    "kn_split_2_op": "Split 2",
    "kn_chunk_0_op": "Chunk 0",
    "kn_chunk_1_op": "Chunk 1",
    "kn_chunk_2_op": "Chunk 2",
    "kn_split_last_op_id": "Split Last",
    "kn_allreduce_op": "AllReduce",
    "kn_customized_op": "Customized\nOp ",
    "tb_unkown": "Unknown",
    "tb_input_op": "Input",
    "tb_output_op": "Output",
    "tb_matmul_op": "MatMul",
    "tb_exp_op": "Exp",
    "tb_square_op": "Square",
    "tb_sqrt_op": "Sqrt",
    "tb_mul_scalar_op": "MulScalar",
    "tb_silu_op": "SiLU",
    "tb_sigmoid_op": "Sigmoid",
    "tb_gelu_op": "GeLU",
    "tb_relu_op": "ReLU",
    "tb_clamp_op": "Clamp",
    "tb_log_op": "Log",
    "tb_add_op": "Add",
    "tb_mul_op": "Multiply",
    "tb_div_op": "Div",
    "tb_sub_op": "Sub",
    "tb_pow_op": "Pow",
    "tb_reduction_first_op_id": "Reduction First",
    "tb_reduction_0_op": "Reduction 0",
    "tb_reduction_1_op": "Reduction 1",
    "tb_reduction_2_op": "Reduction 2",
    "tb_reduction_0_to_dimx_op": "Reduction 0\nto DimX",
    "tb_reduction_1_to_dimx_op": "Reduction 1\nto DimX",
    "tb_reduction_2_to_dimx_op": "Reduction 2\nto DimX",
    "tb_reduction_0_max_op": "Reduction 0 Max",
    "tb_reduction_1_max_op": "Reduction 1 Max",
    "tb_reduction_2_max_op": "Reduction 2 Max",
    "tb_reduction_last_op_id": "Reduction Last",
    "tb_rms_norm_op": "RMS Norm",
    "tb_concat_first_op_id": "Concat First",
    "tb_concat_0_op": "Concat 0",
    "tb_concat_1_op": "Concat 1",
    "tb_concat_2_op": "Concat 2",
    "tb_concat_last_op_id": "Concat Last",
    "tb_concat_then_matmul_op": "Concat Then MatMul",
    "tb_split_first_op_id": "Split First",
    "tb_split_0_op": "Split 0",
    "tb_split_1_op": "Split 1",
    "tb_split_2_op": "Split 2",
    "tb_split_last_op_id": "Split Last",
    "tb_forloop_accum_no_red_op": "ForloopAccum\n(No Reduction)",
    "tb_forloop_accum_red_ld_sum_op": "ForloopAccum\n(Reduction=Sum)",
    "tb_forloop_accum_red_ld_mean_op": "ForloopAccum\n(Reduction=Mean)",
    "tb_forloop_accum_red_ld_rms_op": "ForloopAccum\n(Reduction=RMS)",
    "tb_forloop_accum_redtox_ld_sum_op": "ForloopAccum\n(ReduceToDimx=Sum)",
    "tb_forloop_accum_no_red_rescale_op": "ForloopAccumRescale\n(No Reduction)",
    "tb_forloop_accum_red_ld_sum_rescale_op": "ForloopAccumRescale\n(Reduction=Sum)",
    "tb_forloop_accum_max_op": "ForloopAccum\nMax",
    "tb_forloop_accum_last_op": "ForloopAccum\nLast",
    "tb_customized_op": "Customized\nOp",
}
guid_tensors_map = {}

tensor_name_suffix = "'"
phi_symbol = "\u2205"
arrow_symbol = "↔"

def draw_edge(G, from_node, to_node, graph_type, label=None):
    G.edge(from_node, to_node, color=colors_map[graph_type]["edge"], penwidth="6", 
           label=label, fontname="sans-serif", fontsize=edge_font_size, fontcolor=colors_map[graph_type]["edge_label"])

def get_format_str(operator_data):
    s = ""
    if operator_data['forloop_dim'] >= 0:
        s += str(operator_data['forloop_dim'])
    else:
        s += '\u2205'
    return f"fmap: [i↔{s}]"

def letter_sequence():
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    index = 0
    while True:
        yield alphabet[index % len(alphabet)]
        index += 1

def is_graph_data(item):
    return isinstance(item, dict) and "op_type" in item

class node:
    def __init__(self, name, op_type, id, label, color):
        self.name = name # Unique name
        self.op_type = op_type
        self.id = id
        self.label = label # Shown in the graph
        self.input_tensors = []
        self.output_tensors = []
        self.color = color

    def is_customized_node(self):
        return "customized" in self.op_type
    
    def is_input_node(self):
        return "input" in self.op_type

    def is_output_node(self):
        return "output" in self.op_type
          
class kernel_node(node):
    def __init__(self, name, op_type, id, label):
        super().__init__(name, op_type, id, label, colors_map["kernel"]["node"])

    def is_kernel_output_node(self):
        return not self.output_tensors

    def draw(self, G):
        if not self.is_output_node():
            G.node(self.name, label=self.label, color=self.color, style="rounded,filled", shape="box",
                penwidth="0", fontsize=node_font_size, fontcolor="white", fontname="sans-serif", margin="0.4,0.4")
        
class block_node(node):
    def __init__(self, name, op_type, id, label, iomap=None, forloop_dim=None, forloop_range=None):
        super().__init__(name, op_type, id, label, colors_map["block"]["node"])
        # Only input and output nodes have iomap
        if self.is_input_node() or self.is_output_node():
            self.iomap_str = self.get_iomap_str(iomap)
            self.original_tensor = None
        # Only input nodes have forloop_dim and forloop_range
        if self.is_input_node():
            self.forloop_dim = forloop_dim
            self.forloop_range = forloop_range
            self.formap_str = self.get_formap_str()

    def get_iomap_str(self, io_map):
        map_entries = []
        for key, value in io_map.items():
            if value == -1:
                map_entries.append(f"{key}{arrow_symbol}{phi_symbol}")
            else:
                map_entries.append(f"{key}{arrow_symbol}{value}")
        if self.is_input_node():
            map_string = "imap: {" + ", ".join(map_entries) + "}"
        else:
            map_string = "omap: {" + ", ".join(map_entries) + "}"
        return map_string
    
    def get_formap_str(self):
        s = ""
        if self.forloop_dim >= 0:
            s += str(self.forloop_dim)
        else:
            s += phi_symbol
        return f"fmap: [i{arrow_symbol}{s}]"

    def draw(self, G):
        tensor_color = colors_map["kernel"]["io"]
        if self.is_input_node():
            output_shape = self.output_tensors[0].shape
            shape_before_loop = output_shape.copy()
            if self.forloop_dim >= 0:
                shape_before_loop[self.forloop_dim] = output_shape[self.forloop_dim] * self.forloop_range

            tensor_node_name = self.original_tensor.name + "'_input"
            G.node(tensor_node_name, label=self.original_tensor.name + "'\n" + str(shape_before_loop) + "\n" + self.iomap_str,
                     color=tensor_color, style="filled", shape="box", penwidth="0", fontsize=tensor_node_font_size, fontcolor="white", fontname="sans-serif")
            G.node(self.name, label=self.label + "\n" + self.formap_str, color=self.color, style="rounded,filled",
                   shape="box", penwidth="0", fontsize=node_font_size, fontcolor="white", fontname="sans-serif", margin="0.4,0.4")
            draw_edge(G, tensor_node_name, self.name, "block")
        elif self.is_output_node():
            tensor_node_name = self.original_tensor.name + "'_output"
            G.node(self.name, label=self.label, color=self.color, style="rounded,filled",
                   shape="box", penwidth="0", fontsize=node_font_size, fontcolor="white", fontname="sans-serif", margin="0.4,0.4")
            G.node(tensor_node_name, label=self.original_tensor.name + "'\n" + str(self.input_tensors[0].shape) + "\n" + self.iomap_str,
                     color=tensor_color, style="filled", shape="box", penwidth="0", fontsize=tensor_node_font_size, fontcolor="white", fontname="sans-serif")
            draw_edge(G, self.name, tensor_node_name, "block")
        else:
            G.node(self.name, label=self.label, color=self.color, style="rounded,filled",
                     shape="box", penwidth="0", fontsize=node_font_size, fontcolor="white", fontname="sans-serif", margin="0.4,0.4")

class tensor:
    def _init_(self, guid, color, shape):
        self.guid = guid
        self.color = color
        self.last_node = None
        self.next_nodes = []
        self.shape = shape

class kernel_tensor(tensor):
    def __init__(self, guid, shape):
        super()._init_(guid, colors_map["kernel"]["io"], shape)
        self.name = None
        guid_tensors_map[guid] = self
    
    def draw(self, G):
        if self.next_nodes:
            for next_node in self.next_nodes:
                if self.last_node.is_customized_node() or next_node.is_customized_node():
                    G.node(self.name, label=self.name + "\n" + str(self.shape), color=self.color, style="filled", shape="box",
                        penwidth="0", fontsize=tensor_node_font_size, fontcolor="white", fontname="sans-serif")
                    if not (self.last_node.is_input_node()):
                        draw_edge(G, self.last_node.name, self.name, "kernel")
                    if not (next_node.is_output_node()):
                        draw_edge(G, self.name, next_node.name, "kernel")
                elif self.last_node.is_input_node():
                    if not self.name:
                        self.name = str(self.guid)
                    G.node(self.name, label=str(self.shape), color=self.color, style="filled", shape="box",
                        penwidth="0", fontsize=tensor_node_font_size, fontcolor="white", fontname="sans-serif")
                    if not (next_node.is_output_node()):
                        draw_edge(G, self.name, next_node.name, "kernel")
                elif next_node.is_output_node():
                    if not self.name:
                        self.name = str(self.guid)
                    G.node(self.name, label=str(self.shape), color=self.color, style="filled", shape="box",
                        penwidth="0", fontsize=tensor_node_font_size, fontcolor="white", fontname="sans-serif")
                    draw_edge(G, self.last_node.name, self.name, "kernel")
                else:
                    draw_edge(G, self.last_node.name, next_node.name, "kernel", label=str(self.shape))
class block_tensor(tensor):
    def __init__(self, guid, shape):
        super()._init_(guid, colors_map["block"]["edge"], shape)
        guid_tensors_map[guid] = self
    
    def draw(self, G):
        for next_node in self.next_nodes:
           draw_edge(G, self.last_node.name, next_node.name, "block", label=str(self.shape))


class graph:
    def __init__(self, label, bg_color, edge_color):
        self.label = label
        self.bg_color = bg_color
        self.nodes = []
        self.tensors = []
        self.edge_color = edge_color

class kernel_graph(graph):

    def __init__(self, label):
        super().__init__(label, colors_map["kernel"]["bg"], colors_map["kernel"]["edge"])
        self.block_graphs = []
        self.letter_sequence = letter_sequence()

    def read_nodes(self, graph_data):
        block_graph_datas_to_handle = []
        for node_data in graph_data:
            node_id = id(node_data)
            node_op_type = node_data["op_type"]
            node_name = f"{node_op_type}_{node_id}"
            new_node = kernel_node(node_name, node_op_type, node_id, op_nodelabel_mapping[node_op_type])
            for input_tensor in node_data["input_tensors"]:
                tensor_guid = input_tensor["guid"]
                if tensor_guid in guid_tensors_map:
                    tensor = guid_tensors_map[tensor_guid]
                else:
                    tensor = kernel_tensor(tensor_guid, input_tensor["dim"][:input_tensor["num_dims"]])
                if new_node.is_customized_node() and not tensor.name:
                    tensor.name = next(self.letter_sequence)
                tensor.next_nodes.append(new_node)
                if tensor_guid not in guid_tensors_map:
                    guid_tensors_map[tensor_guid] = tensor
                if tensor not in self.tensors:
                    self.tensors.append(tensor)
                new_node.input_tensors.append(tensor)
            for output_tensor in node_data["output_tensors"]:
                tensor_guid = output_tensor["guid"]
                if tensor_guid in guid_tensors_map:
                    tensor = guid_tensors_map[tensor_guid]
                else:
                    tensor = kernel_tensor(tensor_guid, output_tensor["dim"][:output_tensor["num_dims"]])
                if new_node.is_customized_node() and not tensor.name:
                    tensor.name = next(self.letter_sequence)
                tensor.last_node = new_node
                if tensor_guid not in guid_tensors_map:
                    guid_tensors_map[tensor_guid] = tensor
                if tensor not in self.tensors:
                    self.tensors.append(tensor)
                new_node.output_tensors.append(tensor)
            
            self.nodes.append(new_node)

            if "bgraph" in node_data:
                grid_dim = node_data["bgraph"]["grid_dim"]
                forloop_range = node_data["bgraph"]["forloop_range"]
                new_block_graph = block_graph("Block graph "+str(len(block_graph_datas_to_handle)+1),
                                              grid_dim, forloop_range, self)
                new_node.related_node = new_block_graph
                new_node.label += " "+str(len(block_graph_datas_to_handle)+1)
                block_graph_datas_to_handle.append((new_block_graph, node_data["bgraph"]["operators"]))

        for new_block_graph, block_graph_data in block_graph_datas_to_handle:
            new_block_graph.read_nodes(block_graph_data)
            self.block_graphs.append(new_block_graph)

    def draw_graph(self, G):
        # Draw block graphs from back to front
        for i in range(len(self.block_graphs) - 1, -1, -1):
            self.block_graphs[i].draw_graph(G)

        with G.subgraph(name="cluster" + self.label) as sub:
            sub.attr(rankdir='LR', splines='ortho', bgcolor=self.bg_color, fontname="sans-serif", 
            label=self.label, labelloc='t', labeljust='l', labeldistance="1.5", fontsize=graph_label_font_size, fontcolor="black", 
            style="filled", penwidth="0")
            for node in self.nodes:
                if node.is_input_node():
                    continue
                else:
                    node.draw(sub)
            for tensor in self.tensors:
                tensor.draw(sub)

class block_graph(graph):

    def __init__(self, label, grid_dim, forloop_range, kernel_graph):
        super().__init__(label, colors_map["block"]["bg"], colors_map["block"]["edge"])
        self.related_node = None
        self.grid_dim = grid_dim
        self.forloop_range = forloop_range
        self.kernel_graph = kernel_graph

    def get_grid_size_and_forloop(self):
        grid_size_str = ""
        forloop_str = ""
        grid_size_str = f"grid size: [{', '.join([f'{k}={v}' for k, v in self.grid_dim.items()])}]"

        forloop_str = f"forloop: [i={self.forloop_range}]"

        return "; " + grid_size_str + "; " + forloop_str

    def read_nodes(self, graph_data):
        nodes = []
        for node_data in graph_data:
            node_id = id(node_data)
            node_op_type = node_data["op_type"]
            node_name = f"{node_op_type}_{node_id}"
            io_map = None
            if "input_map" in node_data:
                io_map = node_data["input_map"]
            elif "output_map" in node_data:
                io_map = node_data["output_map"]
            forloop_dim = None if "forloop_dim" not in node_data else node_data["forloop_dim"]
            new_node = block_node(node_name, node_op_type, node_id, op_nodelabel_mapping[node_op_type],
                                  io_map, forloop_dim, self.forloop_range)
            for output_tensor in node_data["output_tensors"]:
                tensor_guid = output_tensor["guid"]
                if tensor_guid in guid_tensors_map:
                    tensor = guid_tensors_map[tensor_guid]
                else:
                    tensor = block_tensor(tensor_guid, output_tensor["dim"][:output_tensor["num_dims"]])
                tensor.last_node = new_node
                if tensor_guid not in guid_tensors_map:
                    guid_tensors_map[tensor_guid] = tensor
                if tensor not in self.tensors:
                    self.tensors.append(tensor)
                new_node.output_tensors.append(tensor)
            for input_tensor in node_data["input_tensors"]:
                tensor_guid = input_tensor["guid"]
                if tensor_guid in guid_tensors_map:
                    tensor = guid_tensors_map[tensor_guid]
                else:
                    tensor = block_tensor(tensor_guid, input_tensor["dim"][:input_tensor["num_dims"]])
                tensor.next_nodes.append(new_node)
                if tensor_guid not in guid_tensors_map:
                    guid_tensors_map[tensor_guid] = tensor
                if tensor not in self.tensors:
                    self.tensors.append(tensor)
                new_node.input_tensors.append(tensor)
            if "dtensor" in node_data:
                new_node.original_tensor = guid_tensors_map[node_data["dtensor"]["guid"]]

            self.nodes.append(new_node)
        return nodes

    def draw_graph(self, G):
        with G.subgraph(name="cluster" + self.label) as sub:
            sub.attr(rankdir='LR', splines='ortho', bgcolor=self.bg_color, fontname="sans-serif", 
            label=self.label + self.get_grid_size_and_forloop(), labelloc='t', labeljust='l',
            labeldistance="1.5", fontsize=graph_label_font_size, fontcolor="black", style="filled", penwidth="0")
            for node in self.nodes:
                node.draw(sub)
            for tensor in self.tensors:
                tensor.draw(sub)

class visualizer:
    def __init__(self, output_filename):
        self.graphs = []
        self.output_filename = output_filename
        self.letter_sequence = letter_sequence()
        self.G = gv.Digraph(format='png', name="Kernel Graph")
        self.G.attr(rankdir='LR', splines='ortho', bgcolor="#ffffff", fontname="sans-serif",
           nodesep="1.6", ranksep="0.3", fontsize="16", fontcolor="black", compound="true")
        self.new_kernel_graph = kernel_graph("Kernel Graph")

    def draw_graphs(self, operators, dot=True, png=True):
        self.new_kernel_graph.read_nodes(operators)
        self.new_kernel_graph.draw_graph(self.G)
        if dot:
            self.G.save(self.output_filename + ".dot")
            print(f"Graph saved as {self.output_filename}.dot")
        if png:
            self.G.render(self.output_filename, cleanup=True)
            print(f"Graph saved as {self.output_filename}.png")
        


def handle_graph_data(graph_data, graph_title, output_filename, dot=True, png=True):
    G = gv.Digraph(format='png', name=graph_title)
    G.attr(rankdir='LR', splines='ortho', bgcolor="#ffffff", fontname="sans-serif",
           nodesep="1.6", ranksep="0.3", fontsize="16", fontcolor="black", compound="true")
    
    new_kernel_graph = kernel_graph("Kernel Graph")
    new_kernel_graph.read_nodes(graph_data)
    new_kernel_graph.draw_graph(G)
    
    if dot:
        G.save(output_filename + ".dot")
        print(f"Graph saved as {output_filename}.dot")
    if png:
        G.render(output_filename, cleanup=True)
        print(f"Graph saved as {output_filename}.png")


if __name__ == "__main__":
    file_name = "mirage_search_checkpoint.json"
    with open(file_name) as f:
        data = json.load(f)

    if isinstance(data, list):
        if all(isinstance(item, list) for item in data):
            for idx, graph_list in enumerate(data):
                handle_graph_data(graph_list, graph_title=f"Combined graph {idx+1}", output_filename=f"reframe_outcome/reframe_combined_graph_{idx+1}")
        elif all(is_graph_data(item) for item in data):
            handle_graph_data(data, graph_title="Combined graph", output_filename="reframe_outcome/reframe_combined_graph")
        else:
            print("Invalid data format.")
    else:
        print("Invalid data format.")
