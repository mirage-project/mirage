import networkx as nx
import matplotlib.pyplot as plt
import json
import argparse
import os

def edge_info(edge):
    dim = edge['dim'][:3]
    layout = edge['layout']
    return f"{dim}\n{layout}"


def get_graph(data):
    node_labels = {idx: node['op_type'] for idx, node in enumerate(data)}
    edge_labels = {(edge_owner[input_tensor["guid"]], idx): edge_info(input_tensor) for idx, node in enumerate(data) for input_tensor in node["input_tensors"]}
    return node_labels, edge_labels


def get_idx(kn_idx, tb_idx):
    return kn_idx * 20 + tb_idx


def input_info(op_type, grid_dim, forloop_range, forloop_dim, input_map):
    return f"{op_type}\ngrid:{grid_dim['x']},{grid_dim['y']},{grid_dim['z']}\nforrange: {forloop_range}\nfordim: {forloop_dim}\nimap:{input_map['x']},{input_map['y']},{input_map['z']}"


def output_info(op_type, output_map):
    return f"{op_type}\nomap:{output_map['x']},{output_map['y']},{output_map['z']}"


def get_op_info(op):
    if op['op_type'] == 'reduction':
        return f"{op['op_type']}\n{op['reduce_dim']}"
    else:
        return op['op_type']


def process_graph(data):
    edge_owner = dict()
    for kn_idx, kn_op in enumerate(data):
        if kn_op['op_type'] == 'kn_customized_op':
            for tb_idx, tb_op in enumerate(kn_op['bgraph']['operators']):
                for tensor in tb_op['output_tensors']:
                    edge_owner[tensor['guid']] = get_idx(kn_idx, tb_idx)
                if tb_op['op_type'] == 'tb_output_op':
                    edge_owner[tb_op['dtensor']['guid']] = get_idx(kn_idx, tb_idx)
        else:
            for tensor in kn_op['output_tensors']:
                edge_owner[tensor['guid']] = kn_idx

    print(edge_owner)

    node_labels = dict()
    edge_labels = dict()
    node_color = dict()

    for kn_idx, kn_op in enumerate(data):
        if kn_op['op_type'] == 'kn_customized_op':
            for tb_idx, tb_op in enumerate(kn_op['bgraph']['operators']):
                node_labels[get_idx(kn_idx, tb_idx)] = \
                    input_info(tb_op['op_type'], kn_op['bgraph']['grid_dim'], kn_op['bgraph']['forloop_range'], tb_op['forloop_dim'], tb_op['input_map']) if tb_op['op_type'] == 'tb_input_op' \
                    else output_info(tb_op['op_type'], tb_op['output_map']) if tb_op['op_type'] == 'tb_output_op' \
                    else get_op_info(tb_op)
                node_color[get_idx(kn_idx, tb_idx)] = (0.5, 1, 0.5)
                if tb_op['op_type'] == 'tb_input_op':
                    edge_labels[(edge_owner[tb_op['dtensor']['guid']], get_idx(kn_idx, tb_idx))] = edge_info(tb_op['dtensor'])
                for input_tensor in tb_op['input_tensors']:
                    edge_labels[(edge_owner[input_tensor['guid']], get_idx(kn_idx, tb_idx))] = edge_info(input_tensor)
        else:
            node_labels[kn_idx] = kn_op['op_type']
            node_color[kn_idx] = (0.5, 0.5, 1)
            for input_tensor in kn_op['input_tensors']:
                edge_labels[(edge_owner[input_tensor['guid']], kn_idx)] = edge_info(input_tensor)

    return node_labels, edge_labels, node_color


def draw_graph(graph, filename):
    node_labels, edge_labels, node_color = process_graph(graph)

    nodes = list(node_color.keys())
    color_list = list(node_color.values())

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(list(edge_labels.keys()))
    plt.figure(figsize=(30, 20))

    pos = nx.nx_agraph.graphviz_layout(G)
    nx.draw_networkx(G, pos, node_color=color_list, node_shape='s', node_size=4000, arrows=True, arrowsize=50, arrowstyle='-|>', with_labels=False)
    nx.draw_networkx_labels(G, pos, node_labels, font_size=12)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=12, label_pos=0.6)
    plt.savefig(filename)


parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('-i', type=str, help='Figure directory')
parser.add_argument('-b', action='store_true')

args = parser.parse_args()

with open(args.filename, 'r') as file:
    data = json.load(file)
    if args.b:
        for idx, graph in enumerate(data):
            output_filename = os.path.join(args.i, f'graph{idx}.png')
            draw_graph(graph, output_filename)
    else:
        output_filename = os.path.join(args.i, f'graph.png')
        draw_graph(data, output_filename)
