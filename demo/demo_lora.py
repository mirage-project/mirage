import mirage as mi
import argparse
import os

def optimize_lora(checkpoint):
    graph = mi.new_graph()
    X = graph.new_input(dims=(16, 256), dtype=mi.float16)
    W = graph.new_input(dims=(256, 4096), dtype=mi.float16)
    A = graph.new_input(dims=(256, 16), dtype=mi.float16)
    B = graph.new_input(dims=(16, 4096), dtype=mi.float16)
    D = graph.matmul(X, A)
    E = graph.matmul(D, B)
    C = graph.matmul(X, W)
    if checkpoint is None:
        graphs = mi.optimize(graph, default_config="lora")
    else:
        graphs = mi.optimize(graph, previous_checkpoint=checkpoint)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    for i, graph in enumerate(graphs):
        graph.generate_triton_program("{}/generated_program_{}.py".format(dir_path, i))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    optimize_lora(args.checkpoint)
