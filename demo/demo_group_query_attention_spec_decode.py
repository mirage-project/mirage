import mirage as mi
import argparse
import os

def optimize_llama_70B(checkpoint):
    graph = mi.new_graph()
    Q = graph.new_input(dims=(2, 256, 64), dtype=mi.float16)
    K = graph.new_input(dims=(2, 64, 4096), dtype=mi.float16)
    V = graph.new_input(dims=(2, 4096, 64), dtype=mi.float16)
    A = graph.matmul(Q, K)
    E = graph.exp(A)
    S = graph.reduction(E, 2)
    D = graph.div(E, S)
    O = graph.matmul(D, V)
    if checkpoint is None:
        graphs = mi.optimize(graph, griddims=[(2, 16, 1), (2, 16, 4)])
    else:
        graphs = mi.optimize(graph, griddims=[(2, 16, 1), (2, 16, 4)], previous_checkpoint=checkpoint)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    for i, graph in enumerate(graphs):
        graph.generate_triton_program("{}/generated_program_{}.py".format(dir_path, i))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    optimize_llama_70B(args.checkpoint)
