import taso

batch_size = 1

def get_rms_norm(graph, X):
    # use l1 norm to approximate the runtime cost
    return graph.div(x=X, y=graph.reduce_sum(input=X, axes=(1,)))

if __name__ == "__main__":
    graph = taso.new_graph()
    X = graph.new_input(dims=(2 * batch_size, 4096))
    W = graph.new_input(dims=(4096, 6144))
    D = get_rms_norm(graph, X)
    O = graph.matmul(D, W)
    
    taso.optimize(graph, budget=1)
    