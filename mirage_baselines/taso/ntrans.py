import taso

batch_size = 1

def get_justnorm(graph, X):
    # use l1 norm to approximate the runtime cost
    return graph.div(x=X, y=graph.reduce_sum(input=X, axes=(1,)))

if __name__ == "__main__":
    graph = taso.new_graph()
    H = graph.new_input(dims=(8 * batch_size, 4096))
    X = graph.new_input(dims=(8 * batch_size, 4096))
    alpha = graph.new_input(dims=(8 * batch_size, 4096))
    H_norm = get_justnorm(graph, H)
    A = graph.sub(x=H_norm, y=X)
    B = graph.mul(alpha, A)
    C = graph.add(X, B)
    O = get_justnorm(graph, C)
    
    taso.optimize(graph, budget=1)
    