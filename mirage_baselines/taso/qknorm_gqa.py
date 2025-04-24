import taso

batch_size = 1

def get_rms_norm(graph, X):
    # use l1 norm to approximate the runtime cost
    return graph.div(x=X, y=graph.reduce_sum(input=X, axes=(1,)))

if __name__ == "__main__":
    graph = taso.new_graph()
    Q = graph.new_input(dims=(2 * batch_size, 256, 64))
    K = graph.new_input(dims=(2 * batch_size, 64, 4096))
    V = graph.new_input(dims=(2 * batch_size, 4096, 64))
    nQ = get_rms_norm(graph, Q)
    nV = get_rms_norm(graph, V)
    A = graph.matmul(nQ, K)
    E = graph.exp(input=A)
    #S = graph.reduce_sum(input=E, axes=(2,))
    S = E
    D = graph.div(x=E, y=S)
    O = graph.matmul(D, nV)
    
    taso.optimize(graph, budget=1)
    
