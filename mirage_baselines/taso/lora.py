import taso

if __name__ == "__main__":
    graph = taso.new_graph()
    X = graph.new_input(dims=(16, 256))
    W = graph.new_input(dims=(256, 4096))
    A = graph.new_input(dims=(256, 16))
    B = graph.new_input(dims=(16, 4096))
    D = graph.matmul(X, A)
    E = graph.matmul(D, B)
    C = graph.matmul(X, W)
    O = graph.add(C, E)
    
    taso.optimize(graph, budget=1)
    