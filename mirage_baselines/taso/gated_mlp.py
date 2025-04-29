import taso

if __name__ == "__main__":
    graph = taso.new_graph()
    X = graph.new_input(dims=(8, 4096))
    W1 = graph.new_input(dims=(4096, 4096))
    W2 = graph.new_input(dims=(4096, 4096))
    O1 = graph.matmul(X, W1)
    O2 = graph.matmul(X, W2)
    O1 = graph.relu(O1)
    O = graph.mul(O1, O2)
    
    taso.optimize(graph, budget=1)
    