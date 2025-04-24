import taso

batch_size = 1

if __name__ == "__main__":
    graph = taso.new_graph()
    Q = graph.new_input(dims=(2 * batch_size, 256, 64))
    K = graph.new_input(dims=(2 * batch_size, 64, 4096))
    V = graph.new_input(dims=(2 * batch_size, 4096, 64))
    A = graph.matmul(input=Q, weight=K)
    E = graph.exp(input=A)
    #S = graph.reduce_sum(input=E, axes=(2,))
    S = E
    D = graph.div(x=E, y=S)
    O = graph.matmul(input=D, weight=V)
    
    taso.optimize(graph, budget=1)
    
