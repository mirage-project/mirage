""" Graph dataset for caching superoptimize results"""

class DatasetEntry:
    def __init__(self, input_graph, optimized_graph, imaps, omaps, griddims, blockdims, fmaps, franges, backend):
        self.input_graph = input_graph
        self.optimized_graph=optimized_graph
        self.imaps=imaps
        self.omaps=omaps
        self.griddims=griddims
        self.blockdims=blockdims
        self.fmaps=fmaps
        self.franges=franges
        self.backend=backend

class GraphDataset:
    def __init__(self):
        self.dataset = dict()

    def find(self, input_graph, imaps, omaps, griddims, blockdims, fmaps, franges, backend):
        hash_value = input_graph.get_owner_independent_hash()
        if hash_value in self.dataset:
            entries = self.dataset[hash_value]
            for e in entries:
                if (e.imaps == imaps) and (e.omaps == omaps) and (e.griddims == griddims) and (e.blockdims == blockdims) and (e.fmaps == fmaps) and (e.franges == franges) and (e.backend == backend):
                    return e.optimized_graph
        return None
    
    def store(self, input_graph, optimized_graph, imaps, omaps, griddims, blockdims, fmaps, franges, backend):
        hash_value = input_graph.get_owner_independent_hash()
        new_entry = DatasetEntry(input_graph, optimized_graph, imaps, omaps, griddims, blockdims, fmaps, franges, backend)
        if hash_value in self.dataset:
            self.dataset[hash_value].append(new_entry)
        else:
            self.dataset[hash_value] = [new_entry]


graph_dataset = GraphDataset()
