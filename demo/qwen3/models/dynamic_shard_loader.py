"""
Library for sharding and loading the model weights directly from HuggingFace.
"""

from huggingface_hub import hf_hub_download
import json

# TODO (emily): see if it's faster to broadcast index path or the actual data
class DynamicShardLoader:
    def __init__(self, model_obj, model_name, mapping, rank, world_size, download=False):
        self.model_obj = model_obj
        self.model_name = model_name
        self.mapping = mapping
        self.rank = rank
        self.world_size = world_size
        self.download = download
      
        # Initialize dict mapping each weight to the file it is in.
        index_path = self._get_model_index_file()
        with open(index_path, "r") as f: 
            index = json.load(f) 
            self.weight_map = index["weight_map"] # key: param name for the weights, val: filename it's in

    def _get_model_index_file(self):
        # Fetch model index file from HuggingFace & broadcast path to all ranks.
        if self.rank == 0:
            index_path = hf_hub_download(repo_id=self.model_name, filename="model.safetensors.index.json")
        else:
            index_path = None

        if self.world_size > 1:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            index_path = comm.bcast(index_path, root=0)

        return index_path