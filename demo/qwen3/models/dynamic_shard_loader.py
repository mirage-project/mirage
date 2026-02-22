"""
Library for sharding and loading the model weights directly from HuggingFace.
"""

from safetensors import safe_open
from huggingface_hub import hf_hub_download
import json
import math
from enum import Enum

class ShardType(Enum):
    COL_PARALLEL = 0
    ROW_PARALLEL = 1
    EXPERT_PARALLEL = 2
    NONE = 100 # No sharding, replicate on all GPUs

class DynamicShardLoader:
    def __init__(self, model_obj, model_name, mapping, rank, world_size, download=False):
        self.model_obj = model_obj
        self.model_name = model_name
        self.mapping = mapping
        self.rank = rank
        self.world_size = world_size
        self.download = download

        if world_size > 1:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
      
        # Initialize dict mapping each weight to the file it is in.
        index_path = self._get_model_index_file()
        with open(index_path, "r") as f: 
            index = json.load(f) 
            self.weight_map = index["weight_map"] # key: param name for the weights, val: filename it's in
        self.shard_and_load()

    # TODO (emily): check if all rank should start from same safetensors file.
    def shard_and_load(self):
        files_mapping = self._download_all_safetensor_files()

        for hf_filename in files_mapping:
            local_filepath = files_mapping[hf_filename]

            with safe_open(local_filepath, framework="pt") as f:
                for name in f.keys():
                    assert name.startswith("model.") or name == "lm_head.weight"
                    key = name.split(".")[-2]

                    # TODO: either throw error (currently) or replicate on all GPUs if user didn't provide key
                    assert key in self.mapping, f"Param Key {key} not found in mapping"
                    
                    parallelism_info = self._get_parallelism_info(key)
                    
                    # Check expert parallelism.
                    if "expert" in name and ShardType.EXPERT_PARALLEL in parallelism_info:
                        if not self._check_expert_parallel(name, parallelism_info[ShardType.EXPERT_PARALLEL]):
                            continue

                    # Allocate tensor on GPU & move data to it based on TP specifications.
                    if ShardType.COL_PARALLEL in parallelism_info:
                        pass
                    elif ShardType.ROW_PARALLEL in parallelism_info:
                        pass
                    else: # No TP.
                        pass


    def _get_parallelism_info(self, param_key):
        """Given param key (ex: q_proj, gate), return parallelism info as a dictionary. 
        """
        mapping_info = self.mapping[param_key]
        parallelism_dict = {} # key: ShardType. val: num groups to parallelize by, or None for default.
        for info in mapping_info["shard_type"]:
            # Ensure the info is a list of tuples.
            if not isinstance(info, tuple):
                info = (info,)
            
            parallelism_dict[info[0]] = info[1] if len(info) > 1 else None

        return parallelism_dict

    def _check_expert_parallel(self, full_weight_name, expert_parallel_size=None):
        """Return true if the weight should be included, false otherwise, based on EP configs.

        Args:
            - full_weight_name: Full name of the weights from the safetensor files.
                           ex: model.layers.18.mlp.experts.94.gate_proj.weight
            - expert_parallel_size: Number of groups to divide the experts into.

        Returns:
            - bool for whether or not this weight should be included.
        """
        # Only check if it's an expert layer.
        if "expert" not in full_weight_name:
            return True

        if expert_parallel_size == None:
            expert_parallel_size = self.world_size

        weight_name_components = full_weight_name.split('.')
        weight_num = int(weight_name_components[5])
        num_experts = self.model_obj.config.num_experts
        
        experts_per_rank = math.ceil(num_experts / self.world_size)

        expert_start = self.rank * experts_per_rank
        expert_end = min(expert_start + experts_per_rank, num_experts)
        return weight_num in range(expert_start, expert_end)


    def _get_model_index_file(self):
        # Fetch model index file from HuggingFace & broadcast path to all ranks.
        if self.rank == 0:
            index_path = hf_hub_download(repo_id=self.model_name, filename="model.safetensors.index.json")
        else:
            index_path = None

        if self.world_size > 1:
            index_path = self.comm.bcast(index_path, root=0)

        return index_path


    def _download_all_safetensor_files(self):
        if self.rank == 0:
            files_list = set(self.weight_map.values())
            files_mapping = {} # key: safetensor filename on HF. val: local filepath.

            for filename in files_list:
                path = hf_hub_download(repo_id=self.model_name, filename=filename)
                print("Downloaded", filename)
                files_mapping[filename] = path
        
        else:
            files_mapping = {}

        if self.world_size > 1:
            files_mapping = self.comm.bcast(files_mapping, root=0)
        
        self.files_mapping = files_mapping
        return files_mapping