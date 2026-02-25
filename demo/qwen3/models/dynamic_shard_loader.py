"""
Library for sharding and loading the model weights directly from HuggingFace.
"""

from safetensors import safe_open
from huggingface_hub import hf_hub_download
import json
import math
from enum import Enum
import torch

class ShardType(Enum):
    COL_PARALLEL = 0
    ROW_PARALLEL = 1
    EXPERT_PARALLEL = 2
    NONE = 100 # No sharding, replicate on all GPUs

class DynamicShardLoader:
    def __init__(self, model, model_name, mapping, rank, world_size, device, download=False):
        self.model = model
        self.model_name = model_name
        self.rank = rank
        self.world_size = world_size
        self.download = download
        self.device = device

        # Reconstruct mapping dict, validate & update parallelism configs.
        self.mapping_dict = self._construct_mapping_dict(mapping)

        if world_size > 1:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD

        # Initialize dict mapping each weight to the file it is in.
        index_path = self._get_model_index_file()
        with open(index_path, "r") as f: 
            index = json.load(f) 
            self.weight_map = index["weight_map"] # key: param name for the weights, val: filename it's in
        self.shard_and_load()
        self.materialize_leftover_buffers()

    # TODO (emily): check if all rank should start from same safetensors file.
    def shard_and_load(self):
        files_mapping = self._download_all_safetensor_files()

        for hf_filename in files_mapping:
            local_filepath = files_mapping[hf_filename]

            # Load onto CPU for now since materialization of tensor happens after getting the slice.
            with safe_open(local_filepath, framework="pt", device="cpu") as f:
                for name in f.keys():
                    name_parts = name.split(".")
                    assert name.startswith("model.") or name == "lm_head.weight"
                    key = name_parts[-2] # ex: q_proj

                    # TODO: either throw error (currently) or replicate on all GPUs if user didn't provide key
                    assert key in self.mapping_dict, f"Param Key {key} not found in mapping"
                    
                    parallelism_info = self.mapping_dict[key]
                    
                    # Check expert parallelism.
                    if "expert" in name and ShardType.EXPERT_PARALLEL in parallelism_info:
                        if not self._check_expert_parallel(name, parallelism_info[ShardType.EXPERT_PARALLEL]):
                            continue

                    # Get model module containing this weight. 
                    module_path = ".".join(name_parts[:-1]) # ex: model.layers.0.mlp.experts.0.gate_proj
                    module = self.model.get_submodule(module_path)

                    meta_tensor = getattr(module, "weight")

                    # Allocate tensor on GPU & move data to it based on TP specifications.
                    weight_slice = f.get_slice(name)
                    if ShardType.COL_PARALLEL in parallelism_info:
                        tp_size = parallelism_info[ShardType.COL_PARALLEL]
                        sharded_tensor, meta_shard = self.handle_tensor_parallelism(ShardType.COL_PARALLEL, tp_size, weight_slice, name, meta_tensor)
                    elif ShardType.ROW_PARALLEL in parallelism_info:
                        tp_size = parallelism_info[ShardType.ROW_PARALLEL]
                        sharded_tensor, meta_shard = self.handle_tensor_parallelism(ShardType.ROW_PARALLEL, tp_size, weight_slice, name, meta_tensor)
                    else: # No TP.
                        sharded_tensor = weight_slice[:]
                        meta_shard = meta_tensor

                    # Attach tensor to model through materialization.
                    self.materialize_and_attach_to_model(meta_shard, sharded_tensor, name, module)


    def _get_parallelism_info(self, param_key, mapping):
        """Given param key (ex: q_proj, gate), validate config and return parallelism info as a dictionary. 
        """
        mapping_info = mapping[param_key]
        parallelism_dict = {} # key: ShardType. val: num groups to parallelize by.
        for info in mapping_info["shard_type"]:
            # Ensure the info is a list of tuples.
            if not isinstance(info, tuple):
                info = (info,)
            
            parallelism_dict[info[0]] = min(info[1], self.world_size) if len(info) > 1 else self.world_size

        # Validate config. By default, this will priortize EP before TP. 
        # This will only throw error if world size is not divisible by TP and EP. Otherwise, it'll update the config to the optimal config.
        # Optimal config will ensure that ep * tp == world_size
        if ShardType.EXPERT_PARALLEL in parallelism_dict:
            ep_size = parallelism_dict[ShardType.EXPERT_PARALLEL]
            assert ep_size <= self.world_size and self.world_size % ep_size == 0, f"world size not divisible by {param_key}'s EP configuration"

            if parallelism_dict[ShardType.EXPERT_PARALLEL] > self.world_size:
                parallelism_dict[ShardType.EXPERT_PARALLEL] = self.world_size
                ep_size = self.world_size
        else:
            ep_size = 1

        if ShardType.ROW_PARALLEL in parallelism_dict:
            tp_size = parallelism_dict[ShardType.ROW_PARALLEL]
            assert self.world_size % tp_size == 0, f"world size {self.world_size} not divisible by {param_key}'s TP configuration of {tp_size}"

            if parallelism_dict[ShardType.ROW_PARALLEL] * ep_size != self.world_size:
                parallelism_dict[ShardType.ROW_PARALLEL] = self.world_size // ep_size

        if ShardType.COL_PARALLEL in parallelism_dict:
            tp_size = parallelism_dict[ShardType.COL_PARALLEL]
            assert self.world_size % tp_size == 0, f"world size {self.world_size} not divisible by {param_key}'s TP configuration of {tp_size}"

            if parallelism_dict[ShardType.COL_PARALLEL] * ep_size != self.world_size:
                parallelism_dict[ShardType.COL_PARALLEL] = self.world_size // ep_size


        return parallelism_dict

    def _construct_mapping_dict(self, mapping):
        """Reconstruct the mapping to be more accessible."""
        updated_mapping_dict = {} # key: param key. val: dict containing ShardType and parallelism size.

        for key in mapping:
            updated_mapping_dict[key] = self._get_parallelism_info(key, mapping)

        return updated_mapping_dict

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
        num_experts = self.model.config.num_experts
        
        experts_per_rank = math.ceil(num_experts / expert_parallel_size)
        ep_rank = self.rank // expert_parallel_size

        expert_start = ep_rank * experts_per_rank
        expert_end = min(expert_start + experts_per_rank, num_experts)
        return weight_num in range(expert_start, expert_end)


    def handle_tensor_parallelism(self, tp_type, tp_size, weight_slice, weight_name, meta_tensor):
        """Perform sharding and create blueprint for the meta tensor.

        Args:
            - meta_tensor (PyTorch.Tensor): the meta tensor initialized in the model for this weight.
        
        Returns:
            A tuple containing:
                - PyTorch.Tensor of the sharded tensor
                - A meta tensor with the correct shape (as blueprint for the actual tensor)
        """
        dim = tp_type.value 
        
        # Valid shape with tensor parallel size.
        shape = weight_slice.get_shape()
        assert (
            shape[dim] % tp_size == 0
        ), f"Error in handle_tensor_parallelism for '{weight_name}': Dimension {dim} must be divisible by {mp}. Tensor shape is {shape}"


        # Perform sharding and return Pytorch tensor.
        shard_size = shape[dim] // tp_size
        tp_rank = self.rank % tp_size
        start = tp_rank * shard_size
        end = (tp_rank + 1) * shard_size

        # Get a meta tensor that is of the right shape.
        if meta_tensor.size(dim) == shard_size:
            meta_shard = meta_tensor
        else:
            meta_shard = meta_tensor.narrow(dim, start, shard_size)

        if tp_type == ShardType.COL_PARALLEL:
            return weight_slice[start:end, :], meta_shard
        else:
            return weight_slice[:, start:end], meta_shard


    # TODO need to also handle cases where tp_type is None (where tp_size will then also be None).
    def materialize_and_attach_to_model(self, meta_tensor, sharded_tensor, weight_name, module):
        """Materialize the tensor in the model to point to the sharded tensor on device.
        
        Args:
            - meta_tensor (PyTorch.Tensor): meta tensor to base the actual device tensor on (has the right shape / strides).
            - sharded_tensor (PyTorch.Tensor): sharded tensor residing on the CPU (after calling get_slice()).
            - weight_name (str): full weight name (ex: model.layers.0.mlp.experts.0.gate_proj.weights).
        """
        # Allocate memory for the actual tensor on the current device & copy tensor data.
        
        tensor = self.materialize_meta_tensor(meta_tensor, self.device)
        with torch.no_grad():
            tensor.copy_(sharded_tensor)

        # Replace model's meta tensor with actual device tensor.
        new_model_param = torch.nn.Parameter(tensor, requires_grad=meta_tensor.requires_grad)
        setattr(module, "weight", new_model_param)
        

    def materialize_meta_tensor(self, meta_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """ Adopted from vLLM's implementation.

        Materialize a meta tensor into an actual tensor on the given device.
        """
        tensor = torch.empty_strided(
            size=tuple(meta_tensor.size()),
            stride=tuple(meta_tensor.stride()),
            dtype=meta_tensor.dtype,
            requires_grad=False,
            device=device
        )
        tensor.__class__ = meta_tensor.__class__
        tensor.__dict__ = meta_tensor.__dict__.copy()
        return tensor


    def materialize_leftover_buffers(self):
        """
        Finds any buffers still on the 'meta' device and moves them to 
        the actual device, initializing them if they are empty.
        """
        count = 0
        for name, buffer in self.model.named_buffers():
            if buffer.is_meta:
                # Allocate memory on device for buffer.
                real_buffer = torch.empty_like(buffer, device=self.device)
                
                # Replace meta buffer with actual tensor in the model.
                parent_name, buf_short_name = name.rsplit('.', 1) if '.' in name else ('', name)
                parent_module = self.model.get_submodule(parent_name) if parent_name else model
                parent_module.register_buffer(buf_short_name, real_buffer, persistent=True)
                
                count += 1
                
        return count



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

    