from dataclasses import dataclass
from typing import Optional, List

# from .models.modeling_qwen3 import Qwen3ForCausalLM
from .model_registry import get_builder
from .models.graph_builder import MirageModelConfig
from ..utils import get_configurations_from_gpu
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os
import mirage as mi
from . import models

@dataclass
class MPKMetadata:
    # ---------- MPK class external state bundled here ----------
    # args
    mode: str = "offline"
    total_num_requests: int = 1
    num_remote_schedulers: int = 0
    max_seq_length: int = 0
    max_num_batched_requests: int = 0
    max_num_batched_tokens: int = 0
    max_num_pages: int = 0
    page_size: int = 0
    max_sm_num: int = 108
    device: str = "cuda"
    # model 
    weight_from_model: bool = False
    model_name: Optional[str] = None # For now, model_name must be provided
    model_path: Optional[str] = None
    # multi device support
    world_size: int = 1
    rank: int = 0
    # Meta tensors
    step: Optional[torch.Tensor] = None
    tokens: Optional[torch.Tensor] = None
    input_tokens: Optional[torch.Tensor] = None
    output_tokens: Optional[torch.Tensor] = None
    num_new_tokens: Optional[torch.Tensor] = None
    prompt_lengths: Optional[torch.Tensor] = None
    qo_indptr_buffer: Optional[torch.Tensor] = None
    paged_kv_indptr_buffer: Optional[torch.Tensor] = None
    paged_kv_indices_buffer: Optional[torch.Tensor] = None
    paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None
    # MirageModelConfig
    model_config: Optional[MirageModelConfig] = None
    # profiling
    profiler_tensor: Optional[torch.Tensor] = None
    trace_name: Optional[str] = None
    # spec decode config
    spec_decode_config: Optional[object] = None
    
    def check_valid(self):
        if self.weight_from_model:
            assert (self.model_name is not None) or (self.model_path is not None), "model_name or model_path is required when weight_from_model is True"
        else:
            assert self.state_dict is not None, "state_dict is required when weight_from_model is False"
            assert self.k_cache is not None, "k_cache is required when weight_from_model is False"
            assert self.v_cache is not None, "v_cache is required when weight_from_model is False"
            
    def info_as_string(self):
        info = "MPKMetadata info:"
        info += f"Mode: {self.mode}\n"
        info += f"Total number of requests: {self.total_num_requests}\n"
        info += f"Number of remote schedulers: {self.num_remote_schedulers}\n"
        info += f"Max sequence length: {self.max_seq_length}\n"
        info += f"Max number of batched requests: {self.max_num_batched_requests}\n"
        info += f"Max number of batched tokens: {self.max_num_batched_tokens}\n"
        info += f"Max number of pages: {self.max_num_pages}\n"
        info += f"Page size: {self.page_size}\n"
        info += f"Max SM number: {self.max_sm_num}\n"
        info += f"Device: {self.device}\n"
        info += f"Weight from model: {self.weight_from_model}\n"
        info += f"Model name: {self.model_name}\n"
        info += f"Model path: {self.model_path}\n"
        info += f"World size: {self.world_size}\n"
        info += f"Rank: {self.rank}\n"
        info += f"Step: {self.step.shape}\n"
        info += f"Tokens: {self.tokens.shape}\n"
        info += f"Input tokens: {self.input_tokens.shape}\n"
        info += f"Output tokens: {self.output_tokens.shape}\n"
        info += f"Num new tokens: {self.num_new_tokens.shape}\n"
        info += f"Prompt lengths: {self.prompt_lengths.shape}\n"
        info += f"QO indptr buffer: {self.qo_indptr_buffer.shape}\n"
        info += f"Paged KV indptr buffer: {self.paged_kv_indptr_buffer.shape}\n"
        info += f"Paged KV indices buffer: {self.paged_kv_indices_buffer.shape}\n"
        info += f"Paged KV last page len buffer: {self.paged_kv_last_page_len_buffer.shape}\n"
        info += f"Model config: \n"
        info += self.model_config.info_as_string()
        info += f"Profiler tensor: {self.profiler_tensor.shape}\n"
        info += f"Trace name: {self.trace_name}\n"
        return info
        
    def write_to_file(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(self.info_as_string())
            
    def print_info(self):
        print(self.info_as_string())
        
class MPK:

    def __init__(self, meta: MPKMetadata):
        print("Initializing mirage_wrapper...")
        
        self.init_mpi()
        # if self.rank != 0:
        #     print = lambda *_, **__: None
        self.metadata = meta
        args = meta
        self.model_name = args.model_name
        self.device = args.device
        self.total_num_requests = args.total_num_requests
        self.weight_from_model = args.weight_from_model
        self.state_dict = args.state_dict
        
        torch.set_default_dtype(torch.bfloat16)
        torch.cuda.set_device(self.rank)
        
        # if args.qo_indptr_buffer is None:
        #     self.get_tensors()
        # else:
        self.step = args.step
        self.tokens = args.tokens
        self.input_tokens = args.input_tokens
        self.output_tokens = args.output_tokens
        self.num_new_tokens = args.num_new_tokens
        self.prompt_lengths = args.prompt_lengths
        self.qo_indptr_buffer = args.qo_indptr_buffer
        self.paged_kv_indptr_buffer = args.paged_kv_indptr_buffer
        self.paged_kv_indices_buffer = args.paged_kv_indices_buffer
        self.paged_kv_last_page_len_buffer = args.paged_kv_last_page_len_buffer
        
        self.profiler_tensor = args.profiler_tensor
        self.spec_decode_config = args.spec_decode_config
        
        self.get_tensors()
        
        self.model_config = args.model_config
        
            
        self.num_workers, self.num_schedulers = get_configurations_from_gpu(self.rank)
        # self.max_sm_num = args.max_sm_num
        
        self.persisten_kernel = mi.PersistentKernel(
            mode=args.mode,
            world_size=self.world_size,
            mpi_rank=self.rank,
            num_workers=self.num_workers,
            num_local_schedulers=self.num_schedulers,
            num_remote_schedulers=0,
            max_seq_length=args.max_seq_length,
            max_num_batched_requests=args.max_num_batched_requests,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_pages=args.max_num_pages,
            page_size=args.page_size,
            eos_token_id=-1, #self.model.config.eos_token_id,
            meta_tensors={
                "step": self.step,
                "tokens": self.tokens,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "num_new_tokens": self.num_new_tokens,
                "prompt_lengths": self.prompt_lengths,
                "qo_indptr_buffer": self.qo_indptr_buffer,
                "paged_kv_indptr_buffer": self.paged_kv_indptr_buffer,
                "paged_kv_indices_buffer": self.paged_kv_indices_buffer,
                "paged_kv_last_page_len_buffer": self.paged_kv_last_page_len_buffer,
            },
            profiler_tensor=self.profiler_tensor,
            trace_name=args.trace_name,
            spec_decode_config=self.spec_decode_config,
        )
        meta_tensors = [
            self.step,
            self.tokens,
            self.input_tokens,
            self.output_tokens,
            self.num_new_tokens,
            self.prompt_lengths,
            self.qo_indptr_buffer,
            self.paged_kv_indptr_buffer,
            self.paged_kv_indices_buffer,
            self.paged_kv_last_page_len_buffer,
        ]
        self.meta_tensors_ptr = [tensor.data_ptr() for tensor in meta_tensors]
        self.profiler_buffer_ptr = (
            self.persisten_kernel.profiler_tensor.data_ptr() if self.persisten_kernel.profiler_tensor is not None else 0
        )
        
        # print("Defining mpk...")
        # self.define_mpk()
        # print("Defining mpk... done")


        # print("Generating task graph...")
        # results = self.persisten_kernel.kn_graph.generate_task_graph(num_gpus=self.world_size, my_gpu_id=self.rank)
        # print("Generating task graph... done")
        # with open(f"task_graph_{self.rank}.json", "w") as f:
        #     f.write(results["json_file"])
        # with open(f"kernel_{self.rank}.cu", "w") as f:
        #     f.write(results["cuda_code"])

        # print("Compiling mpk...")
        # self.persisten_kernel.compile(output_dir=args.output_dir)
        # print("Compiling mpk... done")
        
        self.is_built = False
        self.task_graph_generated = False
        self.is_compiled = False
        
    def init_mpi(self):
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            self.world_size = comm.Get_size()
            self.rank = comm.Get_rank()
            os.environ["RANK"] = str(self.rank)
            os.environ["WORLD_SIZE"] = str(self.world_size)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
        except ImportError:
            self.world_size = 1
            self.rank = 0
        if self.world_size > 1:
            dist.init_process_group(backend="nccl", init_method="env://")
            
    def compensate_meta_tensors(self):
        # TODO: This is a temporary workaround. Ideally we should only allocate tensors we need.
        if self.step is None:
            self.step = torch.full((self.total_num_requests, ), 0, dtype=torch.int32, device="cuda")
        if self.tokens is None:
            self.tokens = torch.full((self.total_num_requests, self.max_seq_length), 0, dtype=torch.long, device="cuda")
        if self.input_tokens is None:
            self.input_tokens = torch.full((self.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
        if self.output_tokens is None:
            self.output_tokens = torch.full((self.max_num_batched_tokens, 1), 0, dtype=torch.long, device="cuda")
        if self.num_new_tokens is None:
            self.num_new_tokens = torch.full((self.total_num_requests, ), 1, dtype=torch.int32, device="cuda")
        if self.prompt_lengths is None:
            self.prompt_lengths = torch.full((self.total_num_requests,), 0, dtype=torch.int32, device="cuda")
        if self.qo_indptr_buffer is None:
            self.qo_indptr_buffer = torch.empty(
                self.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        if self.paged_kv_indptr_buffer is None:
            self.paged_kv_indptr_buffer = torch.empty(
                self.max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        if self.paged_kv_indices_buffer is None:
            self.paged_kv_indices_buffer = torch.empty(
                self.max_num_pages, dtype=torch.int32, device="cuda")
        if self.paged_kv_last_page_len_buffer is None:
            self.paged_kv_last_page_len_buffer = torch.empty(
                self.max_num_batched_requests, dtype=torch.int32, device="cuda")
 
    def get_tensors(self):
        """
        Allocate tensors for the MPK. This is used when we manage tensors by ourselves.
        """
        args = self.metadata
        self.total_num_requests = args.max_num_batched_requests
                
        positions = torch.arange(32768).unsqueeze(0).to(self.model.device)
        # TODO: What if using vllm?
        self.position_embeddings = self.model.model.rotary_emb(positions)
        
        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        #     enable_timing=True
        # )
        
        if args.profiling:
            self.profiler_tensor = torch.zeros(
                3000 * 128, dtype=torch.uint64, device="cuda"
            ).contiguous()
        else:
            self.profiler_tensor = None
            
        self.spec_decode_config = mi.speculative.spec_decode_class(
            args.spec_decode,
            ngram_size=args.ngram_size,
            spec_length=args.spec_length,
        )
        
        self.compensate_meta_tensors()
        
    def load_new_request(self, prompt):
        if not self.is_built:
            raise ValueError("Model is not built yet, so tokenizer is not available")
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        L = model_inputs.input_ids.shape[-1]
        self.tokens.zero_()
        src = model_inputs.input_ids[0]
        self.tokens[:, :L].copy_(src.unsqueeze(0).expand(self.total_num_requests, -1))
                 
        # Clear tensors
        self.input_tokens.fill_(0)
        self.output_tokens.fill_(0)
        self.step.fill_(0)
        self.num_new_tokens.fill_(1)
        self.prompt_lengths.fill_(model_inputs.input_ids.shape[-1])
        # print(f"prompt_lengths filled with {model_inputs.input_ids.shape[-1]}")
        
        self.qo_indptr_buffer.fill_(0)
        self.paged_kv_indptr_buffer.fill_(0)
        self.paged_kv_indices_buffer.fill_(0)
        self.paged_kv_last_page_len_buffer.fill_(0)
                
    def init_per_request(self):
        #meta_tensors_ptr = [tensor.data_ptr() for tensor in self.meta_tensors]
        self.persisten_kernel.init_func(
            self.meta_tensors_ptr,
            self.profiler_buffer_ptr,
            self.persisten_kernel.mpi_rank,
            self.persisten_kernel.num_workers,
            self.persisten_kernel.num_local_schedulers,
            self.persisten_kernel.num_remote_schedulers,
            self.persisten_kernel.max_seq_length,
            self.persisten_kernel.total_num_requests,
            self.persisten_kernel.eos_token_id,
        )
        
    def run(self, prompt):
        self.load_new_request(prompt)
        if not self.first_run:
            self.init_per_request()
        else:
            self.first_run = False
        self.persisten_kernel()
        torch.cuda.synchronize()
        all_responses = []
        for r in range(self.total_num_requests):
            generated_ids = self.tokens[r, : self.step[r] + 1]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_responses.append(response)
        return all_responses
    
    def grid_dim_sanitizer(self, grid_dim):
        #TODO: Do better in sm num limit
        dims = list(grid_dim)
        total_sm_num = dims[0] * dims[1] * dims[2]
        while total_sm_num > self.max_sm_num:
            for idx in (2, 1, 0):
                if dims[idx] > 1:
                    dims[idx] //= 2
                    break
            else:
                # All dimensions are 1; cannot reduce further
                break
            total_sm_num = dims[0] * dims[1] * dims[2]
        return tuple(dims)
        
    def build(self):
        model_builder_class = get_builder(self.model_name)
        self.model_builder = model_builder_class(self.persisten_kernel)
        if self.weight_from_model:
            self.model_builder.build_from_model()
        else:
            self.model_builder.build_from_dict(self.state_dict)
        self.tokenizer = self.model_builder.tokenizer
        
        self.is_built = True
        
    def generate_task_graph(self):
        print("Generating task graph...")
        if not self.is_built:
            raise ValueError("Model is not built yet")
        results = self.persisten_kernel.kn_graph.generate_task_graph(num_gpus=self.world_size, my_gpu_id=self.rank)
        print("Generating task graph... done")
        self.task_graph_generated = True
        return results
    
    def compile(self, output_dir: str = None):
        print("Compiling mpk...")
        if not self.task_graph_generated:
            self.generate_task_graph()

        self.persisten_kernel.compile(output_dir=output_dir)
        print("Compiling mpk... done")
        self.is_compiled = True
        
    def __call__(self, **kwargs):
        if not self.is_compiled:
            self.compile()
        self.persisten_kernel(**kwargs)
        
    def decode(self, ids: torch.Tensor):
        return self.model_builder.decode(ids)
    
    def __del__(self):
        if self.world_size > 1:
            dist.destroy_process_group()