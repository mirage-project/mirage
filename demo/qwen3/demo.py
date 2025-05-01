from models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os
from mpi4py import MPI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    #world_size = int(os.getenv("WORLD_SIZE", "1"))
    #rank = int(os.getenv("RANK", "0"))
    #local_rank = int(os.getenv("LOCAL_RANK", "0"))
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    #model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_name = "Qwen/Qwen3-8B"
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(rank)
    with torch.device("cuda"):
        #model = Qwen3ForCausalLM.from_pretrained(model_name).to("cuda")
        config = AutoConfig.from_pretrained("/opt/dlami/nvme/models/Qwen3-8B/")
        model = Qwen3ForCausalLM(config, world_size)
    load_model(model, f"/opt/dlami/nvme/models/Qwen3-8B/model{rank}-mp{world_size}.safetensors")

    # get all model weight tensors
    weight_tensors = []
    weight_tensors.append(("model.embed_tokens.weight", model.model.embed_tokens.weight))
    for i, layer in enumerate(model.model.layers):
        weight_tensors.append((f"model.layers.{i}.input_layernorm.weight", layer.input_layernorm.weight))
        weight_tensors.append((f"model.layers.{i}.self_attn.q_proj.weight", layer.self_attn.q_proj.weight))
        weight_tensors.append((f"model.layers.{i}.self_attn.k_proj.weight", layer.self_attn.k_proj.weight))
        weight_tensors.append((f"model.layers.{i}.self_attn.v_proj.weight", layer.self_attn.v_proj.weight))
        weight_tensors.append((f"model.layers.{i}.self_attn.q_norm.weight", layer.self_attn.q_norm.weight))
        weight_tensors.append((f"model.layers.{i}.self_attn.k_norm.weight", layer.self_attn.k_norm.weight))
        weight_tensors.append((f"model.layers.{i}.self_attn.o_proj.weight", layer.self_attn.o_proj.weight))
        weight_tensors.append((f"model.layers.{i}.post_attention_layernorm.weight", layer.post_attention_layernorm.weight))
        weight_tensors.append((f"model.layers.{i}.mlp.up_proj.weight", layer.mlp.up_proj.weight))
        weight_tensors.append((f"model.layers.{i}.mlp.gate_proj.weight", layer.mlp.gate_proj.weight))
        weight_tensors.append((f"model.layers.{i}.mlp.down_proj.weight", layer.mlp.down_proj.weight))
    weight_tensors.append(("model.norm.weight", model.model.norm.weight))
    weight_tensors.append(("lm_head.weight", model.lm_head.weight))

    print(weight_tensors)

    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained("/opt/dlami/nvme/models/Qwen3-8B/")
    
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    tokens = torch.full((1, 32768), 0, dtype=torch.long, device="cuda")
    for i in range(model_inputs.input_ids.shape[-1]):
        tokens[0, i] = model_inputs.input_ids[0, i]
    prompt_len = model_inputs.input_ids.shape[-1]
    positions = torch.arange(32768).unsqueeze(0).to(model.device)
    position_embeddings = model.model.rotary_emb(positions)
    prev_pos = 0
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    #g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    step = torch.tensor([0], dtype=torch.int32, device="cuda")
    warmup = 16
    output_len = 512
    for cur_pos in range(prompt_len, prompt_len + output_len):
        step.fill_(cur_pos-1)
        # prefilling phase
        if cur_pos < prompt_len + 1:
            input_ids = tokens[:,prev_pos:cur_pos]
            cos_embeddings = position_embeddings[0][:,prev_pos:cur_pos]
            sin_embeddings = position_embeddings[1][:,prev_pos:cur_pos]
            logits = model.forward(
                        input_ids=input_ids,
                        position_embeddings=(cos_embeddings, sin_embeddings),
                        step=step,
                        stream=stream)
        # decoding phase
        else:
            input_ids = tokens[:,prev_pos:cur_pos]
            cos_embeddings = position_embeddings[0][:,prev_pos:cur_pos]
            sin_embeddings = position_embeddings[1][:,prev_pos:cur_pos]
            logits = model.forward(
                       input_ids=input_ids,
                        position_embeddings=(cos_embeddings, sin_embeddings),
                        step=step,
                        stream=stream)
        next_token = logits.argmax(dim=-1)
        next_token = next_token[0, -1]
        tokens[0, cur_pos] = next_token
        prev_pos = cur_pos
        if (next_token == model.config.eos_token_id):
            break
        if cur_pos == prompt_len + warmup:
            torch.cuda.synchronize()
            starter.record()
    
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)
    
    generated_ids=tokens[:, :prev_pos]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    
    print("Prompt length {}, generate length {}, per-token latency {} ms".format(prompt_len, cur_pos + 1, run_time / (cur_pos + 1 - warmup)))
    if world_size > 1:
        dist.destroy_process_group()
