from models.modeling_qwen2 import Qwen2ForCausalLM
from transformers import AutoTokenizer
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-mirage", action='store_true', help="Disable Mirage kernels")
    args = parser.parse_args()
    print("Input arguments:", args)

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)
    with torch.device("cuda"):
        model = Qwen2ForCausalLM.from_pretrained(model_name)
        model.fuse_weights()
        if not args.disable_mirage:
            model.superoptimize_kernels()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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

    g = torch.cuda.CUDAGraph()
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
        elif cur_pos == prompt_len + 1:
            input_ids = tokens[:,prev_pos:cur_pos]
            cos_embeddings = position_embeddings[0][:,prev_pos:cur_pos]
            sin_embeddings = position_embeddings[1][:,prev_pos:cur_pos]
            assert prev_pos + 1 == cur_pos
            with torch.cuda.graph(g, stream=stream):
                logits = model.forward(
                            input_ids=input_ids,
                            position_embeddings=(cos_embeddings, sin_embeddings),
                            step=step,
                            stream=stream)
        else:
            input_ids.copy_(tokens[:,prev_pos:cur_pos])
            cos_embeddings.copy_(position_embeddings[0][:,prev_pos:cur_pos])
            sin_embeddings.copy_(position_embeddings[1][:,prev_pos:cur_pos])
            g.replay()
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
