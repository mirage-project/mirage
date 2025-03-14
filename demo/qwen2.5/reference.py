from transformers import AutoModelForCausalLM, AutoTokenizer
from models.modeling_qwen2 import Qwen2ForCausalLM
import torch
import flashinfer

model_name = "Qwen/Qwen2.5-7B-Instruct"
torch.set_default_dtype(torch.bfloat16)
torch.cuda.set_device(0)
with torch.device("cuda"):
    model = Qwen2ForCausalLM.from_pretrained(model_name)
    model.fuse_weights()
    #model.superoptimize_kernels()
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
tokens = torch.full((1, 1024), 0, dtype=torch.long, device="cuda")
for i in range(model_inputs.input_ids.shape[-1]):
    tokens[0, i] = model_inputs.input_ids[0, i]
prompt_len = model_inputs.input_ids.shape[-1]
positions = torch.arange(1024).unsqueeze(0).to(model.device)
position_embeddings = model.model.rotary_emb(positions)
prev_pos = 0

workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD", use_tensor_cores=True)
kv_page_indices = torch.arange(1).int().to("cuda")
kv_page_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
kv_last_page_len = torch.tensor([prompt_len], dtype=torch.int32, device="cuda")
decode_wrapper.plan(
    kv_page_indptr,
    kv_page_indices,
    kv_last_page_len,
    28, # num_qo_heads,
    4, # num_kv_heads,
    128, # head_dimension
    1024, # page_size
    pos_encoding_mode="NONE",
    q_data_type=torch.bfloat16,
    kv_data_type=torch.bfloat16)

torch.cuda.synchronize()
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
starter.record()

g = torch.cuda.CUDAGraph()
for cur_pos in range(prompt_len, prompt_len + 512):
    kv_last_page_len[0]=cur_pos
    print("kv_last_page_len", kv_last_page_len)
    if cur_pos < prompt_len + 1024:
        input_ids = tokens[:,prev_pos:cur_pos]
        cos_embeddings = position_embeddings[0][:,prev_pos:cur_pos]
        sin_embeddings = position_embeddings[1][:,prev_pos:cur_pos]
        logits = model.forward(
                    input_ids=input_ids,
                    position_embeddings=(cos_embeddings, sin_embeddings),
                    decode_wrapper=decode_wrapper,
                    use_cache = True)
    elif cur_pos == prompt_len + 1024:
        input_ids = tokens[:,prev_pos:cur_pos]
        cos_embeddings = position_embeddings[0][:,prev_pos:cur_pos]
        sin_embeddings = position_embeddings[1][:,prev_pos:cur_pos]
        assert prev_pos + 1 == cur_pos
        with torch.cuda.graph(g):
            logits = model.forward(
                        input_ids=input_ids,
                        position_embeddings=(cos_embeddings, sin_embeddings),
                        use_cache = True)
    else:
        input_ids.copy_(tokens[:,prev_pos:cur_pos])
        cos_embeddings.copy_(position_embeddings[0][:,prev_pos:cur_pos])
        sin_embeddings.copy_(position_embeddings[1][:,prev_pos:cur_pos])
        g.replay()
    next_token = logits.argmax(dim=-1)
    next_token = next_token[0, -1]
    tokens[0, cur_pos] = next_token
    prev_pos = cur_pos
    #if (next_token == model.config.eos_token_id):
    #    break

ender.record()
torch.cuda.synchronize()
run_time = starter.elapsed_time(ender)

generated_ids=tokens[:, :prev_pos]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

print("Prompt length {}, Generate length {}, Run time {} ms".format(prompt_len, prev_pos-prompt_len, run_time))
