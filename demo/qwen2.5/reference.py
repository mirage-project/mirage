from transformers import AutoModelForCausalLM, AutoTokenizer
from models.modeling_qwen2 import Qwen2ForCausalLM
import torch

model_name = "Qwen/Qwen2.5-7B-Instruct"
torch.set_default_dtype(torch.bfloat16)
torch.cuda.set_device(0)
with torch.device("cuda"):
    model = Qwen2ForCausalLM.from_pretrained(model_name)
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
print(model_inputs)
tokens = torch.full((1, 1024), -1, dtype=torch.long, device="cuda")
for i in range(model_inputs.input_ids.shape[-1]):
    tokens[0, i] = model_inputs.input_ids[0, i]
prompt_len = model_inputs.input_ids.shape[-1]
positions = torch.arange(1024).unsqueeze(0).to(model.device)
prev_pos = 0
print("input_ids", prompt_len, tokens[:,0:prompt_len])
print("model.config", model.config)

for cur_pos in range(prompt_len, prompt_len + 512):
    logits = model.forward(input_ids = tokens[:,prev_pos:cur_pos], position_ids = positions[:,prev_pos:cur_pos], use_cache = True)
    next_token = logits.argmax(dim=-1)
    next_token = next_token[0, -1]
    tokens[0, cur_pos] = next_token
    prev_pos = cur_pos
    if (next_token == model.config.eos_token_id):
        break

#generated_ids = model.generate(
#    **model_inputs,
#    max_new_tokens=512
#)
#generated_ids = [
#    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#]

generated_ids=tokens[:, :prev_pos]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
