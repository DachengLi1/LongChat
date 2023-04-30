# Need to call this before importing transformers.
#from longchat.train.llama_performer_monkey_patch import (
#    replace_llama_attn_with_performer,
#)
import json
#from longchat.train.llama_flash_attn_monkey_patch import (
#    replace_llama_attn_with_flash_attn,
#)

#replace_llama_attn_with_flash_attn()

import transformers
import torch

#path = "vicuna_7b_performer_128_continue/checkpoint-7000/"
#path = "vicuna_7b_flash_seq_8192/"
path = "/home/hao.zhang/dacheng/vicuna-7b"
json_path = "longtest/question2.json"
json_file = json.load(open(json_path, "r"))
question_string = json_file["Question"]

model = transformers.AutoModelForCausalLM.from_pretrained(path).cuda()
tokenizer = transformers.AutoTokenizer.from_pretrained(path, use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: {question_string} ASSISTANT:" 
input = tokenizer(prompt, return_tensors="pt")
print(f"This input is of length {input.input_ids.shape}")
outputs = model.generate(input.input_ids.cuda(), max_new_tokens=100, use_cache=True)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
