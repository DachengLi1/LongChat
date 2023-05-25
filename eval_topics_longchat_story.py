import os
import json

import torch
import transformers
import numpy as np

#path = "longchat_7b_2048"
#path = "longchat_7b_4096"
#path = "longchat_7b_8192"
#path = "llama-7B-hf"
#path = "vicuna_7b_flash_seq_8192_new/checkpoint-3500"
#path = "longchat_13b_4096"
#path = "mpt-7b-storywriter"
#path = "mpt-7b-storywriter"
path = "mpt-7b-story_2048/checkpoint-13000"
output_dir = "evaluation/topics/predictions"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_dir = os.path.join(output_dir, path)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

#config = transformers.AutoConfig.from_pretrained(
#  'mosaicml/mpt-7b-storywriter',
#  trust_remote_code=True
#)
#config.update({"max_seq_len": 83968})
#config.attn_config['attn_impl'] = 'triton'
config = transformers.MPTConfig.from_pretrained(
  path
)
config.attn_config['attn_impl'] = 'torch'
print(config.attn_config)
model = transformers.MPTForCausalLM.from_pretrained(
  path,
  config=config,
#  torch_dtype=torch.float16
  torch_dtype=torch.bfloat16,
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b",
        model_max_length=4096,
        padding_side="right",)
tokenizer.pad_token = tokenizer.unk_token
model.to(device='cuda:1')
model = model.to(torch.bfloat16)
from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

for num_topics in [5, 10, 15, 20]:
    print(f"Start testing {num_topics} per prompt!")
    test_file = f"evaluation/topics/testcases_longchat_story/{num_topics}_topics.jsonl"

    output_file = os.path.join(output_dir, f"{num_topics}_response.txt")
    
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)

    for test_case in json_list:
        test_case = json.loads(test_case)
        prompt = test_case["prompt"]
        prompt_length = test_case["prompt_length"]
        topics = test_case["topics"]
        input = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(input.input_ids.to(device="cuda:1"), max_new_tokens=100, use_cache=True)[0]
        outputs = outputs[prompt_length:]
        summary = f"Label: {topics[0]}, Predict: {tokenizer.batch_decode([outputs], skip_special_tokens=True)}, --- INFO --- Topics: {topics}, Length: {prompt_length}"
        print(summary)
        with open(output_file, "a+") as f:
            f.write(summary)
            f.write("\n")
