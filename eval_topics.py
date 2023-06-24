from longchat.train.monkey_patch.llama_interpolate_monkey_patch import replace_llama_with_interpolate
replace_llama_with_interpolate()

from longchat.train.monkey_patch.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

import os
import json

import torch
import transformers
import numpy as np

from utils import *

#path = "/data/dacheng/vicuna_ft_32K_scale/"
path = "/data/dacheng/vicuna-7b/"
name = path.split("/")[-1]

output_dir = "evaluation/topics/predictions"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_dir = os.path.join(output_dir, name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

model, tokenizer = load_model(path, num_gpus=1)

for num_topics in [10]:
    print(f"Start testing {num_topics} per prompt!")
    test_file = f"evaluation/topics/testcases/{num_topics}_topics.jsonl"

    output_file = os.path.join(output_dir, f"{num_topics}_response.txt")
    
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)

    for test_case in json_list:        
        test_case = json.loads(test_case)
        prompt = test_case["prompt"]
        prompt_length = test_case["prompt_length"]
        topics = test_case["topics"]
        input = tokenizer(prompt, return_tensors="pt")
        #outputs = model.generate(input.input_ids.cuda(), max_new_tokens=100, use_cache=True)[0]
        outputs = model.generate(input.input_ids.cuda(), max_new_tokens=100, use_cache=False)[0]
        outputs = outputs[prompt_length:]
        summary = f"Label: {topics[0]}, Predict: {tokenizer.batch_decode([outputs], skip_special_tokens=True)}, --- INFO --- Topics: {topics}, Length: {prompt_length}"
        
        print(summary)
        with open(output_file, "a+") as f: 
            f.write(summary)
            f.write("\n")
