import os
import json

import torch
import transformers
import numpy as np

from utils import *

#path = "longchat_7b_2048"
#path = "longchat_7b_4096"
#path = "longchat_7b_8192"
# path = "llama-7B-hf"
# path = "/l/users/dacheng.li/axie/vicuna_7b_8192"
# path = "/l/users/dacheng.li/axie/llama-7B"
path = "/l/users/dacheng.li/axie/longchat_partial_book"
name = path.split("/")[-1]


output_dir = "evaluation/topics/predictions"
name = path.split("/")[-1]

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_dir = os.path.join(output_dir, name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

tokenizer = transformers.AutoTokenizer.from_pretrained(path, use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

model = transformers.AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).cuda()

# for num_topics in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
for num_topics in [6,7,8,9,10]:
    print(f"Start testing {num_topics} per prompt!")
    test_file = f"evaluation/topics/testcases/{num_topics}_topics.jsonl"

    output_file = os.path.join(output_dir, f"{num_topics}_response.txt")
    
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)

    num_correct = 0
    num_invalid = 0
    for test_case in json_list:        
        test_case = json.loads(test_case)
        prompt = test_case["prompt"]
        prompt_length = test_case["prompt_length"]
        topics = test_case["topics"]
        input = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(input.input_ids.cuda(), max_new_tokens=100, use_cache=True)[0]
        outputs = outputs[prompt_length:]
        summary = f"Label: {topics[0]}, Predict: {tokenizer.batch_decode([outputs], skip_special_tokens=True)}, --- INFO --- Topics: {topics}, Length: {prompt_length}"
        

        is_correct = let_gpt_check_response(topics[0], tokenizer.batch_decode([outputs], skip_special_tokens=True), "gpt-3.5-turbo")
        if is_correct is True:
            num_correct = num_correct + 1
            summary = "1 " + summary
        elif is_correct is None:
            num_invalid = num_invalid + 1
            summary = "-1 " + summary
        else:
            summary = "0 " + summary

        print(summary)
        with open(output_file, "a+") as f: 
            f.write(summary)
            f.write("\n")

    with open(output_file, "a+") as f:
        f.write(f"\naccuracy: {num_correct/(len(json_list)-num_invalid)}\n")
