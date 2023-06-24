import os
import json
import numpy as np

import torch
import transformers

num_test_samples = 50
num_topics_per_sample_list = [15, 20, 30]
conversations_list = []
topics_list = []

path = "/data/dacheng/longchat_7b_2048"
topics_dir = "./evaluation/topics"
output_dir = os.path.join(topics_dir, "testcases")

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

tokenizer = transformers.AutoTokenizer.from_pretrained(path, use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

with open(os.path.join(topics_dir, 'conversations.jsonl'), 'r') as json_file:
    json_list = list(json_file)

for json_str in json_list:
    result = json.loads(json_str)
    conversations_list.append(result['conversation'])
    topics_list.append(result['topic'])
    print(f"Topic: {result['topic']}, length: {len(tokenizer(result['conversation']).input_ids)}")

for num_topics_per_sample in num_topics_per_sample_list:
    output_path = open(os.path.join(output_dir, f"{num_topics_per_sample}_topics.jsonl"), "w")

    for i in range(num_test_samples):
        indices = np.random.choice(list(range(len(conversations_list))), size=num_topics_per_sample, replace=True)
        record_prompt = f"Below is a record of our previous conversation on {num_topics_per_sample} different topics. You are the ASSISTANT, and I am the USER. At the beginning of each topic, the USER will say 'I would like to discuss the topic of <TOPIC>'. Memorize each <TOPIC>. At the end of the record, I will ask you to retrieve the first topic name. Now the record start. "
        topics = []
        for index in indices:
            record_prompt += conversations_list[index]
            topics.append(topics_list[index])
        prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: {record_prompt} Now the record ends. What is the first topic we discussed? Only give me the topic name. Do not summarize yourself. ASSISTANT:" 
    
        input = tokenizer(prompt, return_tensors="pt")
        prompt_length = input.input_ids.shape[-1]
        print(f"id {i}: length {prompt_length}")
        cur_output = {"test_id": i, "prompt": prompt, "topics": topics, "prompt_length": prompt_length}
        
        json.dump(cur_output, output_path)
        output_path.write("\n")
