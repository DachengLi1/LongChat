import json
import torch
import transformers
import numpy as np

path = "vicuna_7b_flash_seq_4096_to_8192"
tokenizer = transformers.AutoTokenizer.from_pretrained(path, use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

with open('./evaluation/conversations.jsonl', 'r') as json_file:
    json_list = list(json_file)

conversations_list = []
topics_list = []
for json_str in json_list:
    result = json.loads(json_str)
    conversations_list.append(result['conversation'])
    topics_list.append(result['topic'])
    print(f"label: {result['topic']}, length: {len(tokenizer(result['conversation']).input_ids)}")

model = transformers.AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).cuda()

num_test_samples = 10
per_sample_conversations = 5

for i in range(num_test_samples):
    indices = np.random.choice(list(range(len(conversations_list))), size=per_sample_conversations, replace=False)
    question_string = f"Below is a record of a long conversation between a user and another assistant on {per_sample_conversations} different topics. Pay attention to the topics they discuss. At the beginning of each topic, the USER will say 'I would like to discuss the topic of <TOPIC>'. You only need to memorize the word of this <TOPIC>. At the end of the record, I will ask you to retrieve the first topic they discuss. Only give me the topic they disucss, you don't need to summarize their conversation. Now the record start. "
    topic = []
    for index in indices:
        question_string += conversations_list[index]
        topic.append(topics_list[index])
    prompt = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: {question_string} Now the record ends. What is the first topic they talk about? Remember only give me the topic name, you don't need to summarize. ASSISTANT:" 
    
    input = tokenizer(prompt, return_tensors="pt")
    len_input = input.input_ids.shape[-1]
    print(f"This input is of length {len_input}")
    
    print(f"The input is: {prompt}")
    outputs = model.generate(input.input_ids.cuda(), max_new_tokens=100, use_cache=True)[0]
    
    outputs = outputs[len_input:]
    print(f"The ground truth is '{topic}', the model predicts: {tokenizer.batch_decode([outputs], skip_special_tokens=True)}")
