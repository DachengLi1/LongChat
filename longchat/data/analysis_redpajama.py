import json

import torch
from collections import defaultdict

import datasets
import pandas as pd
from transformers import AutoTokenizer
#from datasets import load_dataset

#dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
with open("/home/hao.zhang/dacheng/sharegpt_20230515_clean_lang_gpt4.json", "r") as f:
    dataset = json.load(f)
print(len(dataset))
#print(dataset[0])
dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
tokenizer = AutoTokenizer.from_pretrained("/data/dacheng/llama-7B-hf", use_fast=False)

def tokenization(example):
    conversation = example["conversations"]
    cur = ""
   # print(len(conversation))
    for c in conversation:
       # print(c)
        cur += c["value"]
    #return tokenizer(example["text"])
    return tokenizer(cur)

dataset = dataset.map(tokenization, batched=False)

stages_count = defaultdict(int)
for d in dataset:
    length = len(d["input_ids"])
    if 0 < length and length < 2048:
        stages_count["2048"] += 1
    elif 2048 < length and length < 4096:
        stages_count["4096"] += 1
    elif 4096 < length and length < 8192:
        stages_count["8192"] += 1
    elif 8192 < length and length < 16384:
        stages_count["16384"] += 1
    elif 16384 < length and length < 32768:
        stages_count["32768"] += 1
    else:
        stages_count[">32768"] += 1

print(stages_count)
