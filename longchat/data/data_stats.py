"""
Split data into multiple stages for stage-wise sequence length training
"""
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from typing import Dict, Sequence, Optional
from collections import defaultdict

import transformers
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from longchat import conversation as conversation_lib
from functools import partial

def make_sample(sample, start_idx, end_idx):
    assert (end_idx - start_idx) % 2 == 0
    return {
        "id": sample["id"] + "_" + str(start_idx),
        "conversations": sample["conversations"][start_idx:end_idx],
    }

def filter_one_sample(sample, tokenizer, min_length, max_length):
    length = len(tokenizer(sample["text"]).input_ids)

    if 0 < length and length < 2048:
        return 2048
    elif 2048 < length and length < 4096:
        return 4096
    elif 4096 < length and length < 8192:
        return 8192
    elif 8192 < length and length < 16384:
        return 16384
    elif 16384 < length and length < 32768:
        return 32768
    else:
        return -1

def filter_all(content, begin, end, tokenizer, min_length, max_length):
    assert min_length < max_length, f"min length should be smaller than max lengths, but got min_length: {min_length} and max_length: {max_length}"

    new_content = defaultdict(int)
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(partial(filter_one_sample, tokenizer=tokenizer, min_length=min_length, max_length=max_length), content), total=len(content)):
            #new_content.extend(result)
            new_content[str(result)] += 1

    return new_content

def main(args):
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
    tokenizer = AutoTokenizer.from_pretrained("/home/hao.zhang/dacheng/llama-13B-hf", use_fast=True)

    new_content = filter_all(dataset, args.begin, args.end, tokenizer, 0, 100000)
    print(new_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()
    main(args)
