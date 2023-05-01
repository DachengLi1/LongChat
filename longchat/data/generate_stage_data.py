"""
Split data into multiple stages for stage-wise sequence length training
"""
import os
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from typing import Dict, Sequence, Optional

import transformers
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
    tokenized_lens = []
    conversations = sample["conversations"]
    conversations = conversations[: len(conversations) // 2 * 2]
    for c in conversations:
        length = len(tokenizer(c["value"]).input_ids) + 6
        tokenized_lens.append(length)
    total_length = sum(tokenized_lens)

    start_idx = 0
    cur_len = 0

    if len(conversations) % 2 != 0 or len(conversations) < 2:
        return []

    # Filter out samples that have lengths outside of the range
    if not (min_length <= total_length and total_length < max_length):
        return []
    else:
        return [make_sample(sample, start_idx, len(conversations))]

def filter_all(content, begin, end, tokenizer, min_length, max_length):
    assert min_length < max_length, f"min length should be smaller than max lengths, but got min_length: {min_length} and max_length: {max_length}"

    content = content[begin:end]
    new_content = []
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(partial(filter_one_sample, tokenizer=tokenizer, min_length=min_length, max_length=max_length), content), total=len(content)):
            new_content.extend(result)

    return new_content

def filter_invalid_roles(content):
    new_content = []
    for i, c in enumerate(content):
        roles = ["human", "gpt"]
        if len(c["conversations"]) <= 0:
            continue

        valid = True
        for j, s in enumerate(c["conversations"]):
            if s["from"] != roles[j % 2]:
                valid = False
                break

        if valid:
            new_content.append(c)

    return new_content


def main(args):
    content = json.load(open(args.in_file, "r"))
    stages = [2048, 4096, 8192, 16384, 32768]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=max(stages), #args.max_length,
        padding_side="right",
        use_fast=False,
    )
    for idx, max_length in enumerate(stages):
        if idx == 0:
            min_length = 0
        else:
            min_length = stages[idx - 1]
        print(f"filtering {min_length} to {max_length}")
        new_content = filter_all(content, args.begin, args.end, tokenizer, min_length, max_length)
        new_content = filter_invalid_roles(new_content)
        print(f"total: {len(content)}, between {min_length} and {max_length}: {len(new_content)}")
        out_file = os.path.join(args.out_dir, f"max_length_{max_length}.json")
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        json.dump(new_content, open(out_file, "w"), indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    #parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    main(args)
