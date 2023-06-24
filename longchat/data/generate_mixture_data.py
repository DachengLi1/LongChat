"""
Split data into multiple stages for stage-wise sequence length training
"""
import os
import argparse
import random
import json
from typing import Dict, Sequence, Optional
import time

import transformers
from tqdm import tqdm

def filter_long(args):
    cur_total_token = 0
    dump_count = 0

    for json_path in os.listdir(args.in_dir):
        with open(os.path.join(args.in_dir, json_path), 'r') as json_file:
            json_list = list(json_file)

        for json_str in tqdm(json_list):
            if cur_total_token >= args.total_token:
                break
            
            text = json.loads(json_str)["text"]
            cur_total_token += len(tokenizer(text).input_ids)

            if dump_count % 100 == 0:
                print(f"{dump_count}: total - {cur_total_token}")

            with open(args.out_file, 'a') as outfile:
                json.dump({"text": text}, outfile)
                outfile.write('\n')

            dump_count += 1


def main(args):
    cur_total_token = 0
    cur_sample_token = 0
    cur_sample = ""

    dump_count = 0

    for json_path in os.listdir(args.in_dir):
        with open(os.path.join(args.in_dir, json_path), 'r') as json_file:
            json_list = list(json_file)

        for json_str in tqdm(json_list):
            if cur_total_token >= args.total_token:
                break
            
            if cur_sample_token >= args.per_example_token:
                if dump_count % 100 == 0:
                    print(f"{dump_count}: total - {cur_total_token}")
                
                with open(args.out_file, 'a') as outfile:
                    json.dump({"text": cur_sample}, outfile)
                    outfile.write('\n')
                
                dump_count += 1
                cur_sample_token = 0
                cur_sample = ""
            
            text = json.loads(json_str)["text"]
            cur_sample += text
            cur_length = len(tokenizer(text).input_ids)
            cur_total_token += cur_length
            cur_sample_token += cur_length

def shuffle(args):
    time_s = time.time()
    print("Shuffling Data")
    with open(args.out_file, 'r') as json_file:
        json_list = list(json_file)

    random.shuffle(json_list)

    with open(args.shuffle_out_file, 'a') as outfile:
        for json_file in json_list:
            json.dump(json.loads(json_file), outfile)
            outfile.write('\n')
    print(f"Done shuffle in {time.time() - time_s}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--shuffle_out_file", type=str, default="")
    parser.add_argument("--total_token", type=int, required=True)
    parser.add_argument("--per_example_token", type=int, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--long",  action="store_true")
    
    args = parser.parse_args()
    
    global tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )
  #  if args.long:
  #      filter_long(args)
  #  else:
  #      main(args)

    if args.shuffle_out_file != "":
        shuffle(args)
