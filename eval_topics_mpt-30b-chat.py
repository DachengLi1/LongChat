import argparse
import os
import json

import torch
import numpy as np
from utils import  load_testcases, test, test_with_template

from fastchat.model import load_model, get_conversation_template

# Example usage: python eval_topics_mpt-30b-chat.py --model-name-or-path mosaicml/mpt-30b-chat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True, help="model path")
    args = parser.parse_args()

    import transformers
    
    path = args.model_name_or_path
    name = os.path.split(path)[-1]

    output_dir = "evaluation/topics/predictions"
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_dir = os.path.join(output_dir, name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print(f"output to {output_dir}")

    #model, tokenizer = load_model(path, num_gpus=8)
    model, tokenizer = load_model(
        path,
        device="cuda",
        num_gpus=8,
        max_gpu_memory="30GiB",
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )

    for num_topics in [15, 20, 25, 30]:
        print(f"Start testing {num_topics} per prompt!")
        test_file = f"evaluation/topics/testcases_clean/{num_topics}_topics.jsonl"

        output_file = os.path.join(output_dir, f"{num_topics}_response.txt")
        
        test_cases = load_testcases(test_file)
        with open(test_file, 'r') as json_file:
            json_list = list(json_file)

        for test_case in test_cases:       
            conv = get_conversation_template(args.model_name_or_path)
            outputs, summary = test_with_template(test_case, conv, model, tokenizer, use_cache=False, return_summary=True)
            print(summary)
            with open(output_file, "a+") as f: 
                f.write(summary)
                f.write("\n")
