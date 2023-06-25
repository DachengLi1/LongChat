import argparse
import os
import json

import torch
import numpy as np


from utils import load_model, load_testcases, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True, help="model path")
    parser.add_argument("--ratio", type=int, required=True, help="target sequence lnegth / original sequence length")
    parser.add_argument("--flash", action='store_true', help="whether to use flash attention to save memory, but slower")
    args = parser.parse_args()

    # Monkey Patch
    from longchat.train.monkey_patch.llama_interpolate_monkey_patch import replace_llama_with_interpolate
    replace_llama_with_interpolate(args.ratio)

    if args.flash:
        from longchat.train.monkey_patch.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

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

    model, tokenizer = load_model(path, num_gpus=1)
 
    for num_topics in [25]:
        print(f"Start testing {num_topics} per prompt!")
        test_file = f"evaluation/topics/testcases/{num_topics}_topics.jsonl"

        output_file = os.path.join(output_dir, f"{num_topics}_response.txt")
        
        test_cases = load_testcases(test_file)
        with open(test_file, 'r') as json_file:
            json_list = list(json_file)

        for test_case in test_cases:        
            outputs, summary = test(test_case, model, tokenizer, use_cache=not args.flash, return_summary=True)
            print(summary)
            with open(output_file, "a+") as f: 
                f.write(summary)
                f.write("\n")
