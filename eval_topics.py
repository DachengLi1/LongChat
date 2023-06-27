import argparse
import os
import json

import torch
import numpy as np

from fastchat.model import load_model, get_conversation_template
from utils import load_testcases, test_with_template

# Example usage: python eval_topics.py --model-name-or-path /data/dacheng/longchat_13b_16K/ --flash --ratio 8

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True, help="model path")
    parser.add_argument("--ratio", type=int, required=True, help="target sequence lnegth / original sequence length")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use")
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
    if path[-1] == "/":
        path = path[:-1]
    name = path.split("/")[-1]

    output_dir = "evaluation/topics/predictions_with_template"
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_dir = os.path.join(output_dir, name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print(f"output to {output_dir}")

    model, tokenizer = load_model(
        path,
        device="cuda",
        num_gpus=args.num_gpus,
        max_gpu_memory="30GiB",
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )

    for num_topics in [15, 20, 25]:
        print(f"Start testing {num_topics} per prompt!")
        avg_length = 0

        test_file = f"evaluation/topics/testcases_clean/{num_topics}_topics.jsonl"

        output_file = os.path.join(output_dir, f"{num_topics}_response.txt")
        
        test_cases = load_testcases(test_file)
        with open(test_file, 'r') as json_file:
            json_list = list(json_file)

        for test_case in test_cases:        
            conv = get_conversation_template("vicuna")
            outputs, prompt_length, summary = test_with_template(test_case, conv, model, tokenizer, use_cache=not args.flash, return_summary=True)
            print(summary)

            avg_length += prompt_length / len(test_cases)
            with open(output_file, "a+") as f: 
                f.write(summary)
                f.write("\n")

        print(f"Finish testing {num_topics} per prompt with average length {avg_length}")
