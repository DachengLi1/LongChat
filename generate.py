import argparse
import os
import json
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import pipeline

from fastchat.model import get_conversation_template
from longeval.utils import maybe_monkey_patch, get_output_dir, longeval_load_model, load_testcases, test_topics_one_sample, test_lines_one_sample 



def eval_scrolls(model, tokenizer, args):
    dataset = load_dataset("tau/scrolls", args.dataset)
    pass










if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", default="lmsys/longchat-7b-16k", type=str, help="model path")
    parser.add_argument("--dataset", default="qasper", type=str, help="Which evaluation task to use. currently support [topics, lines]")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--max_gpu_memory", type=int, default=40, help="max per gpu memory in GiB. A100 is 40 or 80.")
    parser.add_argument("--longchat_flash_attn", action='store_true', help="Only apply to longchat models. Whether to enable flash attention to save memory, but slower.")
    parser.add_argument("--longchat_ratio", type=int, default=8, help="Only apply to longchat models. Use ratio=8 for 16K context length model. Only ratio=8 is supported now.")
    parser.add_argument("--eval_shortest_only", action='store_true', default=0, help="Only eval the shortest case for illustration purpose")
    args = parser.parse_args()

    args.model_name_or_path = "mosaicml/mpt-7b-storywriter"

    model, tokenizer = longeval_load_model(args)
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')

    dataset = load_dataset("tau/scrolls", args.dataset)
    dataset = dataset["validation"]

    prediction = {}
    for idx in range(len(dataset)):
        prompt = dataset[idx]["input"]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = pipe(prompt, max_new_tokens=500, do_sample=True, use_cache=True)[0]['generated_text'][len(prompt):]
        prediction.update({dataset[idx]["id"]: output})
        print(dataset[idx]["id"], output)
    
    json_object = json.dumps(prediction, indent=4)
    with open(f"{args.model_name_or_path.replace('/','-')}-{args.dataset}.json", "w") as f:
        f.write(json_object)


