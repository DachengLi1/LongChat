import argparse
import json
import os
import time
from tqdm import tqdm

import torch
import numpy as np

from datasets import load_dataset
from rouge_score import rouge_scorer

from evaluation.utils import load_model

# Example usage: python eval_topics.py --model-name-or-path /data/dacheng/longchat_13b_16K/ --flash --ratio 8


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str,
            # default="/home/haozhang/LongChat/data/dacheng-data/longchat_32K_interpolate",
            default="/home/haozhang/LongChat/data/dacheng-data/longchat_7b_16K",
            # default="/home/haozhang/LongChat/data/dacheng-data/longchat_13b_16K",
            help="model path")
    parser.add_argument("--ratio", type=int, default=8,
            help="target sequence length / original sequence length")
    parser.add_argument("--flash", action='store_true', help="whether to use flash attention to save memory, but slower")
    parser.add_argument("--dataset", type=str, default="qasper")
    args = parser.parse_args()

    SEQ_LEN = 15000

    # load model
    from longchat.train.monkey_patch.llama_interpolate_monkey_patch import replace_llama_with_interpolate
    replace_llama_with_interpolate(args.ratio)

    if args.flash:
        from longchat.train.monkey_patch.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    import transformers
    
    path = args.model_name_or_path
    model, tokenizer = load_model(path, num_gpus=4)
    print("model loaded.")

    # create output dir
    name = os.path.split(path)[-1]
    output_dir = os.path.join("predictions", name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{name}_{args.dataset}_{SEQ_LEN}.raw")
    print(f"output file: {output_file}")

    # load dataset
    data = load_dataset("tau/zero_scrolls", args.dataset)
    test_cases = data["validation"]

    # inference
    print(f"start inference ...")
    tic = time.time()
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    f1 = 0
    predicts = ["Task,ID,Prediction"]
    for x in tqdm(test_cases):
        prompt = x["input"]
        input_ids = tokenizer(prompt).input_ids
        input_ids = torch.tensor([input_ids[-SEQ_LEN:]]).to(model.device)

        outputs = model.generate(input_ids, max_new_tokens=32, use_cache=False)[0][SEQ_LEN:]
        outputs = tokenizer.batch_decode([outputs], skip_special_tokens=True)
        predicts.append(f'{args.dataset},{x["id"]},"{outputs[0]}"')

        score = scorer.score(outputs[0], x["output"])
        f1 += score["rouge1"].fmeasure

    f1 /= len(test_cases)
    print(f"avg f1 over {len(test_cases)} test cases: {f1}")
    print(f"total inference time: {time.time() - tic}")

    with open(output_file, "w") as f:
        for line in predicts:
            f.write(line)

