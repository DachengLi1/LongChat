import argparse
import json
import os
import time
from tqdm import tqdm

import torch
import numpy as np

from datasets import load_dataset
from rouge_score import rouge_scorer

import openai


def fix_prompt(prompt):
    paragraphs = prompt.split("\n\n")
    new_prompt = prompt + "\n\nQuestions:\n" + paragraphs[0] + "\n\nAnswer:\n"
    return new_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="gpt-3.5")
    parser.add_argument("--dataset-version", type=str, default="tau/zero_scrolls")
    parser.add_argument("--dataset", type=str, default="qasper")
    args = parser.parse_args()

    cut_word_len = 2000

    # create output dir
    name = args.model_name_or_path
    output_dir = os.path.join("predictions", name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{name}_{args.dataset}_{cut_word_len}.raw")
    print(f"output file: {output_file}")

    # load dataset
    data = load_dataset(args.dataset_version, args.dataset)
    if args.dataset_version=="tau/scrolls":
        test_cases = data["validation"]
        test_cases = test_cases.shuffle(seed=1123).select(range(200))
    else:
        test_cases = data["validation"]

    # inference
    print(f"start inference ...")
    tic = time.time()
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    predicts = ["Task,ID,Prediction"]
    f1_scores = {}
    for x in tqdm(test_cases):
        prompt = x["input"]
        if args.dataset_version == "tau/scrolls":
            prompt = fix_prompt(prompt)
        words = prompt.split(" ")
        prompt = " ".join(words[:cut_word_len] + words[-cut_word_len:])

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        while True:
            try:
                ret = openai.ChatCompletion.create(
                    model=name, messages=messages, temperature=0)
                break
            except openai.error.OpenAIError as e:
                print(type(e), e)
                time.sleep(10)

        outputs = [ret["choices"][0]["message"]["content"]]
        predicts.append(f'{args.dataset},{x["id"]},"{outputs[0]}"')

        # print("---------------------")
        # print(x["output"])
        # print("---------------------")
        # print(prompt)
        # print("---------------------")
        # print(outputs[0])
        # print("=====================")
        max_score = 0
        for l in [512]:
            score = scorer.score(outputs[0], x["output"])
            max_score = max(max_score, score["rouge1"].fmeasure)
        qid = x["id"]
        if qid not in f1_scores:
            f1_scores[qid] = max_score
        else:
            f1_scores[qid] = max(max_score, f1_scores[qid])

    scores = list(f1_scores.values())
    avg_f1 = np.mean(scores)
    print(f"avg f1 over {len(test_cases)} test cases: {avg_f1:.3f}")
    print(f"total inference time: {time.time() - tic:.0f} s")

    with open(output_file, "w") as f:
        for line in predicts:
            f.write(line + "\n")
