# part of the code retrieved from https://github.com/anadim/the-little-retrieval-test/blob/main/little_retrieval_test.py
import yaml
from yaml import load

import argparse
import logging
import random
import re

import transformers
import torch



def retrieve_expected(lines, random_line_pos):
    correct_line = lines[random_line_pos]
    expected_number = re.search("<\d+>", correct_line).group()[1:-1]

    return expected_number, correct_line

def parse_response(response):
    raise NotImplementedError

def retrieve_prompt_from_lines(lines):
    prompt = ""
    for l in lines:
        prompt += l
    
    return prompt

def block_shuffle(n, B):
    # Create a list of indices divided into blocks of size B
    blocks = [list(range(i, min(i + B, n + 1))) for i in range(1, n + 1, B)]
    # Shuffle the blocks
    random.shuffle(blocks)
    # Flatten the list of blocks into a single list of indices
    shuffled_indices = [i for block in blocks for i in block]

def generate_and_modify_text_file(n, shuffle_flag, B, filename=None):
    """Generates a text file and inserts an execute line at a random position."""
    lines = [f"Testing Long Context\n\n"]
    line_numbers = list(range(1, n + 1))
    if shuffle_flag:
        #if we want random shuffling, B here allows to shuffle every B lines, and within a block there's no shuffle
        line_numbers = block_shuffle(n, B)

    lines.extend([f"line {i}: REGISTER_CONTENT is <{random.randint(1, 50000)}>\n" for i in line_numbers])
    random_line = random.randint(1, n)
    # add the EXECUTE instruction in random line of the text
    # lines.insert(len(lines), f"[EXECUTE THIS]: Go to line {random_line} and report only REGISTER_CONTENT, without any context or additional text, just the number, then EXIT\n")
    lines.insert(len(lines), f"What is the REGISTER_CONTENT in line {random_line}?\n")
    
    if filename is not None:
        with open(filename, "w") as f:
            f.writelines(lines)

    return lines, random_line

def retrieve_cmd_args():
    parser = argparse.ArgumentParser(
        prog='lrt_eval',
        description='lrt_eval'
    )
    parser.add_argument('yaml_path')
    args = parser.parse_args()
    return args


def retrieve_model_response(model, tokenizer, input, lines, random_line_pos, use_gpu):
    if use_gpu:
        response = model.generate(input.input_ids.cuda(), max_new_tokens=256, use_cache=True)
    else:
        response = model.generate(input.input_ids, max_new_tokens=256, use_cache=True)
    response = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(response[0]))
    #response = tokenizer.batch_decode([response], skip_special_tokens=True)

    expected_number, correct_line = retrieve_expected(lines, random_line_pos)

    response_number = re.search("<\d+>", correct_line).group()[1:-1]
    response_line = response

    return expected_number, response_number, correct_line, response_line


def run_experiment(cfg):
    model_name = cfg["model_name"]
    model_path = cfg["model_path"]
    shuffle_flag = cfg["do_shuffle"]
    block_size = cfg["block_size"]

    n_values = cfg["n_values"]
    num_eval_per_len = cfg["num_eval_per_len"]

    use_gpu = cfg["use_gpu"]
    
    if model_name != "None":
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        if use_gpu:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False)

        if use_gpu:
            print(torch.cuda.device_count())
            model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    
    tokenizer.pad_token = tokenizer.unk_token

    for n in n_values:
        correct_count = 0
        n_results = []
        curr_results = []
        for i in range(num_eval_per_len):
            print(f"Running eval {i+1}/{num_eval_per_len} for n = {n}...")
            
            # prepare prompt
            lines, random_line = generate_and_modify_text_file(n, shuffle_flag, block_size)
            prompt = retrieve_prompt_from_lines(lines)
            
            input = tokenizer(prompt, return_tensors="pt")
            token_size = input.input_ids.shape[-1]
            
            print(f"Number of tokens: {token_size}")

            # retrieve model output
            expected_number, \
            model_output, \
            correct_line, \
            incorrect_line = retrieve_model_response(model, tokenizer, input, prompt, random_line, use_gpu)
            print(f"Expected number in the prompt: {expected_number}, Model output: {model_output}")

            # process results
            if expected_number == model_output:
                correct_count += 1
                n_results.append(1)
                curr_results.append([1, expected_number, model_output, correct_line, incorrect_line])
                print("Sweet Success!")
            else:
                n_results.append(0)
                curr_results.append([0, expected_number, model_output, correct_line, incorrect_line])
                print("Oopsies! Didn't get that one right.")
                if correct_line:
                    print(f"Correct result was in this line: {correct_line}")
                if incorrect_line:
                    print(f"Model output was found in this line: {incorrect_line}")
            
        accuracy = (correct_count / num_eval_per_len) * 100
        print(f"Accuracy for n = {n}: {accuracy}%")
        save_results(cfg, curr_results, accuracy, n)


def save_results(cfg, results, accuracy, n):
    name = cfg["model_name"]
    if name is "None":
        name = cfg["model_path"].split("/")[0]

    filename = f"{name}_{n}_non_shuffle_{accuracy}.lrt"

    with open(filename, "w") as f:
        yaml.dump(cfg, f)
        f.write(f"\naccuracy: {accuracy}\n")

        for r in results:
            f.write(r)

def main():
    args = retrieve_cmd_args()
    f = open(args.yaml_path, "r")
    eval_cfg = yaml.load(f, Loader=yaml.CLoader)
    print(yaml.dump(eval_cfg))
    
    run_experiment(eval_cfg)
    

if __name__ == "__main__":
    main()