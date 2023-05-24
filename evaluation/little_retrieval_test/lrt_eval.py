import yaml
from yaml import load

import argparse
import logging
import random
import re

import transformers
import torch

def retrieve_expected(lines, random_line_pos):
    line_num_in_content = int(lines[random_line_pos - 1].split("Go to line ")[1].split(" and")[0])

    correct_line = None
    for line in lines:
        if f"line {line_num_in_content}:" in line:
            expected_number = int(line.split("REGISTER_CONTENT is <")[1].split(">\n")[0])
            correct_line = line
            break

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
        ine_numbers = block_shuffle(n, B)

    lines.extend([f"line {i}: REGISTER_CONTENT is <{random.randint(1, 50000)}>\n" for i in line_numbers])
    random_line = random.randint(1, n)
    # add the EXECUTE instruction in random line of the text
    lines.insert(random_line - 1, f"[EXECUTE THIS]: Go to line {random_line} and report only REGISTER_CONTENT, without any context or additional text, just the number, then EXIT\n")
    
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
        response = model.generate(input.input_ids.cuda(), max_new_tokens=50, use_cache=True)
    else:
        response = model.generate(input.input_ids, max_new_tokens=50, use_cache=True)
    response = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(response[0]))
    response_line_num = parse_response(response) # may not be necessary
    # response = tokenizer.batch_decode(response, skip_special_tokens=True)
    
    expected_number, correct_line = retrieve_expected(lines, random_line_pos)

    model_output_str = response['completion'].strip()
    model_output = int(re.search(r'\d+', model_output_str).group()) if re.search(r'\d+', model_output_str) else None

    incorrect_line = None
    if expected_number != model_output:
        for line in lines:
            if f"<{model_output}>" in line:
                incorrect_line = line
                break

    return expected_number, model_output, correct_line, incorrect_line
    

def run_experiment(cfg):
    model_name = cfg["model_name"]
    model_path = cfg["model_path"]
    shuffle_flag = cfg["do_shuffle"]
    block_size = cfg["block_size"]

    n_values = cfg["n_values"]
    num_eval_per_len = cfg["num_eval_per_len"]

    use_gpu = cfg["use_gpu"]
    
    if model_name is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        if use_gpu:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False)

        if use_gpu:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    
    tokenizer.pad_token = tokenizer.unk_token
    
    accuracies = []
    individual_results = []

    for n in n_values:
        correct_count = 0
        n_results = []
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
                print("Sweet Success!")
            else:
                n_results.append(0)
                print("Oopsies! Didn't get that one right.")
                if correct_line:
                    print(f"Correct result was in this line: {correct_line}")
                if incorrect_line:
                    print(f"Model output was found in this line: {incorrect_line}")
            
        accuracy = (correct_count / num_eval_per_len) * 100
        print(f"Accuracy for n = {n}: {accuracy}%")
        accuracies.append(accuracy)
        individual_results.append(n_results)
        save_accuracies(n_values, accuracies, individual_results, model_name, shuffle_flag, block_size)


def main():
    args = retrieve_cmd_args()
    f = open(args.yaml_path, "r")
    eval_cfg = yaml.load(f, Loader=yaml.CLoader)
    
    run_experiment(eval_cfg)
    

if __name__ == "__main__":
    main()