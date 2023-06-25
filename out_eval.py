import json
import math
import re
import random
import numpy as np

from pathlib import Path
from out_eval.out_eval_util import *

# from longchat.train.monkey_patch.llama_interpolate_monkey_patch import replace_llama_with_interpolate
# replace_llama_with_interpolate()

# from longchat.train.monkey_patch.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
# replace_llama_attn_with_flash_attn()

# from longchat.train.monkey_patch.llama_xformer_monkey_patch import replace_llama_attn_with_xformer
# replace_llama_attn_with_xformer()

SCRIPT_PATH = Path(__file__).resolve()

REPO_DIR = SCRIPT_PATH.parent
WORKING_DIR = REPO_DIR / Path("out_eval")


def run_lrt_exp(cfgs, tokenizer):
    TEST_DIR = WORKING_DIR / Path(f"{cfgs['model_name']}_lrt_testcases") \
        if not cfgs["use_fixed_testcases"] else REPO_DIR / Path(cfgs["lrt_testcases_dir"])
    test_files = list(TEST_DIR.iterdir())

    output_dir = WORKING_DIR / Path(f"{cfgs['model_name']}_lrt_predictions")
    if not output_dir.exists():
        output_dir.mkdir()
    # load model
    model = load_model(cfgs["model_name"], cfgs["model_path"], cfgs["local_model"], cfgs["gpu_id"])

    for n in cfgs["num_lines"]:
        print(f"**********Start testing {n} lines per LRT prompt**********")
        
        for file in test_files:
            if re.search(f"^{n}_lines_", \
                            str(file.name)) is not None:
                    test_file = file
                    break
        
        with open(test_file, 'r') as json_file:
            pt_list = list(json_file)
        
        output_file = output_dir / Path(f"{test_file.stem}_lrt.prediction")
        num_correct = 0
        for test_case in pt_list:
            test_case = json.loads(test_case)
            prompt = test_case["prompt"]
            correct_line = test_case["correct_line"]
            # random_line = test_case["random_line"]
            # num_lines = test_case["num_lines"]
            # token_size = test_case["token_size"]
            expected_number = test_case["expected_number"]

            _, response = query_model(cfgs["model_name"], model, prompt, tokenizer, cfgs["gpu_id"], cfgs["use_flash"])
            
            response_number = re.findall("\d+", response)
            if response_number is not None:
                response_number = int(response_number[-1])
            else:
                print(f"Got unparsable result: {response}")
            
            if expected_number == response_number:
                summary = "[1]"
                num_correct += 1
            else:
                summary = "[0]"
            
            summary += f"Label: {expected_number}, Prediction: {response_number}, Correct_line: {correct_line[:-1]}, Model output: {response}"
            print(summary)
            with open(output_file, "a+") as f:
                f.write(summary)
                f.write("\n")

        acc = num_correct / len(pt_list)
        with open(output_file, "a+") as f:
            f.write(f"\naccuracy: {acc}")
            # f.write(f"\ntoken size: {token_size}\n")
            f.close()
        output_file.rename(output_dir / Path(f"{test_file.stem}_{acc}.prediction"))


def run_conv_eval_exp(cfgs, tokenizer):
    TEST_DIR = WORKING_DIR / Path(f"{cfgs['model_name']}_testcases") \
        if not cfgs["use_fixed_testcases"] else REPO_DIR / Path(cfgs["testcases_dir"])

    output_dir = WORKING_DIR / Path(f"{cfgs['model_name']}_predictions")
    test_files = list(TEST_DIR.iterdir())

    if not output_dir.exists():
        output_dir.mkdir()

    # load model
    model = load_model(cfgs["model_name"], cfgs["model_path"], cfgs["local_model"], cfgs["gpu_id"])

    for num_topics in cfgs["num_topics"]:
        print(f"**********Start testing {num_topics} topics per prompt**********")

        if not cfgs["use_fixed_testcases"]:
            for file in test_files:
                if re.search(f"^{num_topics}_topics_{cfgs['question_dist']}", \
                            str(file.name)) is not None:
                    test_file = file
                    break
        else:
            test_file = TEST_DIR / Path(f"{num_topics}_topics.jsonl")

        with open(test_file, 'r') as json_file:
            conversation_list = list(json_file)

        total_sim_score = 0

        output_file = output_dir / Path(f"{test_file.stem}.prediction")
        for test_case in conversation_list:
            test_case = json.loads(test_case)
            prompt = test_case["prompt"]
            if not cfgs["use_fixed_testcases"]:
                prompt_length = test_case["total_length"]
                topics = test_case["topics"]
                picked_topics = test_case["picked_topics"]
                lenth_dist = [float(i)/sum(test_case["length"]) for i in test_case["length"]]
            else:
                topics = test_case["topics"]
                prompt_length = test_case["prompt_length"]

            token_size, response = query_model(cfgs["model_name"], model, 
                                    prompt, tokenizer, cfgs["gpu_id"], cfgs["use_flash"])

            if not cfgs["use_fixed_testcases"]:
                summary = f"Label:      {picked_topics}, \nPrediction: {response}, \ntopics:     {topics}, \nprompt_length: {prompt_length}, \nlength_dist: {lenth_dist}\n"
            else:
                summary = f"Label: {topics[0]}, Prediction: {response}, --- INFO --- Topics: {topics}, Length: {prompt_length}"

            print(summary)
            with open(output_file, "a+") as f:
                f.write(summary)
                f.write("\n")
        
        acc = total_sim_score / len(conversation_list)
        with open(output_file, "a+") as f:
            f.write(f"\naccuracy: {acc}")
            f.write(f"\ntoken size: {token_size}\n")
            f.close()
        output_file.rename(output_dir / Path(f"{test_file.stem}_{acc}.prediction"))
        # print(f"accuracy: {acc}")

def generate_conversations(cfgs, tokenizer):
    conv_list = []

    output_dir = WORKING_DIR / Path(f"{cfgs['model_name']}_conveval_testcases")
    if not output_dir.exists():
        output_dir.mkdir()
    
    with open(REPO_DIR / Path("evaluation/topics/conversations.jsonl"), 'r') as json_file:
        conv_obj_list = list(json_file)

    for conv_obj in conv_obj_list:
        conv_obj = json.loads(conv_obj)
        conv_len = token_counter(tokenizer, cfgs["model_name"], cfgs["model_path"], conv_obj["conversation"])
        conv_list.append(Conv(conv_obj["topic"], conv_len, conv_obj["conversation"]))

    # generate prompts for each num_topics
    for num_topics in cfgs["num_topics"]:

        prompt_list = []
        
        for i in range(cfgs["num_test_samples"]):
            prompt = Prompt(cfgs["model_name"], cfgs["model_path"], i, cfgs["question_dist"], tokenizer)
            indices = np.random.choice(list(range(len(conv_list))), size=num_topics, replace=False)
            for idx in indices:
                prompt.add_conv(conv_list[idx])
            prompt_list.append(prompt)
            
            prompt = None
        
        # write to output file
        avg_len = 0
        output_path = output_dir / Path(f"{num_topics}_topics_{cfgs['question_dist']}.jsonl")
        f = open(output_path, "w")
        for i, p in enumerate(prompt_list):
            pt, picked_topics = p.assemble_prompt()
            
            curr_output = {"test_id": p.id, 
                           "picked_topics": picked_topics,
                           "topics": p.topic_list, 
                           "length": p.length_list, 
                           "total_length": p.length,
                           "prompt": pt}
            avg_len += p.length
            json.dump(curr_output, f)
            f.write("\n")
        avg_len = math.ceil(avg_len/len(prompt_list))
        f.close()
        output_path.rename(output_dir / Path(f"{num_topics}_topics_{cfgs['question_dist']}_{avg_len}.jsonl"))

def generate_lrt(cfgs, tokenizer):
    output_dir = WORKING_DIR / Path(f"{cfgs['model_name']}_lrt_testcases")
    if not output_dir.exists():
        output_dir.mkdir()

    for n in cfgs["num_lines"]:
        output_path = output_dir / Path(f"{n}_lines.jsonl")
        f = open(output_path, "w")
        avg_token_size = 0
        for i in range(cfgs["num_test_samples"]):          
            lines = [f"Testing Long Context\n\n"]
            line_numbers = list(range(1, n + 1))
            lines.extend([f"line {i}: REGISTER_CONTENT is <{random.randint(1, 50000)}>\n" for i in line_numbers])
            random_line = random.randint(1, n)

            lines.insert(len(lines), f"Tell me what is the REGISTER_CONTENT in line {random_line}? I need the number.\n")
            expected_number, correct_line = retrieve_expected(lines, random_line)

            prompt = retrieve_prompt_from_lines(lines)
            token_size = token_counter(tokenizer, cfgs["model_name"], None, prompt)
            avg_token_size += token_size
            output = {
                "random_line": random_line, # this is the line to retrieve
                "expected_number": expected_number,
                "num_lines": n,
                "token_size": token_size,
                "correct_line": correct_line,
                "prompt": prompt}

            json.dump(output, f)
            f.write("\n")
        f.close()
        avg_token_size = math.ceil(avg_token_size / cfgs["num_test_samples"])
        output_path.rename(output_dir / Path(f"{n}_lines_{avg_token_size}.jsonl"))


def main():
    cfgs = retrieve_cmd_args()

    if cfgs["use_monkey_patch"]:
        if cfgs["use_flash"]:
            # from longchat.train.monkey_patch.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
            # replace_llama_attn_with_flash_attn()

            from longchat.train.monkey_patch.llama_xformer_monkey_patch import replace_llama_attn_with_xformer
            replace_llama_attn_with_xformer()
        
        from longchat.train.monkey_patch.llama_interpolate_monkey_patch import replace_llama_with_interpolate
        replace_llama_with_interpolate(cfgs["ratio"])

    tokenizer = load_tokenizer(cfgs["model_name"], cfgs["model_path"], cfgs["local_model"])
    
    if cfgs["level"] == "easy":
        if cfgs["generate_conversations"]:
            generate_conversations(cfgs, tokenizer)
        run_conv_eval_exp(cfgs, tokenizer)
    else:
        if cfgs["generate_lrt_prompt"] and not cfgs["use_fixed_testcases"]:
            generate_lrt(cfgs, tokenizer)
        
        # run_lrt_exp(cfgs, tokenizer)

if __name__ == "__main__":
    main()