import os
import copy

from utils import *

topics_dir = "./evaluation/topics"
input_dir = os.path.join(topics_dir, "testcases")
output_dir = os.path.join(topics_dir, "testcases_clean")

prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: "
suffix = " ASSISTANT:"

prefix_length = len(prefix)
suffix_length = len(suffix)

for test_file in os.listdir(input_dir):
    test_cases = load_testcases(os.path.join(input_dir, test_file))   
    
    out_file = os.path.join(output_dir, test_file)
    with open(out_file, "w") as f:
        for test_case in test_cases:
            clean_test_case = copy.deepcopy(test_case)
            prompt = clean_test_case["prompt"]
            prompt = prompt[prefix_length:-suffix_length]
            clean_test_case["prompt"] = prompt
            clean_test_case["prompt_length"] = -1
            json.dump(clean_test_case, f)
            f.write("\n")


