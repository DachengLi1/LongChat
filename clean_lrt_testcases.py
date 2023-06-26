from pathlib import Path
import copy

from utils import *

lrt_dir = Path("./evaluation/lrt_testcases")
lrt_opt = ["LRT-NL", "LRT-UUID", "LRT-ABCindex"]
output_dir = ["LRT-NL_clean", "LRT-UUID_clean", "LRT-ABCindex_clean"]

prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: "
suffix = " ASSISTANT:"

prefix_length = len(prefix)
suffix_length = len(suffix)

for out_opt, opt in zip(output_dir, lrt_opt):
    input_dir = lrt_dir / Path(opt)
    # print(input_dir)
    output_dir = lrt_dir / Path(out_opt)
    if not output_dir.exists():
        output_dir.mkdir()

    for test_file in input_dir.iterdir():
        # print(test_file)
        test_cases = load_testcases(test_file)

        out_file = output_dir / test_file.name
        print(out_file)
        with open(out_file, "w") as f:
            for test_case in test_cases:
                clean_test_case = copy.deepcopy(test_case)
                prompt = clean_test_case["prompt"]
                prompt = prompt[prefix_length:-suffix_length]
                clean_test_case["prompt"] = prompt
                clean_test_case["prompt_length"] = -1
                json.dump(clean_test_case, f)
                f.write("\n")