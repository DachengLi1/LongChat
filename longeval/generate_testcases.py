import os
from utils import generate_topics_testcases, generate_lines_testcases, retrieve_cmd_args

if __name__ == "__main__":
    cfgs = retrieve_cmd_args()

    output_dir = os.path.join(cfgs["output_dir"], cfgs["task"], "testcases/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if cfgs["task"] == "topics":
        generate_topics_testcases(cfgs, output_dir)
    else:
        generate_lines_testcases(cfgs, output_dir)