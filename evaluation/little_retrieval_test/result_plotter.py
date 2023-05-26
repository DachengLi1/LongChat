import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re
from pathlib import Path

def retrieve_data(path):
    f = open(path, "r")
    
    reached_data = False
    correct_line_nums = []
    while True:
        line = f.readline()
        if "token_size" in line:
            reached_data = True
            continue
        
        if not line:
            break

        if reached_data:
            is_correct_line = re.search("^\[\d, ", line)
            if is_correct_line is not None:
                is_correct_line = bool(int(is_correct_line.group()[1]))

                if is_correct_line:
                    line_num = int(re.search("'line \d+:", line).group()[6:-1])
                    correct_line_nums.append(line_num)
            else:
                raise ValueError("cannot parse file")

    return correct_line_nums


def retrieve_cmd_args():
    parser = argparse.ArgumentParser(
        prog='result_plotter',
        description='result_plotter'
    )
    parser.add_argument('data_path')
    args = parser.parse_args()
    return args


def plot(args):
    path = Path(args.data_path)

    f_name = path.stem

    data = retrieve_data(path)
    plt.hist(data, color = "blue", edgecolor = 'black', bins = len(data))
    plt.title(f"{f_name}")
    plt.xlabel("line #")
    plt.ylabel("# of corrects")
    plt.savefig(path.parents[0] / Path(f"{f_name}_dist.png"))


def main():
    args = retrieve_cmd_args()
    plot(args)

if __name__ == "__main__":
    main()