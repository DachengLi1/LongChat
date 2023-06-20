# code retrieved from https://github.com/anadim/the-little-retrieval-test/blob/main/little_retrieval_test.py

import os
import random
import anthropic
import re
import matplotlib.pyplot as plt
import requests

# Set actual API key
os.environ["ANTHROPIC_API_KEY"] = "your-key-here"

def block_shuffle(n, B):
    # Create a list of indices divided into blocks of size B
    blocks = [list(range(i, min(i + B, n + 1))) for i in range(1, n + 1, B)]
    # Shuffle the blocks
    random.shuffle(blocks)
    # Flatten the list of blocks into a single list of indices
    shuffled_indices = [i for block in blocks for i in block]



def generate_and_modify_text_file(filename, n, shuffle_flag, B):
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
    with open(filename, "w") as f:
        f.writelines(lines)

    return lines, random_line



def get_model_response(prompt, model_name):
    c = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])
    
    while True:  # This will loop indefinitely until a break statement is encountered. 
        try:
            response = c.completion(
                prompt=prompt,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens_to_sample=10,
                model=model_name,
                temperature=0
            )
            return response
        except requests.exceptions.HTTPError as e:
            print(f"A connection error occurred: {e}. Retrying...")
        except requests.exceptions.RequestException as e:
            print("An error occurred while making the request:", e)
        except KeyboardInterrupt:
            raise KeyboardInterrupt            
        except Exception as e:
            print('Generic')            
            


def test_model(lines, random_line, model_name):
    prompt_text = f"{anthropic.HUMAN_PROMPT} Here are the file contents:\n" + "".join(lines)[:-1] + f"\n{anthropic.AI_PROMPT}"
    response = get_model_response(prompt_text,model_name)
    line_num_in_content = int(lines[random_line - 1].split("Go to line ")[1].split(" and")[0])

    correct_line = None
    for line in lines:
        if f"line {line_num_in_content}:" in line:
            expected_number = int(line.split("REGISTER_CONTENT is <")[1].split(">\n")[0])
            correct_line = line
            break

    model_output_str = response['completion'].strip()
    model_output = int(re.search(r'\d+', model_output_str).group()) if re.search(r'\d+', model_output_str) else None

    incorrect_line = None
    if expected_number != model_output:
        for line in lines:
            if f"<{model_output}>" in line:
                incorrect_line = line
                break

    return expected_number, model_output, correct_line, incorrect_line


def save_accuracies(n_values, accuracies, individual_results, model_name, shuffle_flag, B):
    """Saves accuracies and individual results to a file."""
    filename = f"accuracies_{model_name}_{'shuffled' if shuffle_flag else 'unshuffled'}_{B}.txt"
    with open(filename, "w") as f:
        for n, accuracy, results in zip(n_values, accuracies, individual_results):
            f.write(f"n = {n}: Accuracy = {accuracy}%\n")
            f.write(f"Individual results: {results}\n")


def save_accuracies(n_values, accuracies, individual_results, model_name, shuffle_flag):
    filename = f"accuracies_{model_name}_{'shuffled' if shuffle_flag else 'unshuffled'}.txt"
    with open(filename, "w") as f:
        for n, accuracy, results in zip(n_values, accuracies, individual_results):
            f.write(f"n = {n}: Accuracy = {accuracy}%\n")
            f.write(f"Individual results: {results}\n")


def plot_accuracies(n_values, accuracies, model_name, shuffle_flag):
    plt.plot(n_values, accuracies)
    plt.xlabel('n')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy vs. n for {model_name} with shuffling = {"on" if shuffle_flag else "off"}')
    plt.grid(True)
    plt.show()


def main():
    # can test the following models
    #  "claude-v1-100k"
    #  "claude-instant-v1-100k"
    #  "claude-instant-v1.1-100k"
    #   "claude-v1.3-100k"

    model_name = "claude-instant-v1.1-100k"
    shuffle_flag = True
    B = 10 

    n_values = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 6500]
    num_tests = 50
    accuracies = []
    individual_results = []
    filename = f"prompt_{model_name}_{'shuffled' if shuffle_flag else 'unshuffled'}_{B}.txt"


    for n in n_values:
        correct_count = 0
        n_results = []
        for i in range(num_tests):
            print(f"\nRunning test {i + 1}/{num_tests} for n = {n}...")
            lines, random_line = generate_and_modify_text_file(filename, n, shuffle_flag, B)
            num_tokens = anthropic.count_tokens("".join(lines))
            
            print("Number of tokens in this prompt: ", num_tokens)

            expected_number, model_output, correct_line, incorrect_line = test_model(lines, random_line, model_name)
            print(f"Expected number in the prompt: {expected_number}, Model output: {model_output}")
            
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
        
        accuracy = (correct_count / num_tests) * 100
        print(f"Accuracy for n = {n}: {accuracy}%")
        accuracies.append(accuracy)
        individual_results.append(n_results)
        save_accuracies(n_values, accuracies, individual_results, model_name, shuffle_flag, B)

    plot_accuracies(n_values, accuracies, model_name, shuffle_flag)


if __name__ == "__main__":
    main()