import time
import argparse
import json

import openai

"""
 Example usage: python auto_topic_eval.py --test_file generated_output_file_path \
"""

MAX_API_RETRY = 5
REQ_TIME_GAP = 3

def get_eval(user_prompt):
    for i in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=500,
            )
            content = response["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            print(e)
            time.sleep(5)
    print(f"Failed after {MAX_API_RETRY} retries.")
    return "error"

def chatgpt_auto_eval(json_list):
    print("--------------- Start auto-evaluation, you should verify it does this correctly --------------")
    correct = 0
    for i in range(len(json_list)):
        label = json_list[i].split(',')[0].replace('Label: ', '')
        predict = json_list[i].split(',')[1].replace('Predict: ', '')[2:-2:]
        user_prompt = f"I am testing whether a LLM model can correctly retreieve the first topic, and would like you to help me judge whether the mode ls correct. Please give me 1 for correct and 0 for incorrect. Only give me a single number. Ignore mistakes if the model is paraphasing or using synonyms. Ignore any simple mistakes such as capitalization and punctuation. The ground truth is {label}, the model prediction is {predict}"
        
        content = get_eval(user_prompt)
        
        _correct = content == "1" 
        correct += _correct

        output_string = "correct" if _correct else "wrong"

        print(f"Question #{i}: Label: {label}, Predict: {predict} - auto-eval goes with {output_string}")
        
        # To avoid rate limit by OPENAI
        time.sleep(REQ_TIME_GAP)

    print(f"---------- End auto-evaluation, predict accuracy {correct / len(json_list)} ---------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    args = parser.parse_args()

    test_file = args.test_file

    with open(test_file, 'r') as json_file:
        json_list = list(json_file)
  
    chatgpt_auto_eval(json_list)
  
