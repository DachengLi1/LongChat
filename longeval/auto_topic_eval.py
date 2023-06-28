import openai
import argparse
import json


"""
 Example usage: python auto_topic_eval.py --test_file generated_output_file_path \
"""

def chatgpt_auto_eval(json_list):
    print("--------------- Start auto-evaluation, you should verify it does this correctly --------------")
    correct = 0
    for i in range(len(json_list)):
        label = json_list[i].split(',')[0].replace('Label: ', '')
        predict = json_list[i].split(',')[1].replace('Predict: ', '')[2:-2:]
        user_prompt = f"I am testing whether a LLM model can correctly retreieve the first topic, and would like you to help me judge whether the mode ls correct. Please give me 1 for correct and 0 for incorrect. Only give me a single number. Ignore mistakes if the model is paraphasing or using synonyms. Ignore any simple mistakes such as capitalization and punctuation. The ground truth is {label}, the model prediction is {predict}"
        
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
        
        _correct = content == "1" 
        correct += _correct

        output_string = "correct" if _correct else "wrong"

        print(f"Question #{i}: Label: {label}, model output: {predict} - auto-eval goes with {output_string}")


    print(f"---------- End auto-evaluation, predict accuracy {correct / len(json_list)} ---------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    args = parser.parse_args()

    test_file = args.test_file

    with open(test_file, 'r') as json_file:
        json_list = list(json_file)
  
    chatgpt_auto_eval(json_list)
  
