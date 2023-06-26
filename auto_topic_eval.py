import openai
import torch
import argparse
import json
import time
from fastchat.model import load_model, get_conversation_template

"""
 Example usage: python auto_topic_eval.py --model-name-or-path /data/dacheng/vicuna-7b \
        --test_file evaluation/topics/predictions_with_template/longchat_32K_interpolate/15_response.txt \
        --num_gpus 8
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    test_file = args.test_file

    model, tokenizer = load_model(
        args.model_name_or_path,
        device="cuda",
        num_gpus=args.num_gpus,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )
    
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)
   
    print("--------------- Start auto-evaluation, you should verify it does this correctly --------------")
    correct = 0
    for i in range(len(json_list)):
        label = json_list[i].split(',')[0].replace('Label: ', '')
        predict = json_list[i].split(',')[1].replace('Predict: ', '')[3:-2]
        prompt = f"I am testing whether a LLM model can correctly retreieve the first topic, and would like you to help me judge whether the mode ls correct. Please give me 1 for correct and 0 for incorrect. Only give me a single number. Ignore mistakes if the model is paraphasing or using synonyms. Ignore any simple mistakes such as capitalization and punctuation. The ground truth is {label}, the model prediction is {predict}"
        
        conv = get_conversation_template(args.model_name_or_path)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer([prompt]).input_ids
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            max_new_tokens=10,
        )
        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        
        _correct = outputs == "1" 
        correct += _correct

        output_string = "correct" if _correct else "wrong"

        print(f"# {i}: {label}, {predict} - auto-eval goes with {output_string}")


    print(f"---------- End auto-evaluation, predict accuracy {correct / len(json_list)} ---------------")
