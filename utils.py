import json
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib import pyplot as plt

import openai
import tiktoken
import time
import os

def load_model(path, dtype=torch.bfloat16, device="cuda", num_gpus=1):
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.bfloat16}
        if num_gpus != 1:
            kwargs["device_map"] = "auto"
            kwargs["device_map"] = "sequential"  # This is important for not the same VRAM sizes
            # Hard code for A100s
            available_gpu_memory = [2.5] * num_gpus
            kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs).cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    return model, tokenizer

def load_testcases(test_file):
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)

    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)

    return test_cases

def test(test_case, model, tokenizer, return_summary=True):
    prompt = test_case["prompt"]
    prompt_length = test_case["prompt_length"]
    topics = test_case["topics"]
    input = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input.input_ids.to(model.device), max_new_tokens=100, use_cache=True)[0]
    outputs = outputs[prompt_length:]
    outputs = tokenizer.batch_decode([outputs], skip_special_tokens=True)
    if return_summary:
        summary = f"Label: {topics[0]}, Predict: {outputs}, --- INFO --- Topics: {topics}, Length: {prompt_length}"
        return outputs, summary
    else: 
        return outputs

def attention_span(model, tokenizer, test_case, num_gen_steps=1, raw_attn=False):
    assert num_gen_steps == 1, "Only support span for a single generation step now."
    prompt = test_case["prompt"]
    prompt_length = test_case["prompt_length"]
    topics = test_case["topics"]
    input = tokenizer(prompt, return_tensors="pt")

    output = model(input.input_ids.to(model.device), output_attentions=True)
    num_layer = len(output.attentions)

    attn_mat = torch.cat([a[0, :, -1, :] for a in output.attentions])
    attn_len = attn_mat.shape[-1]
    dist = torch.arange(attn_len, 0, step=-1).cuda()

    span = torch.sum(dist * attn_mat, dim =1)
    span = span.reshape(num_layer, -1)

    span_avg_layer = torch.mean(span, dim=1)
    span_avg_all = torch.mean(span)

    if raw_attn:
        return span, span_avg_layer, span_avg_all, attn_mat
    else:
        return span, span_avg_layer, span_avg_all

def visualize_attn(attn_mat, save_path):
    pass


# some codes taking reference from Auto-GPT
def let_gpt_check_response(topic, response, model_name):
    os.environ['OPENAI_API_KEY'] = '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'
    openai.api_key = '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'

    prompt = f"Respond True if the following paragraph mentions that the first topic is {topic}, " + \
                "otherwise respond False: \n" + \
                f"{response}"

    token_size = len(tiktoken.encoding_for_model(model_name).encode(prompt))
    print(f"Number of tokens: {token_size}")

    num_retries = 10
    completion = None
    for attempt in range(num_retries):
        backoff = 2 ** (attempt)

        try:    
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": f"{prompt}"}        
                ],
                temperature = 0.2
            )
            break
        except openai.error.RateLimitError:
            print("Got rate limit...")
            pass
        except openai.error.APIError as e:
            if e.http_status == 502:
                pass
            else:
                pass

            if attempt == num_retries - 1:
                raise

        time.sleep(backoff)

    if completion is None:
        print(f"Failed to get response after {num_retries} retries")
        return None

    response_line = completion.choices[0].message["content"].lower()
    if "true" in response_line:
        return True
    else:
        return False
