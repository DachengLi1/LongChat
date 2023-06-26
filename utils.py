import json
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib import pyplot as plt

import openai
import tiktoken
import time
import os
import argparse
import yaml

def load_model(path, dtype=torch.bfloat16, device="cuda", num_gpus=1):
    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.bfloat16}
        if num_gpus != 1:
            kwargs["device_map"] = "auto"
            #kwargs["device_map"] = "sequential"  # This is important for not the same VRAM sizes
            # Hard code for A100s
            available_gpu_memory = [16] * num_gpus
            kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
            print(kwargs)
            exit()
    print(num_gpus)
    exit()
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
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
def let_gpt_check_response(topics, response, model_name):
    topics_list = topics[0]
    for i in range(len(topics)):
        if i == 0:
            continue
        topics_list = topics_list + "," + topics[i]

    # prompt = f"Respond True if the topic(s) mentioned in the following paragraph " + \
    #          f"have similar topoics in this list in the same order: {topics_list}; " + \
    #           "otherwise respond False: \n" + \
    #          f"{response}"
    
    # prompt = f"Given this list of {len(topics)} topics separated by ',': {topics_list} " + \
    #     f"\nRespond True if the following list contains {len(topics)} similar topics separated by ',' in the " + \
    #         f"same order. Otherwise, respond False. \n" + \
    #         f"List: {response}"

    prompt = "Compare the topics in two lists and determine the similarity of the topics on a " + \
        "scale of 1 to 100, where 1 indicates very low similarity and 100 indicates " + \
        "very high similarity. The similarity score will be proportional to the " + \
        "number of different topics in the lists.\n\n"
    prompt += f"List 1: {topics} \n"
    prompt += f"List 2: {response} \n"
    prompt += "Question: What is the similarity score between the topics of List 1 and List 2? " + \
        "The score should be proportional to the number of different topics in the lists. \n"
    prompt += "Answer:"

    _, response_line = retrieve_from_openai(prompt, model_name)

    import re
    return re.search("\d+", response_line).group()


# def ask_gpt_for_similarity_score(topic, response, model_name):

def token_counter(model_name, prompt):
    if "gpt" in model_name:
        token_size = len(tiktoken.encoding_for_model(model_name).encode(prompt))
        print(f"Number of tokens: {token_size}")
    else:
        token_size = len(tiktoken.encoding_for_model(model_name).encode(prompt))
        print(f"Number of tokens: {token_size} by using gpt tokenizer as default")

    return token_size


def retrieve_from_openai(prompt, model_name, num_retries=10):
    if "gpt" in model_name:
        token_size = token_counter(model_name, prompt)
        print(f"Number of tokens: {token_size}")
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        token_size = token_counter(model_name, prompt)
        print(f"Number of tokens: {token_size} by using gpt tokenizer as default")

        openai.api_key = os.environ["OPENAI_API_KEY"]
        print("Using openai key as default key")
    
    num_retries = 10
    completion = None
    for attempt in range(num_retries):
        backoff = 2 ** (attempt)

        try:    
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{prompt}"}    
                ],
                temperature = 0
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
        return token_size, -1, "Rate limit"

    response_line = completion.choices[0].message["content"]

    return token_size, response_line

def retrieve_cmd_args(): # setup program params from a given path to a yaml file
    parser = argparse.ArgumentParser(
        prog='lrt_eval',
        description='lrt_eval'
    )
    parser.add_argument('yaml_path')
    args = parser.parse_args()
    f = open(args.yaml_path, "r")
    cfgs = yaml.load(f, Loader=yaml.CLoader)
    print(yaml.dump(cfgs))

    return cfgs

class Conv:
    """a single conversation on a topic"""

    def __init__(self, topic, length, content):
        self.topic = topic
        self.length = length
        self.content = content

class Prompt:
    """the prompt used for testing, composed of multiple  """

    def __init__(self, model_name, id, question_dist):
        self.model_name = model_name
        self.id = id
        self.conv_list = []
        self.topic_list = []
        self.length_list = []
        self.length = -1
        self.question_dist = question_dist

    def add_conv(self, conv):
        self.conv_list.append(conv)
        self.topic_list.append(conv.topic)
        self.length_list.append(conv.length)
    
    def assemble_prompt(self):
        order_word = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh",
                      "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
                      "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
                      "nineteenth", "twentieth", "twenty-first", "twenty-second", "twenty-third",
                      "twenty-fourth", "twenty-fifth", "twenty_sixth", "twenty_seventh", "twenty_eigth",
                      "twenty_ninth", "thritieth"]

        record_prompt = "Below is a record of our previous conversation " + \
            f"on {len(self.topic_list)} different topics. You are the ASSISTANT, and " + \
            "I am the USER. At the beginning of each topic, the USER will say " + \
            "'I would like to discuss the topic of <TOPIC>'. Memorize each " + \
            "<TOPIC>. At the end of the record, I will ask you to retrieve the " + \
            "first topic. Now the record start. "
        
        for conv in self.conv_list:
            record_prompt += conv.content

        question_idx = "first"
        picked_topics = [self.topic_list[0]]
        i = 1
        while ((self.question_dist * i) < len(self.conv_list)):
            question_idx += f", {order_word[(self.question_dist * i)]}"
            picked_topics.append(self.topic_list[self.question_dist * i])
            i += 1

        self.prompt = "A chat between a curious user and an artificial intelligence " + \
            "assistant. The assistant gives helpful, detailed, and polite " + \
            f"answers to the user\'s questions. USER: {record_prompt} Now " + \
            "the record ends. What is the " + question_idx + " topic(s) we discussed? Only give " + \
            "me the topic name(s) in the format of [<topic>, <topic>, ...]. Do not summarize yourself. Do not mention topic order. ASSISTANT:" 

        # self.prompt = "A chat between a curious user and an artificial intelligence " + \
        #     "assistant. The assistant gives helpful, detailed, and polite " + \
        #     f"answers to the user\'s questions. USER: {record_prompt} Now " + \
        #     f"the record ends. What is the {question_idx} topic(s) we discussed? Only give " + \
        #     "me the topic name(s) in the format of [<topic>, <topic>, ...]. Do not summarize yourself. Do not mention topic order. ASSISTANT:" 

        self.length = token_counter(self.model_name, self.prompt)
        
        return self.prompt, picked_topics


