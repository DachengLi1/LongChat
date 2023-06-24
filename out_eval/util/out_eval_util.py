import tiktoken
import time
import argparse
import yaml
import transformers
import torch
import openai
import os

from pathlib import Path


def retrieve_cmd_args(): # setup program params from a given path to a yaml file
    parser = argparse.ArgumentParser(
        prog='lrt_eval',
        description='lrt_eval'
    )
    #parser.add_argument('yaml_path')
    parser.add_argument('--model', '-m', help='Specify model path')
    parser.add_argument('--level', '-l', choices=['easy', 'difficult'], help='Specify difficulty of model evaluation')
    args = parser.parse_args()

    # f = open(args.yaml_path, "r")
    HERE = Path(__file__).resolve()
    CFG_PATH = HERE.parent / Path("../out_eval_config.yaml")
    f = open(CFG_PATH, "r")
    cfgs = yaml.load(f, Loader=yaml.CLoader)
    cfgs["model_path"] = args.model
    cfgs["level"] = args.level
    print(yaml.dump(cfgs))

    return cfgs


def load_tokenizer(model_name, model_path):
    if "gpt" in model_name:
        return None
    elif model_name != None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,
                                                               use_fast=False)
        tokenizer.pad_token = tokenizer.unk_token
    elif model_path != None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, 
                                                               use_fast=False)
        tokenizer.pad_token = tokenizer.unk_token
    else:
        raise RuntimeError("Unable to load tokenizer")

    return tokenizer

def token_counter(tokenizer, model_name, model_path, prompt):
    if "gpt" in model_name:
        token_size = len(tiktoken.encoding_for_model(model_name).encode(prompt))
    else:
        input = tokenizer(prompt, return_tensors="pt")
        token_size = input.input_ids.shape[-1]
    
    print(f"Number of tokens: {token_size}")

    return token_size

def query_model(model_name, model_path, prompt, tokenizer):
    if "gpt" in model_name:
        _, response = retrieve_from_openai(prompt, model_name, num_retries=10)
        return response
    elif model_name != None:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
    elif model_path != None:
         model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
    else:
        raise RuntimeError("Unable to load tokenizer")
         
    input = tokenizer(prompt, return_tensors="pt")
    response = model.generate(input.input_ids.cuda(), max_new_tokens=1024, use_cache=True)
    response = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(response[0]))

    return response

def retrieve_from_openai(prompt, model_name, num_retries=10):
    if "gpt" in model_name:
        token_size = token_counter(None, model_name, None, prompt)
        print(f"Number of tokens: {token_size}")

        os.environ['OPENAI_API_KEY'] = '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'
        openai.api_key = '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'
    else:
        token_size = token_counter(model_name, prompt)
        print(f"Number of tokens: {token_size} by using gpt tokenizer as default")

        os.environ['OPENAI_API_KEY'] = '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'
        openai.api_key = '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'
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

class Conv:
    """a single conversation on a topic"""

    def __init__(self, topic, length, content):
        self.topic = topic
        self.length = length
        self.content = content

class Prompt:
    """the prompt used for testing, composed of multiple  """

    def __init__(self, model_name, model_path, id, question_dist, tokenizer):
        self.model_name = model_name
        self.model_path = model_path
        self.id = id
        self.conv_list = []
        self.topic_list = []
        self.length_list = []
        self.length = -1
        self.question_dist = question_dist
        self.tokenizer = tokenizer

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

        self.length = token_counter(self.tokenizer, self.model_name, self.model_path, self.prompt)
        
        return self.prompt, picked_topics