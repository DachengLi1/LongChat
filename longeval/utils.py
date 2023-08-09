import json
import time
import os
import re
import sys
import argparse
import yaml
import openai
import tiktoken
import random
import requests
import itertools
import uuid

import torch
import transformers
import numpy as np
from transformers import logging
logging.set_verbosity_error()

from fastchat.model import load_model, get_conversation_template

HERE = __file__
REPO_DIR = os.path.join(os.path.dirname(HERE), "../")

class APIModel:
    api_url: str = None 
    framework: str = None

    def __init__(self, model_path, api_url="http://localhost:8000/generate", framework="vllm"):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.api_url = api_url
        self.framework = framework

    def generate(self, input_ids, max_new_tokens, **kwargs):
        prompt = self.tokenizer.decode(input_ids[0])
        if self.framework == "vllm":
            headers = {"User-Agent": "Test Client"}
            pload = {
                "prompt": prompt,
                "n": 1,
                "use_beam_search": False,
                "temperature": 0.0,
                "max_tokens": max_new_tokens,
                "stream": False,
            }
            response = requests.post(self.api_url, headers=headers, json=pload, stream=False)
            text = json.loads(response.content)["text"]
            return self.tokenizer(text).input_ids
        elif self.framework == "lightllm":
            headers = {'Content-Type': 'application/json'}
            pload = {
                'inputs': prompt,
                "parameters": {
                    'do_sample': False,
                    'ignore_eos': False,
                    'max_new_tokens': max_new_tokens,
                }
            }
            response = requests.post(self.api_url, headers=headers, json=pload, stream=False)
            text = response.json()['generated_text'][0]
            return [input_ids[0].tolist() + self.tokenizer(text).input_ids]
        else:
            raise NotImplementedError

def maybe_monkey_patch(args):
    if "longchat" in args.model_name_or_path:
        from longchat.train.monkey_patch.llama_condense_monkey_patch import replace_llama_with_condense
        replace_llama_with_condense(args.longchat_ratio)

        if args.longchat_flash_attn:
            from longchat.train.monkey_patch.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
            replace_llama_attn_with_flash_attn()

    import transformers

def get_output_dir(args):
    path = args.model_name_or_path

    if path[-1] == "/":
        path = path[:-1]
    name = path.split("/")[-1]

    output_dir = f"evaluation/{args.task}/predictions/{name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"output to {output_dir}")
    return output_dir

def longeval_load_model(args):

    if args.framework is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
        return APIModel(args.model_name_or_path, framework=args.framework), tokenizer

    if "mosaicml/mpt-7b-storywriter" in args.model_name_or_path:
        # Adapt from: https://huggingface.co/mosaicml/mpt-7b-storywriter
        filter_string()
        config = transformers.AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        config.attn_config['attn_impl'] = 'triton'

        model = transformers.AutoModelForCausalLM.from_pretrained(
          args.model_name_or_path,
          config=config,
          torch_dtype=torch.bfloat16, # Load model weights in bfloat16
          trust_remote_code=True
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    elif "mosaicml/mpt-30b-chat" in args.model_name_or_path:
        config = transformers.AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            max_seq_len = 16384,
            device_map = "auto",
            max_memory= {i: f"{args.max_gpu_memory}GiB" for i in range(args.num_gpus)},
            torch_dtype=torch.float16
        )
        model.attn_impl = "triton"

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True, use_fast=True, model_max_length=16384
        )
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    elif "THUDM/chatglm2-6b" in args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = transformers.AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True).half().cuda()
        model = model.eval()
    elif "gpt-" in args.model_name_or_path:
        tokenizer = None
        model = None
    elif "claude" in args.model_name_or_path:
        tokenizer = None
        model = None
    else:
        # Use fastchat load_model API
        model, tokenizer = load_model(
            args.model_name_or_path,
            device="cuda",
            num_gpus=args.num_gpus,
            max_gpu_memory=f"{args.max_gpu_memory}GiB",
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )

    return model, tokenizer

def load_testcases(test_file):
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)

    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)

    return test_cases

def test_topics_one_sample(model, tokenizer, test_case, output_file, idx, args):
    prompt = test_case["prompt"]
    topics = test_case["topics"]
    
    if "mosaicml/mpt-7b-storywriter" in args.model_name_or_path:
        from transformers import pipeline
        pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')
        # Use next word prediction to get storywriter answer
        prompt += '\n ASSISTANT: The first topic is'
        prompt_length = len(tokenizer(prompt).input_ids)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = pipe(prompt, max_new_tokens=15, do_sample=True, use_cache=True)[0]['generated_text'][len(prompt):]
    elif "THUDM/chatglm2-6b" in args.model_name_or_path:
        prompt_length = len(tokenizer(prompt).input_ids)
        output, _ = model.chat(tokenizer, prompt, history=[], max_length=16384)
        output = [output]
    elif "gpt-" in args.model_name_or_path:
        prompt_length, output = retrieve_from_openai(prompt, args.model_name_or_path)
    elif "claude" in args.model_name_or_path:
        prompt_length, output = retrieve_from_anthropic(prompt, args.model_name_or_path)
    else:
        if "longchat" in args.model_name_or_path:
            conv = get_conversation_template("vicuna")
        else:
            conv = get_conversation_template(args.model_name_or_path)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input = tokenizer(prompt, return_tensors="pt")
        prompt_length = input.input_ids.size()[-1]
        
        # Disable use_cache if using longchat models with flash attention
        use_cache = not ("longchat" in args.model_name_or_path and args.longchat_flash_attn)

        device = getattr(model, "device", "cpu")

        output = model.generate(input.input_ids.to(device), max_new_tokens=50, use_cache=use_cache)[0]
        output = output[prompt_length:]
        output = tokenizer.batch_decode([output], skip_special_tokens=True)
    
    summary = f"Label: {topics[0]}, Predict: {output}, prompt length: {prompt_length}".replace('\n', ' ')
    print(summary)
    if idx ==0:
        with open(output_file, "w") as f:
            f.write(summary)
            f.write("\n")
    else:
        with open(output_file, "a+") as f:
            f.write(summary)
            f.write("\n")
    
    return None, prompt_length, summary

def test_lines_one_sample(model, tokenizer, test_case, output_file, idx, args):
    prompt = test_case["prompt"]
    correct_line = test_case["correct_line"]
    expected_number = test_case["expected_number"]

    if "mosaicml/mpt-7b-storywriter" in args.model_name_or_path:
        from transformers import pipeline
        pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')
        # Use next word prediction to get storywriter answer
        prompt += f'Line <{test_case["random_idx"][0]}>: <REGISTER_CONTENT> is'
        prompt_length = len(tokenizer(prompt).input_ids)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = pipe(prompt, max_new_tokens=15, do_sample=True, use_cache=True)[0]['generated_text'][len(prompt):]
    elif "THUDM/chatglm2-6b" in args.model_name_or_path:
        prompt_length = len(tokenizer(prompt).input_ids)
        output, _ = model.chat(tokenizer, prompt, history=[], max_length=16384)
    elif "gpt-" in args.model_name_or_path:
        prompt_length, output = retrieve_from_openai(prompt, args.model_name_or_path)
    elif "claude" in args.model_name_or_path:
        prompt_length, output = retrieve_from_anthropic(prompt, args.model_name_or_path)
    else:
        if "longchat" in args.model_name_or_path:
            conv = get_conversation_template("vicuna")
        else:
            conv = get_conversation_template(args.model_name_or_path)
        print(f"Using conversation template: {conv.name}")

        if "mosaicml/mpt-30b-chat" in args.model_name_or_path:
            prompt += f'Answer in the format <{test_case["random_idx"][0]}> <REGISTER_CONTENT>.'
        
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input = tokenizer(prompt, return_tensors="pt")
        prompt_length = input.input_ids.shape[-1]
        
        # Disable use_cache if using longchat models with flash attention
        use_cache = not ("longchat" in args.model_name_or_path and args.longchat_flash_attn)

        device = getattr(model, "device", "cpu")
        
        output = model.generate(input.input_ids.to(device), max_new_tokens=100, use_cache=use_cache)[0]
        output = output[prompt_length:]
        output = tokenizer.batch_decode([output], skip_special_tokens=True)[0]

    # Matching the last digit of the model output
    response_number = re.findall("\d+", output)
    if response_number is not None and len(response_number) > 0:
        response_number = int(response_number[-1])
    else:
        print(f"Got unparsable result")
        response_number = -1

    summary = f"Label: {expected_number}, Predict: {output}, Parsed: {response_number}, prompt length: {prompt_length}".replace('\n', ' ')
    print(summary)
    if idx ==0:
        with open(output_file, "w") as f:
            f.write(summary)
            f.write("\n")
    else:
        with open(output_file, "a+") as f:
            f.write(summary)
            f.write("\n")
    
    return expected_number == response_number, prompt_length, summary

def token_counter(model_name, prompt):
    if "gpt" in model_name:
        token_size = len(tiktoken.encoding_for_model(model_name).encode(prompt))
        print(f"Number of tokens: {token_size}")
    else:
        token_size = len(tiktoken.encoding_for_model(model_name).encode(prompt))
        print(f"Number of tokens: {token_size} by using gpt tokenizer as default")

    return token_size

def retrieve_from_openai(prompt, model_name, num_retries=10):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    token_size = len(tiktoken.encoding_for_model(model_name).encode(prompt))
    
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
        except openai.error.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
            if attempt == num_retries - 1:
                raise
        except openai.error.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            if attempt == num_retries - 1:
                raise
        except openai.error.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            if attempt == num_retries - 1:
                raise
        except openai.error.Timeout as e:
            print(f"OpenAI API request timed out: {e}")
            if attempt == num_retries - 1:
                raise
        except openai.error.InvalidRequestError as e:
            print(f"Invalid request to OpenAI API: {e}")
            if attempt == num_retries - 1:
                raise
        except openai.error.AuthenticationError as e:
            print(f"Authentication error with OpenAI API: {e}")
            if attempt == num_retries - 1:
                raise
        except openai.error.ServiceUnavailableError as e:
            print(f"OpenAI API service unavailable: {e}")
            if attempt == num_retries - 1:
                raise
        time.sleep(backoff)

    if completion is None:
        print(f"Failed to get response after {num_retries} retries")
        return token_size, -1, "Rate limit"

    response_line = completion.choices[0].message["content"]

    return token_size, response_line

def retrieve_from_anthropic(prompt, model_name, num_retries=10):
    import anthropic
    from anthropic import HUMAN_PROMPT, AI_PROMPT

    client = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])

    completion = client.completion(
        model = model_name,
        max_retries=num_retries,
        max_tokens_to_sample=300,
        temperature=0,
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}"
    )

    return -1, completion["completion"]


def filter_string():
    class FilteredStream:
        def __init__(self, original_stream, filter_string):
            self.original_stream = original_stream
            self.filter_string = filter_string

        def write(self, message):
            if self.filter_string not in message:
                self.original_stream.write(message)

        def flush(self):
            self.original_stream.flush()

    # Define the filter string to exclude specific content
    filter_string = "The model 'MPTForCausalLM' is not supported for text-generation. Supported models are ['BartForCausalLM', 'BertLMHeadModel', 'BertGenerationDecoder', 'BigBirdForCausalLM', 'BigBirdPegasusForCausalLM', 'BioGptForCausalLM', 'BlenderbotForCausalLM', 'BlenderbotSmallForCausalLM', 'BloomForCausalLM', 'CamembertForCausalLM', 'CodeGenForCausalLM', 'CpmAntForCausalLM', 'CTRLLMHeadModel', 'Data2VecTextForCausalLM', 'ElectraForCausalLM', 'ErnieForCausalLM', 'GitForCausalLM', 'GPT2LMHeadModel', 'GPT2LMHeadModel', 'GPTBigCodeForCausalLM', 'GPTNeoForCausalLM', 'GPTNeoXForCausalLM', 'GPTNeoXJapaneseForCausalLM', 'GPTJForCausalLM', 'LlamaForCausalLM', 'MarianForCausalLM', 'MBartForCausalLM', 'MegaForCausalLM', 'MegatronBertForCausalLM', 'MvpForCausalLM', 'OpenAIGPTLMHeadModel', 'OPTForCausalLM', 'PegasusForCausalLM', 'PLBartForCausalLM', 'ProphetNetForCausalLM', 'QDQBertLMHeadModel', 'ReformerModelWithLMHead', 'RemBertForCausalLM', 'RobertaForCausalLM', 'RobertaPreLayerNormForCausalLM', 'RoCBertForCausalLM', 'RoFormerForCausalLM', 'Speech2Text2ForCausalLM', 'TransfoXLLMHeadModel', 'TrOCRForCausalLM', 'XGLMForCausalLM', 'XLMWithLMHeadModel', 'XLMProphetNetForCausalLM', 'XLMRobertaForCausalLM', 'XLMRobertaXLForCausalLM', 'XLNetLMHeadModel', 'XmodForCausalLM']."

    # Create the filtered stream and replace sys.stdout with it
    filtered_stream = FilteredStream(sys.stdout, filter_string)
    sys.stdout = filtered_stream

def generate_topics_testcases(cfgs, output_dir):
    conv_list = []
    
    with open(os.path.join(REPO_DIR, "longeval/evaluation/topics/conversations.jsonl"), 'r') as json_file:
        conv_obj_list = list(json_file)

    for conv_obj in conv_obj_list:
        conv_obj = json.loads(conv_obj)
        conv_list.append(Conv(conv_obj["topic"], conv_obj["conversation"]))

    # generate prompts for each num_topics
    for num_topics in cfgs["num_topics"]:

        prompt_list = []
        
        for i in range(cfgs["num_test_samples"]):
            prompt = Prompt(i)
            indices = np.random.choice(list(range(len(conv_list))), size=num_topics, replace=False)
            for idx in indices:
                prompt.add_conv(conv_list[idx])
            prompt_list.append(prompt)
            
            prompt = None
        
        # write to output file
        avg_len = 0

        output_path = os.path.join(output_dir, f"{num_topics}_topics.jsonl")
        f = open(output_path, "w")
        for i, p in enumerate(prompt_list):
            pt = p.assemble_prompt()
            
            curr_output = {"test_id": p.id, 
                           "prompt": pt,
                           "topics": p.topic_list,
                           "prompt_length": -1}
            json.dump(curr_output, f)
            f.write("\n")
        f.close()

def generate_lines_testcases(cfgs, output_dir):
    for n in cfgs["num_lines"]:
        output_path = os.path.join(output_dir, f"{n}_lines.jsonl")
        f = open(output_path, "w")

        for i in range(cfgs["num_test_samples"]):          
            prompt_header = "Below is a record of lines I want you to remember. " + \
                            "Each line begins with 'line <line index>' and contains " + \
                            "a '<REGISTER_CONTENT>' at the end of the line as a numerical value. " + \
                            "For each line index, memorize its corresponding <REGISTER_CONTENT>. At " + \
                            "the end of the record, I will ask you to retrieve the corresponding " + \
                            "<REGISTER_CONTENT> of a certain line index. Now the record start:\n\n"
    
            lines = []

            if cfgs["line_idx_opt"] == "LRT":
                line_idxes = list(range(1, n + 1))
                lines.extend([f"line {i}: REGISTER_CONTENT is <{random.randint(1, 50000)}>\n" for i in line_idxes])
                random_idx = random.randint(1, n)
                random_num = random_idx - 1
            else:
                line_idxes = generate_line_index(n, cfgs["line_idx_opt"])
                lines.extend([f"line {i}: REGISTER_CONTENT is <{random.randint(1, 50000)}>\n" for i in line_idxes])
                random_num = random.randint(0, len(line_idxes)-1)
                random_idx = line_idxes[random_num]

            expected_number, correct_line = retrieve_expected(lines, random_num)
            lines.insert(0, f"{prompt_header}")
            lines.insert(len(lines), f"\nNow the record is over. Tell me what is the <REGISTER_CONTENT> in line {random_idx}? I need the number.")
            prompt = generate_prompt_from_lines(lines)

            output = {
                "random_idx": (random_idx, random_num), # this is the line to retrieve
                "expected_number": expected_number,
                "num_lines": n,
                "correct_line": correct_line,
                "prompt": prompt}

            json.dump(output, f)
            f.write("\n")
        f.close()

class Conv:
    """a single conversation on a topic"""

    def __init__(self, topic, content):
        self.topic = topic
        self.content = content

class Prompt:
    """the prompt used for testing, composed of multiple  """

    def __init__(self, id):
        self.id = id
        self.conv_list = []
        self.topic_list = []

    def add_conv(self, conv):
        self.conv_list.append(conv)
        self.topic_list.append(conv.topic)
    
    def assemble_prompt(self):
        record_prompt = "Below is a record of our previous conversation " + \
            f"on {len(self.topic_list)} different topics. You are the ASSISTANT, and " + \
            "I am the USER. At the beginning of each topic, the USER will say " + \
            "'I would like to discuss the topic of <TOPIC>'. Memorize each " + \
            "<TOPIC>. At the end of the record, I will ask you to retrieve the " + \
            "first topic. Now the record start. "
        
        for conv in self.conv_list:
            record_prompt += conv.content

        self.prompt = f"{record_prompt} Now " + \
            "the record ends. What is the first topic(s) we discussed? Only give " + \
            "me the topic name. Do not summarize yourself." 

        # self.prompt = "A chat between a curious user and an artificial intelligence " + \
        #     "assistant. The assistant gives helpful, detailed, and polite " + \
        #     f"answers to the user\'s questions. USER: {record_prompt} Now " + \
        #     f"the record ends. What is the {question_idx} topic(s) we discussed? Only give " + \
        #     "me the topic name(s) in the format of [<topic>, <topic>, ...]. Do not summarize yourself. Do not mention topic order. ASSISTANT:" 
        
        return self.prompt
    
def retrieve_cmd_args(): # setup program params from a given path to a yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_path', help='path to the yaml configuration')
    args = parser.parse_args()

    f = open(args.yaml_path, "r")
    cfgs = yaml.load(f, Loader=yaml.CLoader)

    print(yaml.dump(cfgs))

    return cfgs

def generate_line_index(num_line, idx_opt):
    if idx_opt == "LRT-ABCindex":
        ingredients = ["A", "B", "C", "D", "E", "F"]

        start = 6
        comb = list(itertools.product(ingredients, repeat=start))
        while len(comb) < num_line:
            start += 1
            comb = list(itertools.product(ingredients, repeat=start))
        
        comb = ["".join(i) for i in comb]

        return comb[:num_line]
    elif idx_opt == "LRT-UUID":
        comb = []
        for i in range(num_line):
            comb.append(str(uuid.uuid4()))
        
        return comb
    elif idx_opt == "LRT-NL":
        import wonderwords

        w = wonderwords.RandomWord()
        adjs = w.random_words(num_line, include_categories=["adjective"])
        nouns = w.random_words(num_line, include_categories=["noun"])

        comb = []
        for i, (adj, noun) in enumerate(zip(adjs, nouns)):
            comb.append(f"{adj}-{noun}")
        
        return comb
    
def retrieve_expected(lines, random_line_pos):
    correct_line = lines[random_line_pos]
    expected_number = re.search("<\d+>", correct_line)
    if expected_number is not None:
        expected_number = int(expected_number.group()[1:-1])
    else:
        print(f"Got unparsable line: {correct_line}")

    return expected_number, correct_line

def generate_prompt_from_lines(lines):
    prompt = ""
    for l in lines:
        prompt += l
    
    return prompt