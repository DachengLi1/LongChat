import json
import time
import os
import re
import sys

import torch
import transformers
from transformers import logging
logging.set_verbosity_error()

import openai
import tiktoken

from fastchat.model import load_model, get_conversation_template

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

        output = model.generate(input.input_ids.to(model.device), max_new_tokens=50, use_cache=use_cache)[0]
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
        
        output = model.generate(input.input_ids.to(model.device), max_new_tokens=100, use_cache=use_cache)[0]
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
