import argparse
import os
import json
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import pipeline

from fastchat.model import get_conversation_template
from longeval.utils import maybe_monkey_patch, get_output_dir, longeval_load_model, load_testcases, test_topics_one_sample, test_lines_one_sample



class Pipeline:
    def __init__(self, model, tokenizer, args):
        self.args = args
        if args.model_name_or_path == "mosaicml/mpt-7b-storywriter":
            self.pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')  #TODO: hardcoded it to truncate the inputs in the pipeline
        else:
            self.model = model
            self.tokenizer = tokenizer

    def generate(self, prompt):
        if self.args.model_name_or_path == "mosaicml/mpt-7b-storywriter":
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = pipe(prompt, max_new_tokens=self.args.max_new_tokens, do_sample=True, use_cache=True)[0]['generated_text'][len(prompt):]
        else:
            if "longchat" in args.model_name_or_path:
                conv = get_conversation_template("vicuna")
            else:
                conv = get_conversation_template(self.args.model_name_or_path)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            inputs = self.tokenizer(prompt, return_tensors="pt")
            prompt_length = inputs.input_ids.size()[-1]

            use_cache = not ("longchat" in self.args.model_name_or_path and self.args.longchat_flash_attn)
            output = self.model.generate(inputs.input_ids.to(self.model.device), max_new_tokens=self.args.max_new_tokens, use_cache=use_cache)[0]
            output = output[prompt_length:]
            output = tokenizer.batch_decode([output], skip_special_tokens=True)[0]
        return output

    def mmlu_generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.args.max_seq_len).to('cuda')
        outputs = self.model.generate(**inputs, max_length=512, do_sample=False)[0]
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[length:], skip_special_tokens=True)

    def get_choice(self, text, **kwargs):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            predictions = self.model(
                **inputs,
                **kwargs,
            ).logits[0, -1]
        A_index = self.tokenizer("A", add_special_tokens=False).input_ids[0]
        B_index = self.tokenizer("B", add_special_tokens=False).input_ids[0]
        A = float(predictions[A_index].cpu())
        B = float(predictions[B_index].cpu())
        return A, B
