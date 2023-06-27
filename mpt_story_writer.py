import torch
import transformers

name = 'mosaicml/mpt-7b-storywriter'

config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
config.attn_config['attn_impl'] = 'triton'
config.init_device = 'cuda:0' # For fast initialization directly on GPU!

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16, # Load model weights in bfloat16
  trust_remote_code=True
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

from transformers import pipeline
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')

import os
from utils import load_testcases

n_shot = 0
output_dir = "evaluation/topics/predictions"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_dir = os.path.join(output_dir, f'mpt-7b-storywriter-{n_shot}shot')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

print(f"output to {output_dir}")


num_map = {1: 'first', 2:'second', 3: 'third', 4: 'fourth', 5:'fifth', 6:'sixth', 7:'seventh', 8:'eighth'}
for i in [5, 10, 15, 20, 25]:
    test_file = f'evaluation/topics/testcases_clean/{i}_topics.jsonl'
    data = load_testcases(test_file)
    for sample in data:
        question = sample['prompt']
        few_shot_samples = ''
        for j in range(n_shot):
            few_shot_samples += f" For example, the {num_map[j+2]} topic is {sample['topics'][j+1]}. Then, what is the first topic?"
        question += few_shot_samples
        question += '\n ASSISTANT: The first topic is'
        print(few_shot_samples)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            answer = pipe(question, max_new_tokens=15, do_sample=True, use_cache=True)[0]['generated_text'][len(question):]
        out = f"Label: {sample['topics'][0]}, Predict: {answer}, --- INFO ---".replace('\n', ' ')
        print(out)
	
        output_file = os.path.join(output_dir, f"{i}_response.txt")
        with open(output_file, "a+") as f:
            f.write(out)
            f.write("\n")

