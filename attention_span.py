import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_model, load_testcases, test, attention_span

torch.set_printoptions(profile="full")

path = "/data/dacheng/longchat_7b_4096"
test_file = "evaluation/topics/testcases/5_topics.jsonl"

model, tokenizer = load_model(path, device="cuda", num_gpus=8)

test_cases = load_testcases(test_file)

for i, test_case in enumerate(test_cases):
    prompt = test_case["prompt"]
    prompt_length = test_case["prompt_length"]
    topics = test_case["topics"]
    span, span_avg_layer, span_avg_all = attention_span(model, tokenizer, test_case)
    print(f"---------- Testcase {i} length {prompt_length} ----")
    print(span, span_avg_layer, span_avg_all)
    assert False
