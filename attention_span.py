import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import load_model, load_testcases, test, attention_span

path = "/data/dacheng/longchat_7b_4096"
test_file = "evaluation/topics/testcases/3_topics.jsonl"

model, tokenizer = load_model(path)

test_cases = load_testcases(test_file)

for i, test_case in enumerate(test_cases):
    span, span_avg_layer, span_avg_all = attention_span(model, tokenizer, testcase)
    print(f"---------- Testcase {i} --------")
    print(span, span_avg_layer, span_avg_all)
    assert False
