import openai
import json

test_file = "evaluation/topics/testcases/2_topics.jsonl"
with open(test_file, 'r') as json_file:
    json_list = list(json_file)

first_test = json_list[0]
result = json.loads(first_test)
user_prompt = result["prompt"]
print(user_prompt)

assert False
response = openai.ChatCompletion.create(
    #model="gpt-4",
    model="gpt-3.5-turbo",
    messages=[
            {
            "role": "user",
            "content": user_prompt,
            },
            ],
            temperature=0.2,  # TODO: figure out which temperature is best for evaluation
            max_tokens=50,
            )
content = response["choices"][0]["message"]["content"]
print(content)
