# some code taken from Auto-GPT
import openai
import os
import re
import tiktoken
import time


def retrieve_from_openai(prompt, model_name):
    token_size = len(tiktoken.encoding_for_model(model_name).encode(prompt))
    print(f"Number of tokens: {token_size}")

    os.environ['OPENAI_API_KEY'] = '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'
    openai.api_key = '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'

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
        return token_size, -1, "Rate limit"

    response_line = completion.choices[0].message["content"]
    response_number = re.findall("\d+", response_line)
    if response_number is not None:
        response_number = int(response_number[-1])
    else:
        print(f"Got unparsable result: {response_line}")
    
    return token_size, response_number, response_line
