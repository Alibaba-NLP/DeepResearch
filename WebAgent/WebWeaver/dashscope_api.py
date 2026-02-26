import requests
import json
import copy
import time
import random
import os


def call_dashscope(model, messages, stop, temperature, top_p, max_tokens, max_retries=10):
    body = {
        "messages": [
            {
                "role": "user",
                "content": "Hello!"
            }
        ],
        "extendParams":{
                "thinkingConfig": {
                "includeThoughts": True,
                # "thinkingBudget": 16000
            }
        },
        "model": "o4-mini-2025-04-16", ####gemini-2.5-pro, gpt-4o-2024-11-20, o4-mini-2025-04-16,
        "max_tokens": max_tokens,
    }
    
    url = os.getenv("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    api_key = os.getenv("DASHSCOPE_API_KEY", "sk-4deb7c21bc32424290dfb2dc127f4054")
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }

    max_try = 10

    body = copy.deepcopy(body)
    body["messages"] = messages
    body["model"] = model
    body["max_tokens"] = max_tokens
    body["stop"] = stop
    body["temperature"] = temperature
    body["top_p"] = top_p

    for i in range(max_try):
        try:
            r = requests.post(url, json.dumps(body, ensure_ascii=False).encode('utf-8'),
                              headers=headers, timeout=6000)
            
            response = json.loads(r.text)

            if response["choices"][0]["message"]["content"] is None:
                print(f"Response is None, retrying {i+1}/{max_try}")
                time.sleep(random.randint(1, 15) * 0.23)
                continue
            return response["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"Request failed: {e}")
            time.sleep(random.randint(1, 15) * 0.23)
            continue

if __name__ == '__main__':
    response = call_dashscope("qwen3-235b-a22b-instruct-2507", messages=[
        {
            "role": "user",
            "content": "Hello!",
        }],
        stop=["\n<tool_response>", "<tool_response>"],
        temperature=0.6,
        top_p=0.95,
        max_tokens=64000
    )

    print(response)

