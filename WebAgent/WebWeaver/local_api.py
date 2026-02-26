
from openai import OpenAI
import jinja2
import datetime
import json
import requests
import os
import sys
sys.path.append(os.path.dirname(__file__))

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
import os
import traceback
from tqdm import tqdm
import time, random

# if os.environ.get("INFER_MODEL"):
#     url = "http://localhost:6001/v1/completions"
# else:
# url = "http://localhost:6002/v1/completions"
# api_key = "EMPTY"
# headers = {"Content-Type": "application/json", "Authorization": f"{api_key}"}

# template_file_path = "tool/gpt_oss_chat_template.jinja"

# def strftime_now_function(fmt):
#     return datetime.datetime.now().strftime(fmt)

# manually set the prompt template
# with open(template_file_path, 'r', encoding='utf-8') as f:
#     template_string = f.read()
# env = jinja2.Environment()
# 这个oss自带的jinja模版里要用到当前时间到函数
# env.globals['strftime_now'] = strftime_now_function
# env.filters['tojson'] = json.dumps
# template = env.from_string(template_string)


# def get_stream_response(url, headers, payload, print_stream=True):
#     full_response = ""

    
#     with requests.post(url, headers=headers, data=json.dumps(payload), stream=True) as response:
#         response.raise_for_status()
        
#         for chunk in response.iter_lines():
#             if chunk:
#                 decoded_line = chunk.decode('utf-8')

#                 if decoded_line.startswith('data: '):
#                     json_str = decoded_line[6:].strip()
#                 elif decoded_line.startswith('data:'):
#                     json_str = decoded_line[5:].strip()
#                 else:
#                     continue

#                 if not json_str or json_str == '[DONE]':
#                     continue

#                 try:
#                     data = json.loads(json_str)
                    
#                     text_chunk = data['choices'][0]['text']
#                     full_response += text_chunk
#                     if print_stream:
#                         print(text_chunk, end='', flush=True)

#                 except (json.JSONDecodeError, KeyError, IndexError) as e:
#                     pass
#     if print_stream:
#         print()
#     return full_response

def call_local_server_chat(msgs, stop, temperature, top_p, max_tokens, max_retries=10): 

    url_llm = "http://localhost:6001/v1/chat/completions"
    Authorization = "EMPTY"

    response = None
    for i in range(max_retries):
        # port = random.choice([6001, 6002, 6003, 6004])
        # url_llm = f"http://dlc1ge7wg8ufh7n4-master-0:{port}/v1/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': Authorization
        }
        payload = json.dumps({
            # "model": "Qwen3-235B-A22B-Instruct-2507",  ###
            # "model": "Qwen2.5-32b-Instruct",
            # "model": "gpt-oss-120b",
            "model": os.environ.get("INFER_MODEL_PATH"),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,   ## 16000
            "presence_penalty": 1.5,
            "messages": msgs,
            # "chat_template_kwargs": {
            #     "enable_thinking": False
            # }
        })
        try:
            response = requests.request("POST", url_llm, headers=headers, data=payload, timeout=240)
            data = response.json()
            if "error" in data and data["error"]["message"] == "Provider returned error":
                print("error in message")
                return ""
            
            content = data["choices"][0]["message"]["content"].replace("```json", "").replace("```", "")
            # try:
            #     content = json.loads(content)
            # except: 
            #     # left_pos, right_pos = content.index("{"), content.rindex("}")
            #     left_pos = content.find("{")
            #     right_pos = content.rfind("}")
            #     if left_pos > 0 and right_pos > left_pos:
            #         content = content[left_pos:right_pos+1]

            return content
        except Exception as e:
            print("Visit Tool Error", e)
            print("Use another EAS Service.")
            print(response.text)
    return ""

# def call_llm_gpt_oss(payload, print_stream=False):

    
#     max_try = 10
#     for i in range(max_try):

#         try:
        
#             response = get_stream_response(url, headers, payload, print_stream)
#             return response
#         # prompt += response

#         # if response.strip().endswith("<|call|>"):
#         #     tool_response, tool_name = execute_tool(response)
#         #     tool_count += 1
#         #     tool_content = f"<|start|>functions.{tool_name} to=assistant<|channel|>commentary<|message|>{tool_response}<|end|>"
#         #     prompt += tool_content
#         #     print(tool_content)

#         #     if tool_count >= args.tool_count_max:
#         #         prompt += "<|start|>assistant<|channel|>final<|message|>I have used too many tools, so I will conclude my answer."
#         #         print("<|start|>assistant<|channel|>final<|message|>I have used too many tools, so I will conclude my answer.")
#         #         if tool_count >= (args.tool_count_max+1):
#         #             break
#         # else:
#         #     break
#         except Exception as e:
#             print(f"Request failed: {e}")
#             time.sleep(random.randint(1, 15) * 0.23)
#             continue
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--tool_count_max", type=int, default=30)
    parser.add_argument("--max_worker", type=int, default=20)
    parser.add_argument("--reasoning_effort", type=str, default="high")
    parser.add_argument("--dataset_names", nargs='+', default=["browsecomp_en_small"])
    parser.add_argument("--print_stream", action="store_true")
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()


    question = "when is NeurIPS 2025?"

    print(f"The question is:\n{question}\n\n================\n")

    item = {
        "question": question
    }
    developer_prompt = """Cleverly leverage appropriate tools assist question answering."""
    messages = [
        {"role": "developer", "content": developer_prompt},
        {"role": "user", "content": question}
    ]

    tool_count = 0

    tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Array of query strings. Include multiple complementary search queries in a single call."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Visit webpage(s) and return the summary of the content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
                    },
                    "goal": {
                        "type": "string",
                        "description": "The specific information goal for visiting webpage(s)."
                    }
                },
                "required": ["url", "goal"]
            }
        }
    }
]
    prompt = template.render(messages=messages, reasoning_effort=args.reasoning_effort, tools=tools, add_generation_prompt=True)

    payload = {
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "stream": True,
            "skip_special_tokens": False,
            "stop": ["<|call|>"],
            "include_stop_str_in_output": True,
        }

    print(call_llm_gpt_oss(payload))

    