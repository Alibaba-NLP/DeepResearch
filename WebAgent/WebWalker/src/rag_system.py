import asyncio
import json
import os
import aiohttp
import requests
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI, OpenAI
from openai.types.chat.chat_completion import Choice
from volcenginesdkarkruntime import Ark
from tqdm.asyncio import tqdm
from typing import Dict, Any
from datasets import load_dataset

# openai_env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
# gemini_env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
# ark_env
ARK_API_KEY = os.getenv("ARK_API_KEY")
ARK_MODEL = os.getenv("ARK_MODEL")
# moonshot_env
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
# baidu_env
BAIDU_API_KEY = os.getenv("BAIDU_API_KEY")
BAIDU_SECRET_KEY = os.getenv("BAIDU_SECRET_KEY")

def o1_api(ds, output_path):
    if OPENAI_BASE_URL is None or OPENAI_API_KEY is None:
        print("Please set OPENAI_API_KEY and OPENAI_BASE_URL environment variables.")
        return
    client = AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    model = "o1-preview-2024-09-12"
    MAX_CONCURRENT = 16

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(min=4, max=60))
    async def get_chat_completion(message, semaphore) -> str:
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": message["question"]}],
                    timeout=80
                )
                message["pred"] = response.choices[0].message.content
                return message
        except Exception as e:
            print(f"Error in get_chat_completion for message  {type(e).__name__} - {str(e)}")
            raise

    async def request_model(prompts):
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        tasks = [get_chat_completion(prompt, semaphore) for prompt in prompts]
        
        for future in tqdm.as_completed(tasks, total=len(tasks), desc="Processing prompts"):
            result = await future
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    visited = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            visited = [json.loads(line)["question"] for line in f]
    prompts = []
    for item in ds["question"]:
        if item not in visited:
            temp = {"question": item}
            prompts.append(temp)

    asyncio.run(request_model(prompts))

def gemini_api(ds, output_path, search=False):
    if GEMINI_API_KEY is None or GEMINI_BASE_URL is None:
        print("Please set GEMINI_API_URL and GEMINI_AUTH_TOKEN environment variables.")
        return
    headers = {
        'Authorization': f'Bearer {GEMINI_API_KEY}',
        'Content-Type': 'application/json'
    }
    MAX_CONCURRENT = 16

    async def fetch(session, url, headers, data, semaphore, query_text):
        async with semaphore:
            return query_text, await _fetch_with_retry(session, url, headers, data)

    @retry(stop=stop_after_attempt(10), wait=wait_exponential(min=4, max=60), reraise=True)
    async def _fetch_with_retry(session, url, headers, data):
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                print(f"Error: {response.status} - {response.reason}")
            response.raise_for_status()
            return await response.json()

    async def run_gemini_api():
        if not os.path.exists(output_path):
            open(output_path, "w").close()
        with open(output_path, "r", encoding="utf-8") as f:
            visited = [json.loads(line)["question"] for line in f]
        data_list = []
        for item in ds["question"]:
            if item not in visited:
                data_list.append(item)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        async with aiohttp.ClientSession() as session:
            if search:
                tasks = [
                    fetch(session, GEMINI_BASE_URL, headers, {
                        "model": "gemini-1.5-pro",
                        "contents": [
                            {"role": "user", 
                             "parts": 
                                [   
                                    {"text": query}, 
                                    {"tools": {
                                        "google_search_retrieval": {
                                            "dynamic_retrieval_config": {
                                                "mode": "MODE_DYNAMIC",
                                                "dynamic_threshold": 0
                                                } 
                                            }
                                        }
                                    }
                                ]
                            }
                        ]
                    }, semaphore, query)
                    for query in data_list
                ]
            else:
                tasks = [
                    fetch(session, GEMINI_BASE_URL, headers, {
                        "model": "gemini-1.5-pro",
                        "contents": [{"role": "user", "parts": [{"text": query}]}],
                        "candidates": 1
                    }, semaphore, query)
                    for query in data_list
                ]
            for future in tqdm.as_completed(tasks, total=len(tasks), desc="Processing queries"):
                query_text, result = await future
                adic = {"question": query_text, "pred": result["candidates"][0]["content"]["parts"][0]["text"]}
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(adic, ensure_ascii=False) + "\n")

    asyncio.run(run_gemini_api())

def doubao_api(ds, output_path):
    if ARK_API_KEY is None or ARK_MODEL is None:
        print("Please set ARK_API_KEY environment variable.")
        return
    client = Ark(base_url="https://ark.cn-beijing.volces.com/api/v3", api_key=ARK_API_KEY)
    MAX_WORKERS = 10

    def call(data):
        try:
            completion = client.bot_chat.completions.create(
                model=ARK_MODEL,
                messages=[
                    {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
                    {"role": "user", "content": data["question"]}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return

    if not os.path.exists(output_path):
        open(output_path, "w").close()
    visited = set()
    with open(output_path, "r", encoding="utf-8") as f:
        visited.update(json.loads(line)["question"] for line in f)
    data_list = []
    for item in ds["question"]:
        if item not in visited:
            temp = {"question": item}
            data_list.append(temp)

    with tqdm(total=len(data_list)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_data = {executor.submit(call, data): data for data in data_list}
            for future in concurrent.futures.as_completed(future_to_data):
                outputs = future.result()
                data = future_to_data[future]
                if outputs:
                    data["pred"] = outputs
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                pbar.update(1)

def kimi_api(ds, output_path):
    if MOONSHOT_API_KEY is None:
        print("Please set MOONSHOT_API_KEY environment variable.")
        return
    client = OpenAI(base_url="https://api.moonshot.cn/v1", api_key=MOONSHOT_API_KEY)
    MAX_WORKERS = 4

    def search_impl(arguments: Dict[str, Any]) -> Any:
        return arguments

    def chat(messages) -> Choice:
        try:
            completion = client.chat.completions.create(
                model="moonshot-v1-128k",
                messages=messages,
                temperature=0.3,
                tools=[{
                    "type": "builtin_function",
                    "function": {"name": "$web_search"}
                }]
            )
            return completion.choices[0]
        except Exception as e:
            print(f"Error: {e}")
            return

    if not os.path.exists(output_path):
        open(output_path, "w").close()
    
    visited = set()
    with open(output_path, "r", encoding="utf-8") as f:
        visited.update(json.loads(line)["question"] for line in f)

    data_list = []
    for item in ds["question"]:
        if item not in visited:
            temp = {"question": item}
            data_list.append(temp)
    def process_data(data):
        messages = [{"role": "system", "content": "你是 Kimi。"}, {"role": "user", "content": data["question"]}]
        finish_reason = None
        while finish_reason is None or finish_reason == "tool_calls":
            choice = chat(messages)
            finish_reason = choice.finish_reason
            if finish_reason == "tool_calls":
                messages.append(choice.message)
                for tool_call in choice.message.tool_calls:
                    tool_result = search_impl(json.loads(tool_call.function.arguments))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps(tool_result),
                    })
        data["pred"] = choice.message.content
        return data

    with tqdm(total=len(data_list)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_data = {executor.submit(process_data, data): data for data in data_list}
            for future in concurrent.futures.as_completed(future_to_data):
                processed_data = future.result()
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(processed_data, ensure_ascii=False) + "\n")
                pbar.update(1)

def wenxin_api(ds, output_path):
    if BAIDU_API_KEY is None or BAIDU_SECRET_KEY is None:
        print("Please set BAIDU_API_KEY and BAIDU_SECRET_KEY environment variables.")
        return
    
    def get_access_token():
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={BAIDU_API_KEY}&client_secret={BAIDU_SECRET_KEY}"
        response = requests.post(url, headers={'Content-Type': 'application/json'})
        return response.json().get("access_token")

    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()

    def call(data):
        payload = json.dumps({
            "messages": [{"role": "user", "content": data["question"]}]
        })
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=payload)
        return response.json()["result"]

    if not os.path.exists(output_path):
        open(output_path, "w").close()
    
    visited = set()
    with open(output_path, "r", encoding="utf-8") as f:
        visited.update(json.loads(line)["question"] for line in f)
    
    data_list = []
    for item in ds["question"]:
        if item not in visited:
            temp = {"question": item}
            data_list.append(temp)

    with tqdm(total=len(data_list)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_data = {executor.submit(call, data): data for data in data_list}
            for future in concurrent.futures.as_completed(future_to_data):
                outputs = future.result(timeout=10)
                data = future_to_data[future]
                data["pred"] = outputs
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                pbar.update(1)

def main(api_name, output_path):
    ds = load_dataset("callanwu/WebWalkerQA", split="main")
    api_functions = {
        "o1_api": o1_api,
        "gemini_api": gemini_api,
        "gemini_search_api": gemini_api,
        "doubao_api": doubao_api,
        "kimi_api": kimi_api,
        "wenxin_api": wenxin_api,
    }
    if api_name == "all":
        for api in api_functions:
            print(api)
            print(output_path + "/" + api+"_result.jsonl")
            os.makedirs(output_path, exist_ok=True)
            if api != "gemini_search_api":
                api_functions[api](ds, output_path + "/" + api+"_result.jsonl")
            else:
                api_functions[api](ds, output_path + "/" + api+"_result.jsonl", search=True)
    else:
        if api_name in api_functions:
            if api_name == "gemini_search_api":
                gemini_api(ds, output_path, search=True)
            asyncio.run(api_functions[api_name](ds, output_path))
        else:
            print(f"API {api_name} is not supported.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run different API models.")
    parser.add_argument("--api_name", type=str, help="Name of the API to run.")
    parser.add_argument("--output_path", type=str, help="Path to the output file. If api_name is 'all', this should be a directory. If api_name is others, this should be a file.")
    args = parser.parse_args()
    main(args.api_name, args.output_path)