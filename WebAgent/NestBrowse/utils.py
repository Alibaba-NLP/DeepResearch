import os
import ast
import json
import random
import aiohttp
from openai import AsyncOpenAI


async def call_llm(sem, prompt, max_tokens, model_name, client=None, mode='agent'):
    if mode == 'agent':
        LLM_API_KEY = os.getenv('AGENT_LLM_API_KEY')
        LLM_BASE_URL = os.getenv('AGENT_LLM_BASE_URL')

    elif mode == 'summary':
        LLM_API_KEY = os.getenv('SUMMARY_LLM_API_KEY', os.getenv('AGENT_LLM_API_KEY'))
        LLM_BASE_URL = os.getenv('SUMMARY_LLM_BASE_URL', os.getenv('AGENT_LLM_BASE_URL'))

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    try:
        LLM_BASE_URL = random.choice(ast.literal_eval(LLM_BASE_URL))
    except:
        pass

    async with sem['llm']:
        for retry in range(10):
            max_tokens = int(max_tokens)
            try:
                assert isinstance(prompt, list), "For nest_browse, prompt must be a list of messages"

                client = AsyncOpenAI(
                    api_key=LLM_API_KEY,
                    base_url=LLM_BASE_URL,
                )
                if mode == 'agent':
                    response = await client.chat.completions.create(
                        model="", 
                        messages=prompt, 
                        stop=["\n<tool_response>", "<tool_response>"],
                        temperature=0.6,
                        top_p=0.95,
                        presence_penalty=1.1,
                        max_tokens=max_tokens
                    )
                else:
                    response = await client.chat.completions.create(
                        model="", 
                        messages=prompt, 
                        temperature=0.6,
                        top_p=0.95,
                        max_tokens=max_tokens
                    )
                result_text = response.choices[0].message.content
                
                return result_text

            except Exception as e:
                print(f"[CALL LLM async error] {e}")
                if "time out" not in str(e).lower():    
                    max_tokens = max_tokens / 2

    return None


def read_jsonl(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    return result


def count_tokens(text, tokenizer):
    if isinstance(text, str):
        return len(tokenizer.encode(text))

    tokens = tokenizer.apply_chat_template(text, tokenize=True)
    return len(tokens)