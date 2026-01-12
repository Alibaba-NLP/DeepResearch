import re
import os
import json
import copy
import random
import asyncio
import traceback
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer

from prompts import *
from toolkit.tool_search import Search
from toolkit.mcp_client import mcp_client
from toolkit.browser import Visit, Click, Fill
from utils import read_jsonl, count_tokens, call_llm


async def call_tool(sem, tool_name: str, tool_args: dict, client, lock):
    global tokenizer
    async with sem['tool']:
        if tool_name == "search":
            return await search.call(tool_args)
        elif tool_name == "visit":
            return await visit.call(tool_args, client=client, lock=lock, tokenizer=tokenizer, sem=sem)
        elif tool_name == "click":
            return await click.call(tool_args, client=client, lock=lock, tokenizer=tokenizer, sem=sem)
        elif tool_name == "fill":
            return await fill.call(tool_args, client=client, lock=lock)
        else:
            await asyncio.sleep(1)
            return f'Tool {tool_name} does not exist.'


async def agentic_loop(sem, data, messages):
    global tokenizer
    question = data['question']
    answer = data['answer']

    record = copy.deepcopy(messages)
    summary_record = []

    termination = 'max_turn_exceeded'
    prediction = '[No Prediction]'

    async with sem['session']:
        async with mcp_client(server_url=BROWSER_SERVER_URL) as (client, lock):
            for turn in range(MAX_AGENT_TURN):
                if count_tokens(record, tokenizer) > MAX_AGENT_LEN:
                    termination = 'max_length_exceeded'
                    break
                
                response = await call_llm(sem, record, int(os.getenv("MAX_SINGLE_GEN_TOKENS")), os.getenv("MODEL_NAME"))
                
                if not response:
                    return {'question': question, 'answer': answer, 'prediction': prediction, 'messages': record, 'summary_record': summary_record, 'termination': 'llm_response_error'}

                record.append({"role": "assistant", "content": response})

                if "<tool_call>" in response and "</tool_call>" in response:
                    cur_summary_record = None
                    tool_call = response.split('<tool_call>')[-1].split('</tool_call>')[0].strip()

                    try:
                        tool_call = json.loads(tool_call)

                        tool_name = tool_call['name']
                        tool_args = tool_call['arguments']
                        
                        if isinstance(tool_args, str):
                            tool_args = json.loads(tool_args)

                        print("========================================================")
                        print(f"Call tool {tool_name}, args: {tool_args}")

                        result = await call_tool(sem, tool_name, tool_args, client, lock)

                        if isinstance(result, tuple):
                            observation, cur_summary_record = result
                        elif isinstance(result, str):
                            observation = result
                        else:
                            raise Exception(f"Invalid tool result format: {result}")

                        if cur_summary_record:
                            summary_record.extend(cur_summary_record)

                        print("========================================================")
                        print(f"Call `{tool_name}`: {tool_args}")
                        print(f"Tool call {tool_name} invocation success with length {len(observation)}")
                        print(observation)
                    
                    except Exception as e:
                        observation = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                        print(f"Tool call error {str(e)}")

                    tool_response = f"<tool_response>\n{observation}\n</tool_response>"

                    if "server-side error" in observation:
                        return {'question': question, 'answer': answer, 'prediction': prediction, 'messages': record, 'summary_record': summary_record, 'termination': 'server_side_error'}

                    record.append({"role": "user", "content": tool_response, "tool_name": tool_name, "tool_args": tool_args, "function_result": observation})

                else:
                    if "<answer>" in response and "</answer>" in response:
                        prediction = response.split('<answer>')[-1].split('</answer>')[0].strip()
                        termination = 'answer'
                    else:
                        termination = 'llm_response_error'
                    
                    break

    return {'question': question, 'answer': answer, 'prediction': prediction, 'messages': record, 'summary_record': summary_record, 'termination': termination}
    

async def main(sem, rollout_count, input_path, output_path):
    global tokenizer
    dataset = read_jsonl(input_path)
    
    visited_counter = Counter()
    if os.path.exists(output_path):
        existing_rollouts = read_jsonl(output_path)
        for visited_data in existing_rollouts:
            question = visited_data['question']
            visited_counter[question] += 1

    # submit task
    tasks = []
    pending_counter = Counter()
    for data in dataset:
        question = data.get('question')
        total_count = visited_counter[question] + pending_counter[question]
        need_to_submit = rollout_count - total_count if rollout_count - total_count > 0 else 0
        for _ in range(need_to_submit):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_OURS},
                {"role": "user", "content": question}
            ]
            tasks.append(agentic_loop(sem, data, messages))
            pending_counter[question] += 1

    print(f"Total number of tasks: {len(tasks)}")

    # process task
    with open(output_path, "a") as f:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"No Blocking Rollout ..."):
            try:
                result = await future
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            except Exception as e:
                exception_type = type(e).__name__
                exception_message = str(e)
                traceback_info = ''.join(traceback.format_tb(e.__traceback__))
                error_message = f'{exception_type}: {exception_message}\n' \
                                f'Traceback:\n{traceback_info}'
                print(f"[ERROR]: {error_message}")


if __name__ == '__main__':
    BROWSER_SERVER_URL = "[YOUR-BROWSER-MCP-SERVER-URL]"
    
    AGENT_LLM_BASE_URL = "http://localhost:8000/v1"  # locally hosted nestbrowse model
    AGENT_LLM_API_KEY = "EMPTY"

    tokenizer = AutoTokenizer.from_pretrained("[TOKENIZER-PATH]")

    # ========================================
    rollout_count = 1
    MAX_AGENT_TURN = 100
    MAX_AGENT_LEN = 128 * 1024
    MAX_SINGLE_GEN_TOKENS = 32 * 1024
    MAX_SUMMARY_SHARD_LEN = 64 * 1024
    benchmark_name = "[BENCHMARK-NAME]"
    MODEL_NAME = "[CUSTOMIZED-MODEL-NAME]"
    MAX_WORKERS = 16
    sem = {
        'session': asyncio.Semaphore(MAX_WORKERS),
        'llm': asyncio.Semaphore(MAX_WORKERS),
        'tool': asyncio.Semaphore(MAX_WORKERS),
    }
    # ========================================


    os.environ["AGENT_LLM_BASE_URL"] = AGENT_LLM_BASE_URL
    os.environ["AGENT_LLM_API_KEY"] = AGENT_LLM_API_KEY
    os.environ["MAX_SINGLE_GEN_TOKENS"] = str(MAX_SINGLE_GEN_TOKENS)
    os.environ["MAX_SUMMARY_SHARD_LEN"] = str(MAX_SUMMARY_SHARD_LEN)
    os.environ["MODEL_NAME"] = MODEL_NAME


    input_path = f"./data/{benchmark_name}.jsonl"
    output_path = f"./results/{MODEL_NAME}_results_{benchmark_name}.jsonl"

    search = Search()
    visit = Visit()
    click = Click()
    fill = Fill()
    
    TOOLS_SCHEMA = [search.tool_schema, visit.tool_schema, click.tool_schema, fill.tool_schema]

    asyncio.run(main(sem, rollout_count, input_path, output_path))