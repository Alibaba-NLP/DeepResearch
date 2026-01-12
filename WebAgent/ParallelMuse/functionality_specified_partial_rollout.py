import os
import re
import math
import copy
import json
import time
import json5
import random
import aiohttp
import asyncio
import datetime
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from openai import AsyncOpenAI
from collections import Counter
from transformers import AutoTokenizer

from tools.tool_search import Search
from tools.tool_visit import Visit


_rr_index = 0
AGENT_LLM_BASE_URL_POOL = [
    "http://localhost:8000/v1",
    # You can add more LLM API URLs here to balance the workload.
]
AGENT_LLM_API_KEY = 'EMPTY'

def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")

def read_jsonl(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    return result


SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Current date: """


def get_next_base_url():
    global _rr_index
    url = AGENT_LLM_BASE_URL_POOL[_rr_index % len(AGENT_LLM_BASE_URL_POOL)]
    _rr_index += 1

    return url


async def call_llm(sem, messages):
    max_tokens = 32*1024

    async with sem:
        for retry in range(10):
            AGENT_LLM_BASE_URL = get_next_base_url()

            client = AsyncOpenAI(
                api_key=AGENT_LLM_API_KEY,
                base_url=AGENT_LLM_BASE_URL,
            )
            try:
                response = await client.chat.completions.create(
                    model="", 
                    messages=messages, 
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=0.6,
                    top_p=0.95,
                    presence_penalty=1.1,
                    logprobs=True, 
                    top_logprobs=16,
                    max_tokens=max_tokens
                )
                result_text = response.choices[0].message.content
                response_logprobs = response.choices[0].logprobs.content

                result_tokens = []
                result_toplogprobs = []
                for item in response_logprobs:
                    result_tokens.append(item.token)
                    result_toplogprobs.append(item.top_logprobs)

                break

            except Exception as e:
                print(f"[Tongyi-DeepResearch async error] {e}")
                if "time out" not in str(e).lower():    
                    max_tokens = max_tokens / 2
        else:
            return None, None

        all_token_entropies = []  # [(token, entropy), ...]
        for tok, toplogprobs in zip(result_tokens, result_toplogprobs):
            logprob_values = np.array([tlp.logprob for tlp in toplogprobs], dtype=np.float64)
            
            probs = np.exp(logprob_values)
            probs = probs / probs.sum()

            entropy = -np.sum(probs * logprob_values)
            all_token_entropies.append((tok, entropy))

        def find_pattern_last_idx(pattern):
            last_index = None
            for i in range(len(result_tokens) - len(pattern) + 1):
                if result_tokens[i:i + len(pattern)] == pattern:
                    last_index = i + len(pattern) - 1
                    return last_index

        if "<tool_call>" in ''.join(result_tokens):
            think_start_pattern = ['<think>']
            think_end_pattern = ['</think>']
            tool_call_start_pattern = ['<tool_call>']
            tool_call_end_pattern = ['</tool_call>']

            think_start = find_pattern_last_idx(think_start_pattern)
            think_end = find_pattern_last_idx(think_end_pattern)
            tool_call_start = find_pattern_last_idx(tool_call_start_pattern)
            tool_call_end = find_pattern_last_idx(tool_call_end_pattern)

            entropies = [entropy for _, entropy in all_token_entropies]

            think_ppl = np.exp(np.mean(entropies[think_start+1: think_end])) if think_end else -1
            tool_call_ppl = np.exp(np.mean(entropies[tool_call_start+1: tool_call_end])) if tool_call_start and tool_call_end else -1
            all_ppl = np.exp(np.mean(entropies))

            step_ppl = {
                'think_ppl': think_ppl,
                'tool_call_ppl': tool_call_ppl,
                'all_ppl': all_ppl
            }
        else:
            step_ppl = {
                'think_ppl': -1,
                'tool_call_ppl': -1,
                'all_ppl': -1
            }

    return result_text, step_ppl


async def call_tool(sem, tool_name: str, tool_args: dict):
    if tool_name == "search":
        async with sem['search']:
            return await search.call(tool_args)
    elif tool_name == "visit":
        async with sem['visit']:
            return await visit.call(tool_args)
    else:
        await asyncio.sleep(1)
        return f'Tool {tool_name} does not exist.'


def count_tokens(messages):
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    return len(token_ids)


def get_initial_rollouts(question, existing_rollouts, initial_rollout_num):
    all_rollouts = [item for item in existing_rollouts if item['question'] == question]
    initial_rollouts = []
    for rollout in all_rollouts:
        if rollout['termination'] == 'answer':
            initial_rollouts.append(rollout)
        if len(initial_rollouts) == initial_rollout_num:
            break

    if len(initial_rollouts) != initial_rollout_num:
        for rollout in all_rollouts:
            if rollout not in initial_rollouts and rollout['termination'] != "max_length_exceeded" and rollout['termination'] != "llm_error_occurred":
                initial_rollouts.append(rollout)
            if len(initial_rollouts) == initial_rollout_num:
                break
    
    if len(initial_rollouts) != initial_rollout_num:
        for rollout in all_rollouts:
            if rollout not in initial_rollouts and rollout['termination'] != "llm_error_occurred":
                initial_rollouts.append(rollout)
            if len(initial_rollouts) == initial_rollout_num:
                break

    return initial_rollouts


async def rollout_single_traj(llm_sem, tool_sem, data, messages, args, max_turn_given=None, rollout_type='traj_level_rollout'):
    max_context_length = args.max_context_length
    max_turn = args.max_turn if not max_turn_given else max_turn_given
    max_turn = int(max_turn)

    question = data['question']
    answer = data['answer']
        
    record = copy.deepcopy(messages)

    termination = 'max_turn_exceeded'
    prediction = '[No Prediction]'

    llm_response_time = -1
    search_time = -1
    visit_time = -1

    for turn in range(max_turn):
        previous_turn_time_consumption = {
            'llm_response_time': llm_response_time,
            'search_time': search_time,
            'visit_time': visit_time
        }

        llm_response_time = -1
        search_time = -1
        visit_time = -1

        if count_tokens(record) > max_context_length:
            termination = 'max_length_exceeded'
            break
        
        tik = time.time()
        llm_response, step_ppl = await call_llm(llm_sem, record)
        toc = time.time()

        llm_response_time = toc - tik

        if llm_response is None:
            return {'question': question, 'answer': answer, 'rollout': record, 'termination': "llm_error_occurred", 'prepand_msg': messages, 'rollout_type': rollout_type}
        
        record.append({"role": "assistant", "content": llm_response, "previous_turn_time_consumption": previous_turn_time_consumption, 'step_ppl': step_ppl})

        if '<tool_call>' in llm_response and '</tool_call>' in llm_response:
            tool_call_str = llm_response.split('<tool_call>')[-1].split('</tool_call>')[0]

            try:
                tool_call = json5.loads(tool_call_str)
            
                tool_name = tool_call['name']
                tool_args = tool_call['arguments']

                tik = time.time()
                tool_response = await call_tool(tool_sem, tool_name, tool_args)
                toc = time.time()

                if tool_name == 'search':
                    search_time = toc - tik
                else:
                    visit_time = toc - tik

                print("======================================")
                print(f"Call `{tool_name}`: {tool_args}")
                print(f"Tool call {tool_name} invocation success with length {len(tool_response)}")
                print(tool_response)
            except Exception as e:
                tool_response = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                print(f"Tool call error {e}")

            record.append({"role": "user", "content": f"<tool_response>\n{tool_response}\n</tool_response>"})

        else:
            prediction = llm_response.strip()
            prediction = prediction.split("<answer>")[-1].split("</answer>")[0].strip()
            termination = 'answer'
            break
                
    return {'question': question, 'answer': answer, 'prediction': prediction, 'rollout': record, 'termination': termination, 'prepand_msg': messages, 'rollout_type': rollout_type}


def branch_high_uncertainty_steps(rollout, partial_sampling_topk, partial_sampling_mode):
    branch_step = []

    if partial_sampling_mode != "mixed_ppl":
        for i, msg in enumerate(rollout):
            if msg.get("step_ppl", None):
                branch_step.append({'step_id': i, 'step_ppl': msg['step_ppl'][partial_sampling_mode]})
        
        branch_step = sorted(branch_step, key=lambda x: x['step_ppl'], reverse=True)[:partial_sampling_topk]

    else:
        tool_call_sampling_topk = math.ceil(partial_sampling_topk / 2)
        think_sampling_topk = math.floor(partial_sampling_topk / 2)

        tool_call_branch_step = []
        think_branch_step = []

        for i, msg in enumerate(rollout):
            if msg.get("step_ppl", None):
                tool_call_branch_step.append({'step_id': i, 'step_ppl': msg['step_ppl']['tool_call_ppl']})
        
        tool_call_branch_step = sorted(tool_call_branch_step, key=lambda x: x['step_ppl'], reverse=True)[:tool_call_sampling_topk]

        for i, msg in enumerate(rollout):
            if msg.get("step_ppl", None):
                think_branch_step.append({'step_id': i, 'step_ppl': msg['step_ppl']['think_ppl']})
        
        think_branch_step = sorted(think_branch_step, key=lambda x: x['step_ppl'], reverse=True)[:think_sampling_topk]

        branch_step.extend(tool_call_branch_step)
        branch_step.extend(think_branch_step)

    return branch_step


async def main(args):
    llm_sem = asyncio.Semaphore(args.max_llm_workers)
    tool_sem = {
        'search': asyncio.Semaphore(args.max_search_workers),
        'visit': asyncio.Semaphore(args.max_visit_workers)
    }

    dataset = read_jsonl(args.qa_file_path)

    full_traj_rollout_output_file_path = os.path.join(args.output_dir, f"{args.qa_file_path.split('/')[-1].replace('.jsonl', '')}_{args.model_path.split('/')[-1]}_1_none_initial_rollout.jsonl")
    if args.partial_sampling_mode == 'none':
        output_file_path = full_traj_rollout_output_file_path
    else:
        output_file_path = os.path.join(args.output_dir, f"{args.qa_file_path.split('/')[-1].replace('.jsonl', '')}_{args.model_path.split('/')[-1]}_{args.initial_rollout_num}_{args.partial_sampling_mode}_{args.partial_sampling_topk}_{args.partial_sampling_rounds}_{args.partial_sampling_times_per_pos}.jsonl")

    visited_counter = Counter()
    if os.path.exists(full_traj_rollout_output_file_path):
        existing_rollouts = read_jsonl(full_traj_rollout_output_file_path)
        for visited_data in existing_rollouts:
            question = visited_data['question']
            visited_counter[question] += 1
            
    # continue partial rollout
    fully_visited_question = []
    visited_initial_rollouts = []
    if os.path.exists(output_file_path) and output_file_path != full_traj_rollout_output_file_path:
        if "browsecomp_en" in args.qa_file_path:
            initial_num = 200
        elif "browsecomp_zh" in args.qa_file_path:
            initial_num = 289
        elif "gaia" in args.qa_file_path:
            initial_num = 103
        elif "hle_search" in args.qa_file_path:
            initial_num = 157
        else:
            raise ValueError(f"Unknown dataset {args.qa_file_path}")

        visited_rollouts = read_jsonl(output_file_path)
        visited_initial_rollouts = visited_rollouts[:initial_num*args.initial_rollout_num]
        visited_rollouts = visited_rollouts[initial_num*args.initial_rollout_num:]

        visited_rollouts_counter = Counter()
        for visited_data in visited_rollouts:
            question = visited_data['question']
            visited_rollouts_counter[question] += 1

        fully_visited_question = [question for question, count in visited_rollouts_counter.items() if count == args.sampling_budget - args.initial_rollout_num]
        fully_visited_rollouts = [rollout for rollout in visited_rollouts if rollout['question'] in fully_visited_question]
        
        os.remove(output_file_path)
        with open(output_file_path, "a") as f:
            for r in visited_initial_rollouts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for r in fully_visited_rollouts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # submit task
    tasks = []
    if args.partial_sampling_mode == 'none':  # traj-level rollout
        pending_counter = Counter()
        for data in dataset:
            question = data['question']
            total_count = visited_counter[question] + pending_counter[question]
            need_to_submit = args.sampling_budget - total_count
            for _ in range(need_to_submit):
                cur_date = today_date()
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT + str(cur_date)},
                    {"role": "user", "content": question}
                ]
                tasks.append(rollout_single_traj(llm_sem, tool_sem, data, messages, args))
                pending_counter[question] += 1
                
    else:  # partial-sampling rollout
        partial_sampling_mode = args.partial_sampling_mode
        sampling_budget = args.sampling_budget
        max_turn = args.max_turn

        initial_rollout_num = args.initial_rollout_num
        partial_sampling_topk = args.partial_sampling_topk
        partial_sampling_rounds = args.partial_sampling_rounds
        partial_sampling_times_per_pos = args.partial_sampling_times_per_pos

        assert sampling_budget >= initial_rollout_num + initial_rollout_num * partial_sampling_topk * partial_sampling_rounds * partial_sampling_times_per_pos, "Sampling budget is not set correctly."
        try:
            assert all(visited_counter[data['question']] >= args.initial_rollout_num for data in dataset), "Initial rollouts are not sufficient."
        except:
            raise ValueError
        
        for i, data in tqdm(enumerate(dataset), total=len(dataset), desc="Detecting branching point ..."):
            question = data['question']

            if question in fully_visited_question:
                continue
            
            if not visited_initial_rollouts:
                initial_rollouts = get_initial_rollouts(question, existing_rollouts, initial_rollout_num)
                # save initial rollouts first
                with open(output_file_path, "a") as f:
                    for r in initial_rollouts:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                initial_rollouts = [v_initial_rollouts for v_initial_rollouts in visited_initial_rollouts if v_initial_rollouts['question'] == question]

            assert len(initial_rollouts) == initial_rollout_num, "Initial rollouts are not sufficient."
            
            for sampling_round in range(partial_sampling_rounds):  # TODO: multi-round partial rollout, now only support one round
                for r in initial_rollouts:
                    tmp_tasks_count = 1

                    partial_completion_count = None
                    branch_step = branch_high_uncertainty_steps(r['rollout'], partial_sampling_topk, partial_sampling_mode)
                    
                    if len(branch_step) != partial_sampling_topk:
                        if partial_completion_count is None:
                            partial_completion_count = 0

                        partial_completion_count += len(branch_step)
                    
                    tasks.extend(
                        [rollout_single_traj(
                            llm_sem, tool_sem, data, r['rollout'][:int(b['step_id'])],
                            args, max_turn-(int(b['step_id'])-2)/2,
                            'partial_rollout'
                        )
                        for b in branch_step
                        for sampling_times in range(partial_sampling_times_per_pos)]
                    )

                    tmp_tasks_count += len(branch_step) * partial_sampling_times_per_pos

                    sampling_budget_per_initial_rollout = int(sampling_budget / initial_rollout_num)
                    if sampling_budget_per_initial_rollout > 1 + partial_sampling_topk * partial_sampling_rounds * partial_sampling_times_per_pos:
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": question}
                        ]
                        if partial_completion_count is not None:
                            tasks.extend(
                                [rollout_single_traj(llm_sem, tool_sem, data, messages, args, None, 'traj_level_rollout')
                                for count in range(sampling_budget_per_initial_rollout-1-partial_completion_count*partial_sampling_rounds*partial_sampling_times_per_pos)]
                            )

                            tmp_tasks_count += sampling_budget_per_initial_rollout-1-partial_completion_count*partial_sampling_rounds*partial_sampling_times_per_pos

                        else:
                            tasks.extend(
                                [rollout_single_traj(llm_sem, tool_sem, data, messages, args, None, 'traj_level_rollout')
                                for count in range(sampling_budget_per_initial_rollout-1-partial_sampling_topk*partial_sampling_rounds*partial_sampling_times_per_pos)]
                            )

                            tmp_tasks_count += sampling_budget_per_initial_rollout-1-partial_sampling_topk*partial_sampling_rounds*partial_sampling_times_per_pos
                    
                    assert tmp_tasks_count == sampling_budget_per_initial_rollout, "Sampling budget mismatch."

    print(f"Total number of tasks: {len(tasks)}")

    # process task
    with open(output_file_path, "a") as f, open('./inference_error.txt', 'a') as log:
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
                log.write(f"[ERROR]: {error_message}" + "\n\n")
                log.flush()
                os.fsync(log.fileno())
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # basic info
    parser.add_argument("--qa_file_path", type=str, default="./data/browsecomp_200.jsonl")
    parser.add_argument("--output_dir", type=str, default="./deepresearch_results")
    parser.add_argument("--model_path", type=str, default="./llm_ckpt/tongyi-deepresearch-30b-a3b")

    # for rollout
    parser.add_argument("--max_llm_workers", type=int, default=32)
    parser.add_argument("--max_search_workers", type=int, default=32)
    parser.add_argument("--max_visit_workers", type=int, default=32)
    parser.add_argument("--sampling_budget", type=int, default=8)
    parser.add_argument("--max_turn", type=int, default=100)
    parser.add_argument("--max_context_length", type=int, default=128*1024)

    # for partial sampling
    parser.add_argument("--initial_rollout_num", type=int, default=1)
    parser.add_argument("--partial_sampling_mode", type=str, default="tool_call_ppl", choices=["none", "all_ppl", "think_ppl", "tool_call_ppl", "mixed_ppl"])
    parser.add_argument("--partial_sampling_topk", type=int, default=2)
    parser.add_argument("--partial_sampling_rounds", type=int, default=1)
    parser.add_argument("--partial_sampling_times_per_pos", type=int, default=3)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    search = Search()
    visit = Visit()

    asyncio.run(main(args))