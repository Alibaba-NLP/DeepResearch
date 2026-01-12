import re
import os
import json
import time
import random
import asyncio
import datetime
import traceback
from tqdm import tqdm
from openai import AsyncOpenAI
from collections import defaultdict


# tongyi-deepresearch-30b-a3b
REPORT_CONVERGE_BASE_URL_POOL = [
    "http://localhost:8000/v1",
    # You can add more LLM API URLs here to balance the workload.
]
REPORT_CONVERGE_API_KEY = "EMPTY"


CONVERGE_SYSTEM_PROMPT = """Follow the user's instructions and present your answer in the format they request.
""".strip()


REPORT_PROMPT = """Given the following problem-solving trajectory:
{traj}

Your task is to distill this trajectory into a concise yet sufficiently detailed problem-solving report. 
You must only use the information contained within the provided trajectory — no additional or external information is allowed.

The report must include:

1. **Solution Planning**: Identify how the main problem is decomposed into subproblems, and describe the sequence and dependency relationships among those subproblems.
2. **Solution Methods**: For each subproblem, indicate which tools were invoked to solve it, the parameters used in those tool calls, and any resulting partial answers that contributed directly or indirectly to progress toward the final answer.  
   _Do not repeat the full output of the tools; only include the specific fragments of tool results that were essential in deriving the subanswers._
3. **Final Reasoning**: Clearly outline the reasoning process by which the subproblems and their associated subanswers led to the derivation of the final answer.

Additional requirements:
- The report must remain **concise** and **focused**.  
- Remove any content unrelated to problem-solving or any ineffective tool calls.  
- Ensure the final report has clear logical structure, with each step traceable and analyzable.

Finally, present the complete report in **Markdown format**, and wrap the entire report content within <report> </report> tags.
""".strip()


INTEGRATE_PROMPT = """You are tasked with solving the question: {question}.

Multiple independent teams have provided detailed process reports describing their approaches to solving this problem. As the final analyst, your role is to consolidate these reports, carefully examine the problem-solving methods they contain, and identify the key information obtained in each.

Your goal is to produce a final answer that perfectly resolves the question. Note that some of the reports may contain inconsistencies — you must critically evaluate which report(s) are reasonable and trustworthy.

If multiple reports reach the same conclusion, this increases the likelihood that the conclusion is correct; however, this is not guaranteed. You must still carefully verify and reflect to ensure that the final selected answer is truly the most accurate possible.

Wrap your final answer in <answer> </answer> tags.

Important:
- Every question has a definitive, certain answer.  
- You are not allowed to decline answering on the grounds of uncertainty.
- For any report that does not provide a clear and definite final answer, its confidence level should be significantly reduced.
- You must ultimately select one report as having the most correct answer.
- You are not allowed to call or use any external tools for verification. You must rely solely on the information already provided, conduct in-depth analysis, and then produce the final answer.
- You are not allowed to merge multiple different answers, nor are you allowed to produce an overly broad answer that attempts to encompass all candidate answers — such an answer should be eliminated first.
- You do not need to restate or summarize the reports; instead, provide a short-form answer that directly answers the question.

Below is the content of these reports:
""".strip()


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


async def get_llm_response(messages, max_tokens):
    client = AsyncOpenAI(
        base_url=random.choice(REPORT_CONVERGE_BASE_URL_POOL),
        api_key=REPORT_CONVERGE_API_KEY
    )
    try:
        response = await client.chat.completions.create(
            model="",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
            presence_penalty=1.1,
        )
        response_content = response.choices[0].message.content
        return response_content
    except Exception as e:
        print(f"[REPORT CONVERGE CLIENT ERROR]: {e}")
        await asyncio.sleep(2)

        if "time out" in str(e):
            return "[Request time out]"

    return "[Error getting visit response]"


def read_jsonl(file_path):
    result = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    return result


def write_jsonl(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def cluster_by_question(dataset):
    cluster = defaultdict(list)
    for item in dataset:
        cluster[item['question']].append(item)
    return list(cluster.values())


def construct_interaction_from_record(record):
    interaction = ''

    for r in record:
        # tool call
        if r['role'] == 'assistant' and "<tool_call>" in r['content'] and "</tool_call>" in r['content']:
            raw_content = r['content']
            thinking = raw_content.split("<think>")[-1].split("</think>")[0].strip()
            tool_call = raw_content.split("<tool_call>")[-1].split("</tool_call>")[0].strip()

            interaction += f"**{r['role']}:**\n*Thinking:* {thinking}\n*Tool Call:* {tool_call}\n\n"

        # tool response
        elif r['role'] == 'user' and "<tool_response>" in r['content'] and "</tool_response>" in r['content']:
            tool_response = r['content'].split("<tool_response>")[-1].split("</tool_response>")[0].strip()
            interaction += f"**Tool Response:**\n{tool_response}\n\n"
        
        # last response
        elif r['role'] == 'assistant' and "<tool_call>" not in r['content'] and "</tool_call>" not in r['content']:
            raw_content = r['content']
            thinking = raw_content.split("<think>")[-1].split("</think>")[0].strip()
            prediction = raw_content.split("</think>")[-1].split('<answer>')[-1].split('</answer>')[0].strip()

            if thinking == prediction:
                prediction = "[No Prediction]"

            interaction += f"**{r['role']}:**\n*Thinking:* {thinking}\n*Answer:* {prediction}\n\n"

        # system + initial user input
        else:
            interaction += f"**{r['role']}:**\n{r['content']}\n\n"

    return interaction.strip()


async def call_state_report(sem, traj, max_retries=10):
    max_tokens = 32 * 1024
    interaction = construct_interaction_from_record(traj['rollout'])
    user_input = REPORT_PROMPT.format(traj=interaction)

    if traj['prediction'] == '[No Prediction]':
        return "[Error getting state report]"

    async with sem:
        for retry in range(max_retries):
            try:
                messages = [
                    {'role': 'system', 'content': CONVERGE_SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_input}
                ]

                response = await get_llm_response(messages, max_tokens, data_path)
                response = response.split('</think>')[-1].strip()
                if "Error getting visit response" in response:
                    raise Exception(response)
                else:
                    break
            except Exception as e:
                await asyncio.sleep(2)
                if "time out" not in str(e).lower():    
                    max_tokens = max_tokens / 2
                response = None

    if response is None:
        return "[Error getting state report]"

    report = response.split("<report>")[-1].split("</report>")[0].strip()
    print(f"Obtained State Report")

    return report


async def call_info_integrate(sem, question, report_group, max_retries=10):
    max_tokens = 64 * 1024
    user_input = INTEGRATE_PROMPT.format(question=question)

    report_group = [r for r in report_group if r != "[Error getting state report]"]
    if len(report_group) == 0:
        return 0, "[No Valid Answer]"
    
    for i, report in enumerate(report_group):
        user_input += f"\n\n[Report {i+1}]: {report}"

    async with sem:
        for retry in range(max_retries):
            try:
                messages = [
                    {'role': 'system', 'content': CONVERGE_SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_input}
                ]
                
                response = await get_llm_response(messages, max_tokens)
                response = response.split('</think>')[-1].strip()
                if "Error getting visit response" in response or "time out" in response:
                    raise Exception(response)
                else:
                    break
            except Exception as e:
                await asyncio.sleep(2)
                if "time out" not in str(e).lower():    
                    max_tokens = max_tokens / 2
                response = None

    if response is None:
        return "[Error getting integrated answer]"

    final_answer = response.split("<answer>")[-1].split("</answer>")[0].strip()
    print(f"Obtained Integrated Answer: {final_answer}")
    
    return len(report_group), final_answer


async def call_converge(sem, traj_group):
    question = traj_group[0]['question']
    answer = traj_group[0]['answer']
    report_group = []
    for traj in traj_group:
        report = await call_state_report(sem['report'], traj)
        report_group.append(report)

    merge_num, prediction = await call_info_integrate(sem['merge'], question, report_group)

    if merge_num == 0:
        prediction = "[No Prediction]"
    
    return {'question': question, 'answer': answer, 'prediction': prediction, 'merge_num': merge_num, 'report_group': report_group, 'traj_group': traj_group}


async def main():
    data_path = "[YOUR-ROLLOUT-FILE-PATH-HERE]"  # TODO

    report_sem = asyncio.Semaphore(64)
    merge_sem = asyncio.Semaphore(32)
    sem = {
        'report': report_sem,
        'merge': merge_sem
    }
    mode = 'converge_info'

    dataset = read_jsonl(data_path)

    tasks = []
    clustered_dataset = cluster_by_question(dataset)
    for cluster in clustered_dataset:
        filtered_cluster = []
        for traj in cluster:
            if 'prediction' in traj.keys() and traj['prediction'] != '[No Prediction]':
                filtered_cluster.append(traj)
    
        tasks.append(call_converge(sem, filtered_cluster, data_path))

    results = []

    with open(f"{data_path.replace('.jsonl', f'_{mode}.jsonl')}", "a") as f:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Converging ..."):
            try:
                result = await future
                results.append(result)
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
    asyncio.run(main())