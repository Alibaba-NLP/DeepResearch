from pydantic import BaseModel
from openai import OpenAI
import concurrent.futures
from typing import Literal
import litellm 
import os 
import argparse
import json
import concurrent 
from tqdm import tqdm 
from transformers import AutoTokenizer 
import re 
from prompt import * 
import traceback
import tiktoken
import time
import threading
thread_local = threading.local()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY","")
os.environ['OPENAI_API_BASE'] = os.getenv("OPENAI_API_BASE","") 
API_KEY= os.getenv("API_KEY","")
BASE_URL=os.getenv("BASE_URL","")

def get_client():
    if not hasattr(thread_local, 'client'):
        thread_local.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
    return thread_local.client

extracted_answer_format_for_confidence = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_answer",
        "schema": {
            "type": "object",
            "properties": {
                "extracted_final_answer": {"type": "string"},
                "reasoning": {"type": "string"},
                "correct": {"type": "string", "enum": ["yes", "no"]},
                "confidence": {"type": "number"},
                "strict": {"type": "boolean"},
            },
            "required": ["extracted_final_answer", "reasoning", "correct", "confidence", "strict"],
            "additionalProperties": False
        },
        "strict": True
    }
}

extracted_answer_format_for_xbench = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_answer",
        "schema": {
            "type": "object",
            "properties": {
                "最终答案": {"type": "string"},
                "解释": {"type": "string"},
                "结论": {"type": "string", "enum": ["正确", "错误"]},
            },
            "required": ["最终答案", "解释", "结论"],
            "additionalProperties": False
        },
        "strict": True
    }
}


def is_correct_judgement(judgement):
    return judgement.lower() == "correct" or (judgement and judgement.lower()[0] == "a")


def call_llm_judge(item): 
    global judge_prompt, dataset, judge_model
    
    question = item["question"]
    correct_answer = item["answer"]
    response = item["prediction"].strip()
    prompt = judge_prompt.format(question=question, correct_answer=correct_answer, response=response)
    
    for attempt in range(100):
        try: 
            if judge_model == "openai/qwen2.5-72b-instruct":
                response = litellm.completion(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    num_retries=5
                )
                judgement = response.choices[0].message["content"]
            elif judge_model == "google/gemini-2.0-flash-001":
                client = get_client()
                response_obj = client.beta.chat.completions.parse(
                    model=judge_model,
                    max_completion_tokens=8192, 
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    response_format=extracted_answer_format_for_xbench,
                    timeout=100.0
                ) 
                raw_judge = json.loads(response_obj.choices[0].message.content)
                judgement = "Correct" if raw_judge["结论"].lower() == "正确" else ""

            elif 'browsecomp' in dataset:
                os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY","")
                response = litellm.completion(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    num_retries=5,
                    response_format=extracted_answer_format_for_confidence
                )
                
                raw_content = response.choices[0].message["content"]
                raw_judge = json.loads(raw_content)
                judgement = "Correct" if raw_judge["correct"].lower() == "yes" else ""
                
            else:
                response = litellm.completion(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    num_retries=5
                )
                judgement = response.choices[0].message["content"]

            return {
                "question": question, 
                "answer": correct_answer, 
                "judgement": judgement
            }
        
        except Exception as e:
            if attempt == 4:  
                print(f"Error judgement for question: {question}: {e}")
                return {
                    "question": question,  
                    "answer": correct_answer, 
                    "judgement": "Error",
                    "error": str(e)
                }
            time.sleep(3)
            continue  


def process_single_round(input_file): 
    with open(input_file, 'r', encoding='utf-8') as f: 
        items = [json.loads(line) for line in f]
    
    return items 


def get_termination_value(item):
    if "termination" in item:
        return item["termination"]
    
    messages = item.get("messages", [])
    if not messages:
        return "unknown"
    
    last_message = messages[-1]["content"] if messages else ""
    
    
    if "max_turns_reached" in last_message.lower():
        return "max_turns_reached"
    elif "max_tokens_reached" in last_message.lower():
        return "max_tokens_reached"
    elif "<answer>" in last_message and "</answer>" in last_message:
        return "answered"
    else:
        return "unknown"


def count_tokens_with_tokenizer(text, tokenizer):
    try:
        if hasattr(tokenizer, 'encode'):
            return len(tokenizer.encode(text))
        else:  
            return len(tokenizer.encode(text))
    except:
        
        return len(text) // 4


def aggregate_statistics(round1_file, round2_file, round3_file):
    round1_stats = single_round_statistics(round1_file)
    round2_stats = single_round_statistics(round2_file)
    round3_stats = single_round_statistics(round3_file)
    
    keys = round1_stats.keys()  
    avg_stats = {} 
    for key in keys: 
        if isinstance(round1_stats[key], dict):
            
            avg_stats[key] = {}
            all_keys = set(round1_stats[key].keys()) | set(round2_stats[key].keys()) | set(round3_stats[key].keys())
            for nested_key in all_keys:
                val1 = round1_stats[key].get(nested_key, 0)
                val2 = round2_stats[key].get(nested_key, 0)
                val3 = round3_stats[key].get(nested_key, 0)
                avg_stats[key][nested_key] = round((val1 + val2 + val3) / 3, 3)
        else:
            avg_stats[key] = round((round1_stats[key] + round2_stats[key] + round3_stats[key]) / 3 , 3)
    
    return avg_stats


def single_round_statistics(input_file):
    contents = process_single_round(input_file) 

    
    num_invalid, num_extra = 0, 0  
    
    tool_use_cnt, visit_tool_cnt, search_tool_cnt, other_tool_cnt = [], [], [], [] 
    
    all_ans_lengths, all_think_lengths = [], []
    
    
    all_tool_calls_per_question = []
    all_assistant_tokens_per_question = []
    all_assistant_tokens_per_message = []
    termination_counts = {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(os.getenv("Qwen2_5_7B_PATH", ""))
    except Exception as e: 
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    for item in contents:
        messages = item["messages"]
        final_msg = messages[-1]["content"] if len(messages) else ""
        
        
        if "<answer>" not in final_msg or "</answer>" not in final_msg: 
            num_invalid += 1 
            answer_length = 0 
        else:
            answer_length = len(final_msg.split("<answer>")[1].split("</answer>")[0].strip())
        
        
        num_tool_use, num_visit_tool, num_search_tool, num_other_tool = 0, 0, 0, 0 
        think_lengths = []
        question_assistant_tokens = 0
        
        
        for msg in messages:
            if msg['role'] == 'assistant':
                content = msg['content']
                
                
                remaining_content = content
                while "<tool_call>" in remaining_content and "</tool_call>" in remaining_content:
                    start_idx = remaining_content.find("<tool_call>")
                    end_idx = remaining_content.find("</tool_call>")
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        tool_call_content = remaining_content[start_idx + 11:end_idx].strip()
                        if tool_call_content:  
                            num_tool_use += 1
                            
                            try:
                                tool_call = json.loads(tool_call_content)
                                tool_name = tool_call.get('name', '')
                                if tool_name == 'search':
                                    num_search_tool += 1
                                elif 'visit' in tool_name:
                                    num_visit_tool += 1
                                else:
                                    num_other_tool += 1
                            except Exception:
                                if "visit" in tool_call_content:
                                    num_visit_tool += 1
                                elif "search" in tool_call_content:
                                    num_search_tool += 1
                                else:
                                    num_other_tool += 1
                        
                        remaining_content = remaining_content[end_idx + 12:]  
                    else:
                        break
                
                think_lengths.append(len(content))
                
                assistant_tokens = count_tokens_with_tokenizer(content, tokenizer)
                question_assistant_tokens += assistant_tokens
                all_assistant_tokens_per_message.append(assistant_tokens)

        tool_use_cnt.append(num_tool_use) 
        visit_tool_cnt.append(num_visit_tool)  
        search_tool_cnt.append(num_search_tool)   
        other_tool_cnt.append(num_other_tool) 

        all_ans_lengths.append(answer_length) 
        think_length = sum(think_lengths) / len(think_lengths) if think_lengths else 0  
        all_think_lengths.append(think_length) 
        
        all_tool_calls_per_question.append(num_tool_use)
        all_assistant_tokens_per_question.append(question_assistant_tokens)
        
        termination = get_termination_value(item)
        termination_counts[termination] = termination_counts.get(termination, 0) + 1

        try:
            if len(tokenizer.encode("".join([msg["content"] for msg in messages]))) > 30000:
                num_extra += 1  
        except:
            pass
    
    total_questions = len(contents)
    termination_freq = {k: round(v / total_questions, 3) for k, v in termination_counts.items()}
    
    return {
        "extra_length": num_extra, 
        "num_invalid": num_invalid,
        "avg_action": sum(tool_use_cnt) / len(tool_use_cnt),
        "avg_visit_action": sum(visit_tool_cnt) / len(visit_tool_cnt), 
        "avg_search_action": sum(search_tool_cnt) / len(search_tool_cnt), 
        "avg_other_action": sum(other_tool_cnt) / len(other_tool_cnt), 
        "avg_ans_length": sum(all_ans_lengths) / len(all_ans_lengths), 
        "avg_think_length": sum(all_think_lengths) / len(all_think_lengths),
        "avg_tool_calls_per_question": sum(all_tool_calls_per_question) / len(all_tool_calls_per_question) if all_tool_calls_per_question else 0,
        "avg_assistant_tokens_per_question": sum(all_assistant_tokens_per_question) / len(all_assistant_tokens_per_question) if all_assistant_tokens_per_question else 0,
        "avg_assistant_tokens_per_message": sum(all_assistant_tokens_per_message) / len(all_assistant_tokens_per_message) if all_assistant_tokens_per_message else 0,
        "termination_freq": termination_freq
    }


def calculate_enhanced_statistics(round_results, round_items):
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(os.getenv("Qwen2_5_7B_PATH", ""))
    except Exception as e: 
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    enhanced_stats = {}
    correct_tool_calls = []
    correct_assistant_tokens = []
    
    for round_name in ["round1", "round2", "round3"]:
        results = round_results[round_name]
        items = round_items[round_name]

        for result in results:
            if not is_correct_judgement(result["judgement"]):
                continue
            try:
                matching_item = [item for item in items if item['messages'][1]['content'] == result['question']]
            except:
                items = [item for item in items if len(item['messages'])>0]
                matching_item = [item for item in items if item['messages'][1]['content'] == result['question']]
            if not matching_item:
                continue
            item = matching_item[0]
            
            messages = item["messages"]
            num_tool_use = 0
            question_assistant_tokens = 0
            
            for msg in messages:
                if msg['role'] == 'assistant':
                    content = msg['content']

                    think_content = content.split('<think>')[-1].split('</think>')[0]

                    num_tool_use += 1
                    
                    assistant_tokens = count_tokens_with_tokenizer(think_content, tokenizer)
                    question_assistant_tokens += assistant_tokens
            
            correct_tool_calls.append(num_tool_use)
            correct_assistant_tokens.append(question_assistant_tokens)
        
    avg_tool_calls_correct = sum(correct_tool_calls) / len(correct_tool_calls) if correct_tool_calls else 0
    avg_assistant_tokens_correct = sum(correct_assistant_tokens) / len(correct_assistant_tokens) if correct_assistant_tokens else 0

    return {
        "avg_tool_calls_per_question_correctly_solved": round(avg_tool_calls_correct, 3),
        "avg_assistant_tokens_per_question_correctly_solved": round(avg_assistant_tokens_correct, 3)
    }
        
        
def aggregate_results(round1_results, round2_results, round3_results): 
    global dataset 
    query_results = {} 
    
    for results, round_name in zip([round1_results, round2_results, round3_results], ["round1", "round2", "round3"]): 
        for result in results: 
            query = result["question"] 
            if query not in query_results:  
                query_results[query] = {
                    "round1": None, 
                    "round2": None,  
                    "round3": None, 
                    "answer": result["answer"]
                }
            
            if is_correct_judgement(result["judgement"]):
                query_results[query][round_name] = "Correct"
            else:
                query_results[query][round_name] = result["judgement"].capitalize()
    
    return query_results


def calculate_pass_at_k(query_results, k=10): 
    total_correct = 0 

    for query, results in query_results.items():
        rounds = [results["round1"], results["round2"], results["round3"]][:k] 

        if "Correct" in rounds: 
            total_correct += 1 
    
    overall_pass = total_correct / len(query_results)
    return round(overall_pass * 100, 2)


def calculate_best_pass_at_1(query_results):  
    round_correct = {round_name: 0 for round_name in ["round1", "round2", "round3"]}

    for query, results in query_results.items():
        for round_name in ["round1", "round2", "round3"]: 
            if results[round_name] == "Correct":  
                round_correct[round_name] += 1 

    overall_best = max(
        round_correct[round_name] / len(query_results)
        for round_name in ["round1", "round2", "round3"]
    )

    return round(overall_best * 100, 2)


def calculate_avg_pass_at_3(query_results): 
    round_names = ["round1", "round2", "round3"]
    total_correct = {round_name: 0 for round_name in round_names}

    for query, results in query_results.items():  
        for round_name in round_names:  
            if results[round_name] == "Correct":
                total_correct[round_name] += 1 
    
    avg_overall = sum(total_correct[r] / len(query_results) for r in round_names) / len(round_names)
    
    return round(avg_overall * 100, 2)


def main():
    global judge_prompt, dataset, judge_model
    parser = argparse.ArgumentParser(description="Evaluate model predictions across multiple rounds")
    parser.add_argument("--input_folder", help="Path to prediction files")
    parser.add_argument("--restore_result_path",default='summary.jsonl', help="record result")
    parser.add_argument("--dataset", type=str, default="browsecomp_en", choices=["gaia", 
                                                                        "browsecomp_zh",
                                                                        "browsecomp_en_full", 
                                                                        "webwalker", 
                                                                        "xbench-deepsearch",
                                                                        ])
    args = parser.parse_args()
    
    dataset = args.dataset  
    if dataset in ["gaia", "webwalker"]: 
        judge_model = "openai/qwen2.5-72b-instruct"
        judge_prompt = JUDGE_PROMPT_GAIA 
    elif dataset in ["xbench-deepsearch"]: 
        judge_prompt = JUDGE_PROMPT_XBENCH
        judge_model = "google/gemini-2.0-flash-001"
    elif dataset.startswith("browsecomp_zh"):
        judge_model = "gpt-4o-2024-08-06"
        judge_prompt = JUDGE_PROMPT_BROWSECOMP_OFFICIAL 
    elif dataset.startswith("browsecomp_en"):
        judge_model = "gpt-4o-2024-08-06"
        judge_prompt = JUDGE_PROMPT_BROWSECOMP_OFFICIAL
    else:
        judge_model = "openai/qwen2.5-72b-instruct"
        judge_prompt = JUDGE_PROMPT_GAIA 
    print(f"Using {dataset} judge prompt ...")
    print(f"Judge prompt:\n {judge_prompt}")
    print(f"Judge model:\n {judge_model}")

    round1_file, round2_file, round3_file = os.path.join(args.input_folder, "iter1.jsonl"), os.path.join(args.input_folder, "iter2.jsonl"), os.path.join(args.input_folder, "iter3.jsonl") 
    for file in [round1_file, round2_file, round3_file]:
        assert os.path.exists(file), f"Prediction {file} not found, three  rounds are required "
     
    round_items = {
        "round1": process_single_round(round1_file),
        "round2": process_single_round(round2_file),
        "round3": process_single_round(round3_file)
    }

    round_results = {} 

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:  
        for round_name, items in round_items.items():
            futures = {executor.submit(call_llm_judge, item): item for item in items} 
            round_results[round_name] = [] 

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Evaluating {round_name}"):
                round_results[round_name].append(future.result())
    
    for round_name in ["round1", "round2", "round3"]:
        input_file = {"round1": round1_file, "round2": round2_file, "round3": round3_file}[round_name]
        scored_file = input_file.replace(".jsonl", "_scored.jsonl")
        original_items = round_items[round_name]
        
        sorted_results = sorted(round_results[round_name], 
                              key=lambda x: original_items.index(next(item for item in original_items if item["question"] == x["question"])))
        
        with open(scored_file, 'w', encoding='utf-8') as f:
            for orig_item, scored_result in zip(original_items, sorted_results):
                scored_item = {
                    "is_correct": is_correct_judgement(scored_result["judgement"]),
                    "judgement": scored_result["judgement"]
                }
                if "error" in scored_result:
                    scored_item["error"] = scored_result["error"]
                    
                scored_item.update(orig_item)
                f.write(json.dumps(scored_item, ensure_ascii=False) + '\n')

    aggr_results = aggregate_results(round_results["round1"], round_results["round2"], round_results["round3"]) 
    
    pass_at_3 = calculate_pass_at_k(aggr_results, k=3)
    best_pass_at_1 = calculate_best_pass_at_1(aggr_results)
    avg_pass_at_3 = calculate_avg_pass_at_3(aggr_results) 
    

    round_performance = {
        f"Round{i}_Pass@1": round(sum(1 for r in round_results[f"round{i}"] if is_correct_judgement(r["judgement"])) / len(round_results[f"round{i}"]) * 100, 2)
        for i in [1, 2, 3]
    }

    print(f"===========")
    print(f"Avg. Pass@3 {avg_pass_at_3}%") 
    print(f"Best Pass@1 {best_pass_at_1}%")  
    print(f"Pass@3 {pass_at_3}%") 
    print(f"Pass@1 Round 1: {round_performance['Round1_Pass@1']}%  Round 2: {round_performance['Round2_Pass@1']}%  Round 3: {round_performance['Round3_Pass@1']}% \n")
    
    aggr_statistics = aggregate_statistics(round1_file, round2_file, round3_file)
    print(f"# Invalid {aggr_statistics['num_invalid']}  # Extra Length {aggr_statistics['extra_length']}") 
    print(f"Avg. Action {aggr_statistics['avg_action']:.2f}  Avg. Visit Action {aggr_statistics['avg_visit_action']:.2f}  Avg. Search Action {aggr_statistics['avg_search_action']:.2f}  Avg. Other Action {aggr_statistics['avg_other_action']:.2f}") 
    print(f"Avg. Answer Length {aggr_statistics['avg_ans_length']:.2f}  Avg. Thinking Length {aggr_statistics['avg_think_length']:.2f}")
    enhanced_statistics = calculate_enhanced_statistics(round_results, round_items)
    print(f"\n=== ADDITIONAL STATISTICS ===")
    print(f"Avg. Tool Calls per Question: {aggr_statistics['avg_tool_calls_per_question']:.2f}")
    print(f"Avg. Tool Calls per Question (Correctly Solved): {enhanced_statistics['avg_tool_calls_per_question_correctly_solved']:.2f}")
    print(f"Avg. Assistant Tokens per Question: {aggr_statistics['avg_assistant_tokens_per_question']:.2f}")
    print(f"Avg. Assistant Tokens per Question (Correctly Solved): {enhanced_statistics['avg_assistant_tokens_per_question_correctly_solved']:.2f}")
    print(f"Avg. Assistant Tokens per Message: {aggr_statistics['avg_assistant_tokens_per_message']:.2f}")
    
    print(f"\n=== TERMINATION FREQUENCIES ===")
    for termination_type, frequency in aggr_statistics['termination_freq'].items():
        print(f"{termination_type}: {frequency:.3f}")
    
    print(f"===========" )

    overall_eval_dict = {
        "dataset": dataset, 
        "files": {
            "round1": round1_file,  
            "round2": round2_file,   
            "round3": round3_file
        }, 
        "overall": {
            "avg_pass_at_3": avg_pass_at_3, 
            "best_pass_at_1": best_pass_at_1, 
            "pass_at_3": pass_at_3
        }, 
        "individual": round_performance, 
        "statistics": {**aggr_statistics, **enhanced_statistics}
    }

    with open(args.restore_result_path, 'a', encoding='utf-8') as jsonl_file:
        jsonl_file.write(json.dumps(overall_eval_dict, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    judge_prompt, dataset = None, ""
    try:
        main()
    except Exception as e:
        error_str = traceback.format_exc()
        print(f"Evaluation Failed: {e}") 
        print("Trace Back", error_str)
