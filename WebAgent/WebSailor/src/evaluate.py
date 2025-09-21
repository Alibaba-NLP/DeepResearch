import concurrent.futures
import os 
import argparse
import json
import concurrent 
from tqdm import tqdm 
import re 
from prompt import JUDGE_PROMPT_GAIA, JUDGE_PROMPT_BC, JUDGE_PROMPT_QA
import traceback
from openai import OpenAI
import tiktoken


def extract_correct_judgement(response: str) -> str:
    match = re.search(r'correct\s*:\s*(yes|no)', response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    else:
        return None
def call_llm_judge(item): 
    """Judge if predicted answer matches ground-truth""" 
    global judge_prompt
    try: 
        question = item["question"]
        correct_answer = item["answer"]

        response = item["prediction"].strip()

        prompt = judge_prompt.format(question=question, correct_answer=correct_answer, response=response)

        openai_api_key = "EMPTY"
        openai_api_base = "http://127.0.0.1:6002/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        max_tries = 10
        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(
                                    model='qwen2.5-72b-instruct',
                                    messages=[{"role": "user", "content": prompt}],
                )    
                response = chat_response.choices[0].message.content
                if response:
                    break
            except Exception as e:
                if attempt == (max_tries - 1):
                    raise e

        return {
            "question": question, 
            "answer": correct_answer, 
            "judgement": response
        }
    
    except Exception as e:
        print(f"Error judgement for question: {question}: {e}")
        return {
            "question": question,  
            "answer": correct_answer, 
            "judgement": "Error",
            "error": str(e)
        }


def process_single_round(input_file): 
    with open(input_file, 'r', encoding='utf-8') as f: 
        items = [json.loads(line) for line in f]
    
    return items 


def aggregate_statistics(round1_file, round2_file, round3_file):
    round1_stats = single_round_statistics(round1_file)
    round2_stats = single_round_statistics(round2_file)
    round3_stats = single_round_statistics(round3_file)
    
    keys = round1_stats.keys()  
    avg_stats = {} 
    for key in keys: 
        avg_stats[key] = round((round1_stats[key] + round2_stats[key] + round3_stats[key]) / 3 , 3)
    
    return avg_stats


def single_round_statistics(input_file):
    contents = process_single_round(input_file) 

    # Illegal Analysis 
    num_invalid, num_extra = 0, 0  
    # Tool Analysis 
    tool_use_cnt, visit_tool_cnt, search_tool_cnt, other_tool_cnt = [], [], [], [] 
    # Thinking Analysis 
    all_ans_lengths, all_think_lengths = [], []  

    tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    for item in contents:
        texts = item["messages"]
        final_msg = texts[-1]["content"] if len(texts) else ""
        
        # Analyze answer 
        if "<answer>" not in final_msg or "</answer>" not in final_msg: 
            num_invalid += 1 
            answer_length = 0 
        else:
            answer_length = len(final_msg.split("<answer>")[1].split("</answer>")[0].strip())
        
        # Analyze tool use & thinking
        num_tool_use, num_visit_tool, num_search_tool, num_other_tool = 0, 0, 0, 0 
        think_lengths = []
        for idx in range(2, len(texts), 2):
            response = texts[idx]["content"] 
            # TODO: relaxed matching for thinking, strict mode: <think>(.*?)</think>
            # thinking = re.findall(r"(.*?)</think>", response, re.DOTALL) 
            tool = re.findall(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL) 
            
            if tool: 
                try:
                    tool_name = tool[0].split("name\": \"")[1].split("\"")[0].strip()
                except Exception:
                    tool_name = ""
                    
                num_tool_use += 1 
                if "visit" in tool_name:  
                    num_visit_tool += 1 
                elif "search" in tool_name: 
                    num_search_tool += 1  
                else:
                    num_other_tool += 1

            think_lengths.append(len(response)) 

        tool_use_cnt.append(num_tool_use) 
        visit_tool_cnt.append(num_visit_tool)  
        search_tool_cnt.append(num_search_tool)   
        other_tool_cnt.append(num_other_tool) 

        all_ans_lengths.append(answer_length) 
        think_length = sum(think_lengths) / len(think_lengths) if think_lengths else 0  
        all_think_lengths.append(think_length) 

        # Overlength 
        if len(tokenizer.encode("".join([text["content"] for text in texts]))) > 30000:
            num_extra += 1  
    
    return {
        "extra_length": num_extra, # number of overlength responses  
        "num_invalid": num_invalid, # number of invalid responses 
        "avg_action": sum(tool_use_cnt) / len(tool_use_cnt), # avg. number of tool invocation 
        "avg_visit_action": sum(visit_tool_cnt) / len(visit_tool_cnt), # avg. number of visit tool invocation 
        "avg_search_action": sum(search_tool_cnt) / len(search_tool_cnt), # avg. number of search tool invocation 
        "avg_other_action": sum(other_tool_cnt) / len(other_tool_cnt), # avg. number of other tool invocation 
        "avg_ans_length": sum(all_ans_lengths) / len(all_ans_lengths), 
        "avg_think_length": sum(all_think_lengths) / len(all_think_lengths)
    }
        
        
def aggregate_results(round1_results, round2_results, round3_results): 
    """Aggregate results from multiple rounds""" 
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
            
            # For BrowseComp and XX_QA datasets, prompt requires "A" or "a" as the correct answer
            if ("browsecomp" in dataset) or ("qa" in dataset): 
                judge = extract_correct_judgement(result["judgement"])
                if judge:
                    if judge.lower() == 'yes':
                        result["judgement"] = "Correct"
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
    global judge_prompt, dataset
    parser = argparse.ArgumentParser(description="Evaluate model predictions across multiple rounds")
    parser.add_argument("--input_folder", help="Path to prediction files")
    parser.add_argument("--restore_result_path",default='summary.jsonl', help="record result")
    parser.add_argument("--dataset", type=str, default="gaia", choices=["gaia", 
                                                                        "browsecomp_zh", "browsecomp_zh_small", 
                                                                        "browsecomp_en", "browsecomp_en_full", "browsecomp_en_small", 
                                                                        "webwalker", 
                                                                        "simple_qa", "simple_qa_small",
                                                                        "time_qa",
                                                                        "xbench-deepsearch",
                                                                        "hle"])
    args = parser.parse_args()
    
    dataset = args.dataset  
    if dataset in ["gaia", "webwalker", "xbench-deepsearch", "hle"]: 
        judge_prompt = JUDGE_PROMPT_GAIA 
    elif dataset.startswith("browsecomp_zh"):
        judge_prompt = JUDGE_PROMPT_BC
    elif dataset.startswith("browsecomp_en"):
        judge_prompt = JUDGE_PROMPT_BC
    elif dataset.endswith("qa") or dataset.endswith("qa_small"): # for simple_qa and time_qa
        judge_prompt = JUDGE_PROMPT_QA
    else:
        judge_prompt = JUDGE_PROMPT_GAIA 
    print(f"Using {dataset} judge prompt ...")

    round1_file, round2_file, round3_file = os.path.join(args.input_folder, "iter1.jsonl"), os.path.join(args.input_folder, "iter2.jsonl"), os.path.join(args.input_folder, "iter3.jsonl") 
    for file in [round1_file, round2_file, round3_file]:
        assert os.path.exists(file), f"Prediction {file} not found, three  rounds are required "
     
    round_items = {
        "round1": process_single_round(round1_file),
        "round2": process_single_round(round2_file),
        "round3": process_single_round(round3_file)
    }

    round_results = {} 

    with concurrent.futures.ThreadPoolExecutor() as executor:  
        for round_name, items in round_items.items():
            futures = {executor.submit(call_llm_judge, item): item for item in items} 
            round_results[round_name] = [] 

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Evaluating {round_name}"):
                round_results[round_name].append(future.result())

    aggr_results = aggregate_results(round_results["round1"], round_results["round2"], round_results["round3"]) 
    
    pass_at_3 = calculate_pass_at_k(aggr_results, k=3)
    best_pass_at_1 = calculate_best_pass_at_1(aggr_results)
    avg_pass_at_3 = calculate_avg_pass_at_3(aggr_results) 
    
    aggr_statistics = aggregate_statistics(round1_file, round2_file, round3_file)

    round_performance = {
        f"Round{i}_Pass@1": round(sum(1 for r in round_results[f"round{i}"] if r["judgement"] == "Correct") / len(round_results[f"round{i}"]) * 100, 2)
        for i in [1, 2, 3]
    }

    print(f"===========")
    print(f"Avg. Pass@3 {avg_pass_at_3}%") 
    print(f"Best Pass@1 {best_pass_at_1}%")  
    print(f"Pass@3 {pass_at_3}%") 
    print(f"Pass@1 Round 1: {round_performance['Round1_Pass@1']}%  Round 2: {round_performance['Round2_Pass@1']}%  Round 3: {round_performance['Round3_Pass@1']}% \n")
    print(f"# Invalid {aggr_statistics['num_invalid']}  # Extra Length {aggr_statistics['extra_length']}") 
    print(f"Avg. Action {aggr_statistics['avg_action']:.2f}  Avg. Visit Action {aggr_statistics['avg_visit_action']:.2f}  Avg. Search Action {aggr_statistics['avg_search_action']:.2f}  Avg. Other Action {aggr_statistics['avg_other_action']:.2f}") 
    print(f"Avg. Answer Length {aggr_statistics['avg_ans_length']:.2f}  Avg. Thinking Length {aggr_statistics['avg_think_length']:.2f}")  
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
        "statistics": aggr_statistics
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
