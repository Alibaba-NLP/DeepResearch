import argparse 
from concurrent.futures import ThreadPoolExecutor, as_completed
from judge_prompt import JUDGE_PROMPT_GAIA, JUDGE_PROMPT_BC_en, JUDGE_PROMPT_BC_zh
import os 
import json 
import glob
from collections import defaultdict
from tqdm import tqdm 
import time
from transformers import AutoTokenizer 
# TODO: Replace with your own DashScope API key
import dashscope
dashscope.api_key = 'YOUR_DASHSCOPE_API_KEY'


def call_llm_judge(item, max_retries=10): 
    """Judge if predicted answer matches ground-truth""" 
    for _ in range(max_retries): 
        try: 
            question = item.get(args.question_key, "")
            correct_answer = item.get(args.answer_key, "") 
            response = item.get(args.prediction_key, "")
 
            judge_prompt = JUDGE_PROMPT_GAIA
            if args.dataset.startswith("browsecomp_zh"):
                judge_prompt = JUDGE_PROMPT_BC_zh
            elif args.dataset.startswith("browsecomp_en"):
                judge_prompt = JUDGE_PROMPT_BC_en
            
            prompt = judge_prompt.format(question=question, correct_answer=correct_answer, response=response)
        
            response = dashscope.Generation.call(
                model='qwen2.5-72b-instruct',
                messages=[{"role": "user", "content": prompt}],
            )
            judgement = response.output.text
            judgement = "Correct" if judgement[:1] in ["a", "A"]  else judgement
       
            if judgement == "Correct" and args.print_correct_question:
                print("Correct Question: ", question, "Prediction ", item.get(args.prediction_key, ""), "Ground-truth", correct_answer, "\n")

            return {
                "question": question, 
                "answer": correct_answer, 
                "judgement": judgement, 
            }
    
        except Exception as e:
            time.sleep(1)
            if _ == max_retries - 1: 
                print(f"Error judgement for question: {question}: {e}")
                return {
                    "question": question,  
                    "answer": correct_answer, 
                    "judgement": "Error",
                    "error": str(e),
                 }


def single_round_statistics(input_file, available_tools=None):  
    """Calculate statistics for a single round"""
    def avg_statistic(value_list):
        if value_list:
            return sum(value_list) / len(value_list)
        return 0 
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f: 
            samples = [json.loads(line) for line in f]
    except Exception as e:
        print(f"Error loading file {input_file}: {e}")
        return {}

    num_invalid = 0 
    tool_invocation = defaultdict(list)
    answer_lengths, traj_lengths = [], []
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("/path/to/your/Qwen2.5-72B-Instruct")
    except Exception as e: 
        import tiktoken
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    for sample in samples:
        msgs = sample.get("messages", [])
        final_msg = msgs[-1]["content"] if len(msgs) else "" 

        if "<answer>" not in final_msg or "</answer>" not in final_msg:  
            num_invalid += 1 
            answer_length = 0 
        else:
            answer = final_msg.split("<answer>")[1].split("</answer>")[0].strip()
            answer_length = len(tokenizer.encode(answer))
        answer_lengths.append(answer_length)
        
        cur_tool_invocation = defaultdict(int)
        for msg in msgs:
            if msg["role"] == "assistant": 
                try:
                    tool_call = msg["content"].split("<tool_call>")[1].split("</tool_call>")[0].strip() 
                    tool_call = json.loads(tool_call) 
                    tool_name = tool_call["name"] 
                    if available_tools and tool_name in available_tools:
                        cur_tool_invocation[tool_name] += 1 
                    else:
                        cur_tool_invocation["invalid"] += 1 
                    cur_tool_invocation["total"] += 1 
                except:
                    continue 
        
        for k, v in cur_tool_invocation.items(): 
            tool_invocation[k].append(v)
        
        traj_length = len(tokenizer.encode("".join(msg["content"] for msg in msgs)))
        traj_lengths.append(traj_length)
    
    metrics = {
        "num_invalid": num_invalid,  
        "avg_answer_length": avg_statistic(answer_lengths), 
        "avg_traj_length": avg_statistic(traj_lengths)
    }

    for k, v in tool_invocation.items(): 
        if k != "invalid":
            metrics[f"avg_tool_{k}"] = avg_statistic(v)
        else:
            metrics[f"avg_tool_invalid"] = sum(v) / len(samples)

    return metrics


def process_one_prediction(prediction_file): 
    try:
        iteration_name = prediction_file.split("/")[-1].replace(".jsonl", "")
        
        # Check if the scored file exists
        scored_file = prediction_file.replace(".jsonl", "_scored.jsonl")
        if os.path.exists(scored_file):
            print(f"Found existing scored file for {iteration_name}, loading results...")
            with open(scored_file, 'r', encoding='utf-8') as f:
                scored_items = [json.loads(line) for line in f]
            
            correct_predictions = []
            score_dict = defaultdict(bool)
            
            for scored_item in scored_items:
                if scored_item.get("is_correct", False):
                    correct_predictions.append({
                        "question": scored_item["question"], 
                        "answer": scored_item["answer"], 
                    })
                score_dict[scored_item["question"]] = scored_item.get("is_correct", False)
            
            acc = round(len(correct_predictions) / len(scored_items) * 100, 2)
            print(f"Loaded scored file: {scored_file} has {len(correct_predictions)} correct predictions (total {len(scored_items)}). Pass@1 {acc}%")
            
            return {
                "file": prediction_file,
                "accuracy": acc,
                "correct_count": len(correct_predictions),
                "total_count": len(scored_items),
                "correct_predictions": correct_predictions,
                "score_dict": score_dict
            }
        
        # If the scored file does not exist, score the predictions
        with open(prediction_file, 'r') as file: 
            predictions = [json.loads(line) for line in file]
        
        correct_predictions, score_dict = [], defaultdict(bool)
        judgement_results = []
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor: 
            futures = [executor.submit(call_llm_judge, item) for item in predictions]
            for future in tqdm(as_completed(futures), desc=f"Judging {iteration_name}", total=len(futures)): 
                result = future.result()
                judgement_results.append(result)
                
                if result["judgement"] == "Correct":
                    correct_predictions.append({
                        "question": result["question"], 
                        "answer": result["answer"], 
                    })
                
                score_dict[result["question"]] = result["judgement"] == "Correct"
        
        acc = round(len(correct_predictions) / len(predictions) * 100, 2)
        print(f"Prediction file: {prediction_file} has {len(correct_predictions)} correct predictions (total {len(predictions)}). Pass@1 {acc}%")
        
        # Save scored results
        if not os.path.exists(scored_file):
            print(f"Saving scored results for {iteration_name}...")
            with open(scored_file, 'w', encoding='utf-8') as f:
                for judgement_result in judgement_results:
                    orig_item = next((item for item in predictions if item["question"] == judgement_result["question"]), None) 
                    
                    save_item = orig_item.copy()
                    save_item["is_correct"] = judgement_result["judgement"] == "Correct"
                    save_item["origin_judgement"] = judgement_result["judgement"]
                    
                    if "error" in judgement_result:
                        save_item["error"] = judgement_result["error"]
                    f.write(json.dumps(save_item, ensure_ascii=False) + '\n')
        
        return {
            "file": prediction_file,
            "accuracy": acc,
            "correct_count": len(correct_predictions),
            "total_count": len(predictions),
            "correct_predictions": correct_predictions,
            "score_dict": score_dict
        }
    except Exception as e:
        print(f"Error processing file {prediction_file}: {e}")
        return {
            "file": prediction_file,
            "error": str(e)
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder")
    parser.add_argument("--question_key", type=str, default="question")
    parser.add_argument("--answer_key", type=str, default="answer")
    parser.add_argument("--prediction_key", type=str, default="prediction")
    parser.add_argument("--print_correct_question", action="store_true")
    parser.add_argument("--dataset", type=str, default="gaia")
    parser.add_argument("--available_tools", type=str, default="search,visit")
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--restore_result_path", default='output/summary.jsonl', help="record result")
    args = parser.parse_args()
    
    available_tools = args.available_tools.split(",") if args.available_tools else ["search", "visit"]
    
    all_scores_dict = defaultdict(list)
    acc_list = []
    file_list = []
    
    for path in glob.glob(os.path.join(args.input_folder, "iter*.jsonl")):
        if "_scored" in path:
            continue 

        result = process_one_prediction(path) 
        if "error" not in result:
            for question, score in result["score_dict"].items():
                all_scores_dict[question].append(score)
            acc_list.append(result["accuracy"])
            file_list.append(path)
    
    if not acc_list:
        print("No valid results found!")
        exit(1)
    
    # Compute Average Pass@1 
    avg_pass_at_1 = sum(acc_list) / len(acc_list)
    print(f"Average Pass@1: {avg_pass_at_1:.2f}")

    # Compute Best Pass@1 
    best_pass_at_1 = max(acc_list)
    print(f"Best Pass@1: {best_pass_at_1:.2f}")

    # Compute Pass@k
    correct_num = 0 
    for question, scores in all_scores_dict.items():
        if sum(scores) >= 1:
            correct_num += 1
    pass_at_k = correct_num / len(all_scores_dict) * 100
    print(f"Pass@{len(acc_list)}: {pass_at_k:.2f}")
    
    # Calculate statistics
    print("\n========== Statistics ==========")
    all_stats = []
    for file_path in file_list:
        stats = single_round_statistics(file_path, available_tools)
        if stats:
            all_stats.append(stats)
    
    if all_stats:
        # Aggregate statistics
        avg_stats = {}
        for key in all_stats[0].keys():
            avg_stats[key] = round(sum(stats.get(key, 0) for stats in all_stats) / len(all_stats), 2)
        
        print(f"# Invalid: {avg_stats.get('num_invalid', 0)}")
        print(f"Avg. Answer Length: {avg_stats.get('avg_answer_length', 0)}")
        print(f"Avg. Trajectory Length: {avg_stats.get('avg_traj_length', 0)}")
        
        for k, v in avg_stats.items(): 
            if k.startswith("avg_tool_"): 
                print(f"{k}: {v}")
    
    # Save overall results
    overall_eval_dict = {
        "dataset": args.dataset, 
        "files": file_list,
        "overall": {
            "avg_pass_at_1": avg_pass_at_1, 
            "best_pass_at_1": best_pass_at_1, 
            "pass_at_k": pass_at_k
        }, 
        "individual": {f"iter{i+1}_pass_at_1": acc for i, acc in enumerate(acc_list)},
        "statistics": avg_stats if all_stats else {}
    }

    with open(args.restore_result_path, 'a', encoding='utf-8') as jsonl_file:
        jsonl_file.write(json.dumps(overall_eval_dict, ensure_ascii=False) + '\n')
    
    print(f"\nResults saved to {args.restore_result_path}")
