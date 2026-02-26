import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import threading
from datetime import datetime
from react_agent_search_id import MultiTurnReactAgentSearch
from prompt.search_sys_prompt_2 import SEARCH_SYSTEM_PROMPT
from prompt.search_user_prompt_id_3 import SEARCH_USER_PROMPT

from tool.tool_search_and_visit import *
from tool.tool_visit import * 
from tool.tool_retrieve import *

def check_invalid_item(item_list):
    new_item_list = []
    for item in item_list:
        if "outline" not in item:
            continue
        if len(item["outline"]) < 100:
            continue
        new_item_list.append(item)
    return new_item_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output_path", type=str, default="output/debug.jsonl")
    parser.add_argument("--dataset", type=str, default="sample", choices=["sample",]
                                                                        )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--sys_prompt", type=str, default="SYSTEM_PROMPT_MULTI")
    parser.add_argument("--roll_out_count", type=int, default=1)
    parser.add_argument("--if_infer", type=bool, default=True)
    args = parser.parse_args()

    model = args.model
    # output_base = args.output
    roll_out_count = args.roll_out_count

    #### set env 
    os.environ['INFER_MODEL_PATH'] = model

    model_name = os.path.basename(model.rstrip('/'))
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print(f"model_name: {model_name}")
    print(f"dataset: {args.dataset}")
    print(f"output_path: {args.output_path}")
    print(f"Rollout次数: {roll_out_count}")

    data_filepath = f"eval_data/{args.dataset}.jsonl"

    try:
        if data_filepath.endswith(".json"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                raise ValueError("Input JSON must be a list of objects.")
            if items and not isinstance(items[0], dict):
                raise ValueError("Input JSON list items must be objects.")
        elif data_filepath.endswith(".jsonl"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")
        items = items
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_filepath}")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing input file {data_filepath}: {e}")
        exit(1)

    # 为每个rollout创建任务
    for rollout_idx in range(1, roll_out_count + 1):
        # output_file = os.path.join(dataset_dir, f"iter{rollout_idx}.jsonl")
        output_file = args.output_path
        
        print(f"\n开始第 {rollout_idx}/{roll_out_count} 次rollout")
        print(f"输出文件: {output_file}")
        
        # 检查已处理的查询
        processed_queries = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            # Check for successful completion based on absence of top-level error key, outline key
                            if "question" in data and "error" not in data and "outline" in data:
                                if len(data["outline"]) > 100:
                                    processed_queries.add(data["question"].strip())
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid line in output file: {line.strip()}")
            except FileNotFoundError:
                pass


        tasks_to_run = []
        for item in items:
            question = item.get("question", "").strip()
            if question == "":
                try:
                    user_msg = item["messages"][1]["content"] 
                    question = user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
                    item["question"] = question
                except Exception as e:
                    print(f"Extract question from user message failed: {e}")
            if not question:
                print(f"Warning: Skipping item with empty question: {item}")
                continue

            if question not in processed_queries:
                tasks_to_run.append({"item": item.copy(), "rollout_id": rollout_idx})
            else:
                print(f"Skipping already processed question: {question}")

        print(f"Total questions in input: {len(items)}")
        print(f"Already successfully processed: {len(processed_queries)}")
        print(f"Total tasks to run for this rollout: {len(tasks_to_run)}")

        if not tasks_to_run:
            print(f"Rollout {rollout_idx} 已完成，跳过")
            continue

        llm_cfg = {
            'model': model,
            'generate_cfg': {
                'max_input_tokens': 320000,
                'max_retries': 10, 
                'temperature': args.temperature, 
                'top_p': args.top_p,
                'if_infer': args.if_infer,
            }, 
            'model_type': 'qwen_dashscope'
        }
        
        system_message = SEARCH_SYSTEM_PROMPT + "\nCurrent date: " + datetime.now().strftime("%Y-%m-%d")
        

        test_agent = MultiTurnReactAgentSearch(
            llm=llm_cfg,
            function_list=["search_and_visit"],
            system_message=system_message
        )

        # 创建文件写入锁
        write_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit tasks
            future_to_task = {
                executor.submit(
                    test_agent._run,
                    task,
                    model
                ): task
                for task in tasks_to_run
            }

            for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run), desc=f"Processing Rollout {rollout_idx}"):
                task_info = future_to_task[future]
                try:
                    result = future.result(timeout=1800)
                    # 使用锁保护文件写入操作
                    with write_lock:
                        with open(output_file, "a", encoding="utf-8") as f:
                            language = task_info["item"].get("language", "")
                            if language != "":
                                result["language"] = language
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                except concurrent.futures.TimeoutError:
                    print(f'Timeout (>1800s): "{task_info["item"]["question"]}" '
                          f'(Rollout {task_info["rollout_id"]})')
                    future.cancel()
                    error_result = {
                        "question": task_info["item"]["question"],
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_id": task_info["rollout_id"],
                        "error": "Timeout (>1800s)",
                        "messages": [],
                        "prediction": "[Failed]"
                    }
                    with write_lock:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                except Exception as exc:
                    print(f'Task for question "{task_info["item"]["question"]}" (Rollout {task_info["rollout_id"]}) generated an exception: {exc}')
                    # Log error to the output file
                    error_result = {
                        "question": task_info["item"]["question"],
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_id": task_info["rollout_id"],
                        "error": f"Future resolution failed: {exc}",
                        "messages": [],
                        "prediction": "[Failed]",
                    }
                    language = task_info["item"].get("language", "")
                    if language != "":
                        error_result["language"] = language
                    print("===============================")
                    print(error_result)
                    print("===============================")
                    
                    # 同样使用锁保护错误写入
                    with write_lock:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
        
        print(f"Rollout {rollout_idx} 完成")
    
    print(f"\n所有 {roll_out_count} 次rollout完成!")
