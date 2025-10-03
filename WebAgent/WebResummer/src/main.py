import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import threading
from react_agent import MultiTurnReactAgent
from prompt import SYSTEM_PROMPT
from tool_search import *
from tool_visit import * 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--dataset", type=str, default="gaia")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--roll_out_count", type=int, default=3)
    parser.add_argument("--summary_iteration", type=int, default=1000)
    args = parser.parse_args()

    model = args.model
    output_base = args.output
    roll_out_count = args.roll_out_count

    # Parse model name (the part after the last / in the path)
    model_name = os.path.basename(model.rstrip('/'))
    
    # Create output directory structure: output_base/model_name_sglang/dataset_name/
    model_dir = os.path.join(output_base, f"{model_name}_sglang")
    dataset_dir = os.path.join(model_dir, args.dataset)
    
    # Create directories
    os.makedirs(dataset_dir, exist_ok=True)
    
    print(f"Model name: {model_name}")
    print(f"Dataset name: {args.dataset}")
    print(f"Output directory: {dataset_dir}")
    print(f"Rollout count: {roll_out_count}")

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
    
    tasks_to_run = []
    for rollout_idx in range(1, roll_out_count + 1):
        output_file = os.path.join(dataset_dir, f"iter{rollout_idx}.jsonl")
        print(f"\nBegin rollout {rollout_idx}/{roll_out_count}")
        base_num = len(tasks_to_run)

        processed_queries = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            # Check for successful completion based on absence of top-level error key
                            if "question" in data and "error" not in data:
                                processed_queries.add(data["question"].strip())
                        except json.JSONDecodeError:
                            print(f"Warning: Skipping invalid line in output file: {line.strip()}")
            except FileNotFoundError:
                pass

        for item in items:
            question = item.get("question", "").strip()
            if not question:
                print(f"Warning: Skipping item with empty question: {item}")
                continue

            if question not in processed_queries:
                tasks_to_run.append({"item": item.copy(), "rollout_id": rollout_idx})
            else:
                print(f"Skipping already processed question: {question}")

        print(f"Total questions in input: {len(items)}")
        print(f"Already successfully processed: {len(processed_queries)}")
        print(f"Total tasks to run for rollout {rollout_idx}: {len(tasks_to_run) - base_num}")

    if not tasks_to_run:
        print(f"All rollouts completed, no need to execute")
    else:
        llm_cfg = {
            'model': model,
            'generate_cfg': {
                'max_input_tokens': 320000,
                'max_retries': 10, 
                'temperature': args.temperature, 
                'top_p': args.top_p
            }, 
            'model_type': 'qwen_dashscope'
        }
        
        test_agent = MultiTurnReactAgent(
            llm=llm_cfg,
            function_list=["search", "visit"],
            system_message=SYSTEM_PROMPT
        )

        write_locks = {i: threading.Lock() for i in range(1, roll_out_count + 1)}

        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit tasks
            future_to_task = {
                executor.submit(
                    test_agent._run,
                    task,
                    model,
                    args.summary_iteration,
                ): task
                for task in tasks_to_run
            }

            for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run), desc=f"Processing Rollout {rollout_idx}"):
                task_info = future_to_task[future]
                rollout_idx = task_info["rollout_id"]
                output_file = os.path.join(dataset_dir, f"iter{rollout_idx}.jsonl")
                try:
                    result = future.result(timeout=3600)
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
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
                    print("===============================")
                    print(error_result)
                    print("===============================")
                    
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

    print(f"\nAll {roll_out_count} rollouts completed!")
