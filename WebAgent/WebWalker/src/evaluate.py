import os
import json
import time
import concurrent.futures
from tqdm import tqdm
from datasets import load_dataset
from langchain.evaluation import load_evaluator

# Dictionary to store questions, answers, and additional information
info_adic = {}

# Load the dataset
ds = load_dataset("callanwu/WebWalkerQA", split="main")
for question, answer, info in zip(ds["question"], ds["answer"], ds["info"]):
    info_adic[question] = [answer, info]

def eval_result(input_path, output_path):
    """
    Evaluates prediction results against reference answers and generates a report.
    
    Parameters:
        input_path (str): Path to the input predictions file.
        output_path (str): Path to save the evaluation results and report.
    """
    evaluator = load_evaluator("cot_qa")
    data_list = []
    visited = []

    # Ensure output file exists
    if not os.path.exists(output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("")

    # Load already processed questions
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            visited.append(json.loads(line)["question"])

    # Load and filter data
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data["question"] not in visited:
                data["answer"] = info_adic.get(data["question"], [None, None])[0]
                if data["answer"] is not None:
                    data_list.append(data)

    def call(data):
        """Handles evaluation retries with exponential backoff."""
        max_retries = 10
        for attempt in range(max_retries):
            try:
                return evaluator.evaluate_strings(
                    prediction=data["pred"],
                    input=data["question"],
                    reference=data["answer"]
                )
            except Exception as e:
                print(f"Error during evaluation: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1 * (2 ** attempt))  # Exponential backoff
                else:
                    raise e  # Raise the exception if the last retry fails

    s = 0
    cnt = 0

    with tqdm(total=len(data_list)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_data = {executor.submit(call, data): data for data in data_list}
            for future in concurrent.futures.as_completed(future_to_data):
                try:
                    outputs = future.result(timeout=4)
                    data = future_to_data[future]
                    data["score"] = outputs["score"]

                    cnt += data["score"]
                    s += 1

                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")

                    pbar.update(1)
                    print("Current accuracy:", cnt / s)

                except Exception as e:
                    print(f"Error processing data: {e}")

    # Prepare statistics for the report
    single_source_easy, single_source_medium, single_source_hard = [], [], []
    multi_source_easy, multi_source_medium, multi_source_hard = [], [], []
    overall = []

    datas = []

    # Reload processed data
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["question"] in info_adic:
                item["info"] = info_adic[item["question"]][1]
                datas.append(item)

    for temp in datas:
        score = temp.get("score")
        if score is not None:
            info = temp.get("info", {})
            q_type = info.get("type")
            difficulty = info.get("difficulty_level")

            if q_type == "single_source":
                if difficulty == "easy":
                    single_source_easy.append(score)
                elif difficulty == "medium":
                    single_source_medium.append(score)
                elif difficulty == "hard":
                    single_source_hard.append(score)

            elif q_type == "multi_source":
                if difficulty == "easy":
                    multi_source_easy.append(score)
                elif difficulty == "medium":
                    multi_source_medium.append(score)
                elif difficulty == "hard":
                    multi_source_hard.append(score)

            overall.append(score)

    # Safely compute averages to avoid division by zero
    def safe_average(scores):
        return sum(scores) / len(scores) if scores else None

    result = {
        "single_source_easy": safe_average(single_source_easy),
        "single_source_medium": safe_average(single_source_medium),
        "single_source_hard": safe_average(single_source_hard),
        "multi_source_easy": safe_average(multi_source_easy),
        "multi_source_medium": safe_average(multi_source_medium),
        "multi_source_hard": safe_average(multi_source_hard),
        "overall": safe_average(overall)
    }

    # Save the report
    report_path = output_path.split(".jsonl")[0] + "_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Input prediction result path")
    parser.add_argument("--output_path", type=str, help="Evaluation output path")
    args = parser.parse_args()

    eval_result(args.input_path, args.output_path)
