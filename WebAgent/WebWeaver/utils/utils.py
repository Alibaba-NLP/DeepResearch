import json
import os
import logging

def read_jsonl(file_path):
    """Reads a JSONL file, parsing each line as a JSON object."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error in file {file_path}: {e}")
        return []

def save_jsonl(data_list, file_path):
    """Saves a list of processed data to a JSONL file."""
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logging.info(f"Processed data has been saved to: {file_path}")