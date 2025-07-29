#%%
import os
import json
import jsonlines
import pandas as pd
import numpy as np
from loguru import logger
from typing import List, Dict, Any, Union


def load_data_from_jsonl_file(file_path: str) -> list[dict]:
    """"""
    with open(file_path, "r", encoding="utf-8") as fp:
        data = [line for line in jsonlines.Reader(fp)]
        return data


# 函数：读取 JSON 或 JSONL 文件
def read_json_or_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    读取 JSON 或 JSONL 文件内容。

    参数:
        file_path: 文件路径（.json 或 .jsonl）。

    返回:
        包含文件内容的列表，若失败返回空列表。
    """
    try:
        if not os.path.isfile(file_path):
            logger.error(f"File does not exist: {file_path}")
            return []
        if os.stat(file_path).st_size == 0:
            logger.warning(f"File is empty: {file_path}")
            return []
        
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        elif file_path.endswith('.jsonl'):
            data = []
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON in {file_path} at line {line_num}: {e}")
            return data
        else:
            logger.error(f"Unsupported file format: {file_path}. Expected '.json' or '.jsonl'.")
            return []
    except PermissionError as e:
        logger.error(f"Permission denied reading {file_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return []


def read_all_jsonl_in_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    递归读取指定目录及其所有子目录下的所有 JSONL 文件。

    参数:
        directory_path: 目录路径。

    返回:
        包含所有 JSONL 文件内容的列表。
    """
    try:
        if not os.path.isdir(directory_path):
            logger.error(f"Invalid directory: {directory_path}")
            return []
        
        all_data = []
        for root, dirs, files in os.walk(directory_path):
            for file_name in files:
                if file_name.endswith('.jsonl'):
                    file_path = os.path.join(root, file_name)
                    logger.info(f"Reading JSONL file: {file_path}")
                    file_data = read_json_or_jsonl(file_path)
                    all_data.extend(file_data)
        return all_data
    except PermissionError as e:
        logger.error(f"Permission denied accessing directory {directory_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Failed to read directory {directory_path}: {e}")
        return []



def save_jsonl(data: Union[Dict[str, Any], List[Dict[str, Any]]], output_path: str, mode="w") -> bool:
    """
    将数据保存为 JSONL 文件。

    参数:
        data: 要保存的数据(单条字典或字典列表)。
        output_path: 输出文件路径。
        mode: 写入模式，'w'表示覆盖，'a'表示追加。

    返回:
        成功返回 True，失败抛出异常。
    """
    try:
        dir_path = os.path.dirname(output_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        with open(output_path, mode=mode, encoding="utf-8") as f:
            if isinstance(data, dict):
                # 单个字典，直接写入一行
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")
            else:
                # 字典列表，逐行写入
                for row in data:
                    try:
                        json.dump(row, f, ensure_ascii=False)
                        f.write("\n")
                    except:
                        continue
                    
        logger.info(f"Data successfully saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving to {output_path}: {e}")
        raise

def save_to_json(data: Union[Dict[str, Any], List[Dict[str, Any]]], file_name: str, mode="w") -> bool:
    """
    将数据保存为 JSON 文件。

    参数:
        data: 要保存的数据(单条字典或字典列表)。
        file_name: 输出 JSON 文件路径。
        mode: 写入模式，'w'表示覆盖，'a'表示追加。

    返回:
        成功返回 True，失败抛出异常。
    """
    try:
        dir_path = os.path.dirname(file_name)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        if mode == "a" and os.path.exists(file_name) and os.path.getsize(file_name) > 0:
            # 追加模式且文件存在
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                if isinstance(existing_data, list):
                    if isinstance(data, dict):
                        # 添加单个字典到现有列表
                        existing_data.append(data)
                    else:
                        # 添加字典列表到现有列表
                        existing_data.extend(data)
                    
                    # 写回完整数据
                    with open(file_name, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=4)
                else:
                    logger.warning(f"Existing JSON is not a list. Overwriting with new data.")
                    with open(file_name, 'w', encoding='utf-8') as f:
                        if isinstance(data, dict):
                            json.dump([data], f, ensure_ascii=False, indent=4)
                        else:
                            json.dump(data, f, ensure_ascii=False, indent=4)
            except json.JSONDecodeError:
                logger.warning(f"Existing JSON file is invalid. Overwriting with new data.")
                with open(file_name, 'w', encoding='utf-8') as f:
                    if isinstance(data, dict):
                        json.dump([data], f, ensure_ascii=False, indent=4)
                    else:
                        json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            # 新文件或覆盖模式
            with open(file_name, 'w', encoding='utf-8') as f:
                if isinstance(data, dict):
                    # 单条数据，但JSON文件应保持为列表格式
                    json.dump([data], f, ensure_ascii=False, indent=4)
                else:
                    # 列表数据，直接写入
                    json.dump(data, f, ensure_ascii=False, indent=4)
                    
        logger.info(f"Data successfully saved to {file_name}")
        return True
    except Exception as e:
        logger.error(f"Error saving to {file_name}: {e}")
        raise


def read_parquet(parquet_file_path: str) -> List[Dict]:
    try:
        df = pd.read_parquet(parquet_file_path)
        data = df.to_dict(orient="records")
        return data
    except Exception as e:
        return []
    

def read_parquet_and_print_row(parquet_file_path, num=10):
    """
    读取一个 Parquet 文件并打印出第一条记录 (第 1 行数据).
    
    Args:
        parquet_file_path (str): Parquet 文件的路径.
        
    Returns:
        None
    """
    try:
        # 使用 pandas 读取 Parquet 文件
        df = pd.read_parquet(parquet_file_path)
        
        # 检查是否为空
        if df.empty:
            print("Parquet 文件为空，没有数据！")
            return
        
        # 打印第一条记录
        for i in range(num):
            print(f"Parquet 文件第{i}条记录如下:")
            print(df.iloc[i].to_dict())
            print("----------------------------------------------------")

        print(f"Parquet 文件第{len(df)}条记录如下:")
        print(df.iloc[len(df)-1].to_dict())

        print(f"Parquet 文件一共有{len(df)}条数据")
        
    except Exception as e:
        print(f"读取 Parquet 文件时出错: {e}")

# 函数：保存数据为 Parquet 文件
def save_as_parquet(data, file_path):
    try:
        # 使用 pandas DataFrame 将 JSON 数据保存为 Parquet 格式
        df = pd.DataFrame(data)
        df.to_parquet(file_path, index=False)
        logger.info(f"Data successfully saved to {file_path}.")
    except Exception as e:
        logger.error(f"Error saving data to Parquet: {e}")


def convert_numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj


if __name__ == "__main__":
    read_parquet_and_print_row(r"/wind/aispace/train/source/dataset_files/e2e-dataset/v2.3.0002/data/e2e_agent_20250704_train_data_1/train.parquet")
# %%
