import json
import os
import requests
from urllib.parse import urlparse

def download_images_from_jsonl(jsonl_file, output_dir):
    """
    从JSONL文件中提取图片链接，下载图片并保存到本地指定文件夹。
    
    :param jsonl_file: 输入的jsonl文件路径
    :param output_dir: 图片保存的目标目录
    """
    # 如果输出目录不存在，创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开并读取JSONL文件
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            image_url = data.get('file_path')
            
            if image_url:
                # 获取图片文件名（从URL中提取）
                image_name = os.path.basename(urlparse(image_url).path)
                
                # 拼接成本地保存路径
                image_path = os.path.join(output_dir, image_name)
                
                # 下载图片
                try:
                    response = requests.get(image_url, stream=True)
                    if response.status_code == 200:
                        with open(image_path, 'wb') as img_file:
                            for chunk in response.iter_content(1024):
                                img_file.write(chunk)
                        print(f"成功下载图片: {image_name}")
                    else:
                        print(f"下载失败: {image_url} (状态码: {response.status_code})")
                except Exception as e:
                    print(f"下载图片时出错: {image_url} 错误: {e}")

# 使用示例
jsonl_file = 'vl_search_r1/eval_data/hle_50.jsonl'  # 你的jsonl文件路径
output_dir = 'scripts_eval/images/hle_50'  # 图片保存的目标文件夹
download_images_from_jsonl(jsonl_file, output_dir)
