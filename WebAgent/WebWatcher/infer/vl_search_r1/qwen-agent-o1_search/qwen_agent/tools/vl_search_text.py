import os
import argparse
import requests
import json
from tqdm import tqdm
from io import BytesIO
import re
import random
import string
import oss2
from pathlib import Path
from functools import wraps
import atexit
from dotenv import load_dotenv
import logging
import subprocess
import time
from urllib.parse import quote
import hashlib
import base64

from qwen_agent.tools.private.cache_utils import JSONLCache
from qwen_agent.tools.base import BaseTool, register_tool


load_dotenv()
TEXT_SEARCH_KEY = os.getenv('TEXT_SEARCH_KEY',"")
accessKeyId = os.getenv('OSS_KEY_ID','')
accessKeySecret = os.getenv('OSS_KEY_SECRET','')
SEARCH_STRATEGY =  os.getenv('SEARCH_STRATEGY',"incremental")


enable_search_cache = os.getenv('VL_TEXT_SEARCH_ENABLE_CACHE', 'false').lower() in ('y', 'yes', 't', 'true', '1', 'on')
cache = JSONLCache(os.path.join(os.path.dirname(__file__), "vl_search/search_cache_text.jsonl"))

if enable_search_cache:
    atexit.register(cache.update_cache)
def search_cache_decorator(func):
    @wraps(func)
    def wrapper(self, img_url, *args, **kwargs):
       
        if enable_search_cache:
            key = str(img_url)
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            result = func(self, img_url, *args, **kwargs)
            cache.set(key, result)
        else:
            result = func(self, img_url, *args, **kwargs)
        return result
    return wrapper


@register_tool("VLSearchText", allow_overwrite=True)
class VLSearchText(BaseTool):     
  
    name = "VLSearchText"
    description = "Utilize the vl search engine to retrieve relevant information based on the input image."
    parameters = {
        "type": "object",
        "properties": {
        "queries": {
            "type": "array",
            "items": {
            "type": "string",
            "description": "The search query."
            },
            "description": "The list of search queries."
        }
        },
        "required": [
        "queries"
        ]
    }

    @search_cache_decorator
    def search_image_by_text(self, gold_query, max_retry: int = 10, timeout: int = 30):
        url = 'http://101.37.167.147/gw/v1/api/msearch-sp/qwen-search'
        headers = {"Host": "pre-nlp-cn-hangzhou.aliyuncs.com", "Authorization": f"Bearer {TEXT_SEARCH_KEY}", "Content-Type": "application/json"}
        template = {
            "rid": "",
            "scene": "dolphin_search_google_image",
            "type": "image",
            "start": -1,
            "uq": gold_query,
            "debug": False,
            "fields": [],
            "page": 1,
            "rows": 10,
            "customConfigInfo": {
                "qpSpellcheck": False,
                "knnWithScript": False,
                "qpTokenized": False,
                "qpEmbedding": False,
                "qpQueryRewrite": False,
                "qpTermsWeight": False,
                "inspection": False, 
            },
            "rankModelInfo": {
                "default": {
                    "features": [
                        {"name": "static_value", "field": "_weather_score", "weights": 1.0},
                        {
                            "name": "qwen-rerank",
                            "fields": ["hostname", "title", "snippet", "timestamp_format"],
                            "weights": 1,
                            "threshold": -50,
                            "max_length": 512,
                            "rank_size": 100,
                            "norm": False,
                        },
                    ],
                    "aggregate_algo": "weight_avg",
                }
            },
            "headers": {"__d_head_qto": 500000},
        }

        resp = ""
       
        for _ in range(max_retry):
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(template), timeout=timeout)
                rst = json.loads(resp.text)
                #TODO：
                # rst = {"data":{"originalOutput":{"organic":[{"imageUrl":"https://mitalinlp.oss-cn-hangzhou.aliyuncs.com/rallm/mm_data/vfreshqa_datasets_v2/Freshqa_en_zh/Freshqa_en_extracted_images/image_1.jpeg","title":'test',"link":'test',"source":'test'}]}}}
                code = resp.status_code
                docs = rst["data"]["docs"]
                # docs = rst.get("data", {}).get("originalOutput", {}).get("organic", [])
                assert len(docs) != 0, template['rid'] + "搜索为空，重试"

                search_data = [
                    {
                        "image_path": item.get("image", ""),
                        "snippet": item.get("snippet", ""),
                        # "url": item.get("url", ""),
                        # "source": item.get("sc_name", "")
                    }
                    for item in docs
                ]
                return self.download_upload(search_data[:10])
            except requests.exceptions.Timeout:
                print(f"请求超时（尝试 {_ + 1}/{max_retry}:{queries}")
            except Exception as e:
                print(f"Error searching text: {str(e)}. Code = {code}, Retrying...")
                time.sleep(1 * (_ + 1))
                continue
        return []

    def save_image_detail(self, data_dict, save_dir):
        with open(save_dir, 'a') as df:
            df.write(json.dumps(data_dict, ensure_ascii=False) + '\n')
    def try_download(self, url, method, timeout, max_retries):
        api_base = 'http://ep-0jlie2a8a5bb014097a8.epsrv-0jl2cvnsbvm4j1kacs4u.cn-wulanchabu.privatelink.aliyuncs.com:8001/download_by_url?url='
        constructed_url = api_base + quote(url)

        if method == 'requests':
            fetch_function = lambda: requests.get(url, timeout=timeout)
        elif method == 'wget':
            fetch_function = lambda: subprocess.run(["wget", "-O-", url], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout).stdout
        elif method == 'curl':
            fetch_function = lambda: subprocess.run(
                ["curl", "--location", "--silent", "--fail", constructed_url],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout
            ).stdout
        else:
            raise ValueError("Invalid download method")
        
        for attempt in range(max_retries):
            try:
                response = fetch_function()
                if method == 'requests' and response.status_code != 200:
                    raise Exception(f"HTTP status code: {response.status_code}")
                return response.content if method == 'requests' else response
            except Exception as e:
                logging.warning(f"{method.capitalize()} failed for {url} on attempt {attempt + 1}/{max_retries}: {e}")
        return None

    def clean_image_url(self, url):
        jpeg_index = url.find('.jpeg')
        if jpeg_index != -1:
            return url[:jpeg_index + 5]
        else:
            return url

    def generate_random_string(self, length=10):
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choice(characters) for _ in range(length))
        return random_string

    def upload(self, image, byte=False):
        
        image_name = f'{self.generate_random_string()}.jpeg'
        target_path = f'rallm/deep_research_vl_image/hle_eval_image_gpt4o_25_0325_v2/{image_name}'

        if byte:
            image_bytes = BytesIO(image)
            image_size = image_bytes.getbuffer().nbytes
        else:
            try:
                image_size = os.path.getsize(image)
            except OSError:
                return None

        if image_size <= 1024:
            return None

        auth = oss2.Auth(accessKeyId, accessKeySecret)
        bucket = oss2.Bucket(auth, 'http://oss-cn-hangzhou.aliyuncs.com', 'mitalinlp')
        
        if byte:
            bucket.put_object(target_path, image_bytes.getvalue())
        else:
            bucket.put_object_from_file(target_path, image)

        file_url = bucket.sign_url('GET', target_path, 360000)
        return file_url


    def download_upload(self, search_data, max_retries=5, max_time=10):
        
        upload_data = []
        for item in tqdm(search_data, desc="download"):
            image_path = item.get("image_path", '')
            if image_path.startswith('x-raw-image:///'):
                continue

            image_data = self.try_download(image_path, 'curl', max_time, max_retries)

            if image_data is None:
                logging.info(f"Using wget to download {image_path} (curl failed)")
                image_data = self.try_download(image_path, 'wget', max_time, max_retries)

                if image_data is None:
                    logging.error(f"Using request to download {image_path} (curl and wget failed)")
                    image_data = self.try_download(image_path, 'request', max_time, max_retries)

            if image_data is not None:
                image_url_ = self.upload(image_data, byte=True)
                if image_url_:
                    image_url = self.clean_image_url(image_url_)
                    item['image_path'] = image_url

                    data_dict = {
                        "image_url": image_url,
                        "caption": item['snippet'],
                        # "source": item['source'],
                        # "ori_url": item['url'],
                        "requests_success": True
                    }
                
                    upload_data.append(data_dict)
                    
                else:
                    continue
            else:
                logging.error(f"All download attempts failed for {image_path}")
                return [{"caption": item['snippet'], "image_url":None}]

        return upload_data

    def parse_image_search_result(self, search_result):
        search_images = [item.get('image_url', '') for item in search_result]
        search_texts = [item.get('caption', '') for item in search_result]

        if all(img == '' for img in search_images) or search_images[0] == None:
            if all(text == '' for text in search_texts) or search_texts[0] == None:
                return ([],[])
            return ([],search_texts)

        return (
            search_images,
            search_texts
        )

    def call(self, items, **kwargs):
        # 只支持一个query
        try:
            assert isinstance(items, dict)
            gold_queries = items["queries"]
        except:
            return "[VL Search] Invalid request format: Input must be a DICT containing 'queries' field"

        if len(gold_queries) == 0:
            return "[VL Search] Empty search queries."

        ctxs = []
        # if SEARCH_STRATEGY == "incremental":
        #     for q in gold_queries:
        #         ctxs += self.search_image_by_text(q, **kwargs)
        # else:
        #     ctxs = self.search_image_by_text(gold_queries, **kwargs)
        for q in gold_queries:

            search_result = self.search_image_by_text(q)
            
            search_images, search_texts = self.parse_image_search_result(search_result)
           

            if len(search_images) > 0:
                contents = "Contents of retrieved images:\n"
                for img, txt in zip(search_images, search_texts):
                    contents += f"Image: {img}, Text: {txt}\n"

            elif len(search_images) == 0 and len(search_texts) > 0:
                contents = "Captions of retrieved images:\n"
                for txt in search_texts:
                    contents += f"Text: {txt}\n"
            
            else:
                contents = "No image found."
            
            ctxs.append(contents)
            
        return ctxs[0]



if __name__ == '__main__':

    image_url1 = ["特朗普最近一次被刺杀是什么时候？"]
    
    results = VLSearchText().call({"queries": image_url1})
    print(results)
