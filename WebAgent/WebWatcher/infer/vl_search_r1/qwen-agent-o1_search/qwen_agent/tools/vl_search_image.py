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
import logging
import subprocess
from urllib.parse import quote
import hashlib
import base64
from serpapi import GoogleSearch

from qwen_agent.tools.private.cache_utils import JSONLCache
from qwen_agent.tools.base import BaseTool, register_tool

try:
    IMG_SEARCH_KEY = os.getenv('IMG_SEARCH_KEY')
except Exception:
    print("Warning: IMG_SEARCH_KEY not set.")

try:
    accessKeyId = os.getenv('OSS_KEY_ID','')
    accessKeySecret = os.getenv('OSS_KEY_SECRET','')
except Exception:
    print("Warning: OSS_KEY_ID or OSS_KEY_SECRET not set.")

SEARCH_STRATEGY =  os.getenv('SEARCH_STRATEGY',"incremental")
enable_search_cache = os.getenv('VL_IMG_SEARCH_ENABLE_CACHE', 'false').lower() in ('y', 'yes', 't', 'true', '1', 'on')
cache = JSONLCache(os.path.join(os.path.dirname(__file__), "vl_search/search_cache_image.jsonl"))

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


@register_tool("VLSearchImage", allow_overwrite=True)
class VLSearchImage(BaseTool): 
    name = "VLSearchImage"
    description = "Utilize the vl search engine to retrieve relevant information based on the input image."
    parameters = {
        "type": "object",
        "properties": {
            "image_urls": {
                "type": "array",
                "items": {"type": "string", "description": "The search image url."},
                "description": "The list of search image url.",
            }
        },
        "required": ["image_urls"],   
    }

    @search_cache_decorator
    def search_image_by_image_url(self, download_url, img_save_path=None, byte=True, retry_attempt=10, timeout=30):

        params = {
        "engine": "google_reverse_image",
        "image_url": download_url,
        "api_key": IMG_SEARCH_KEY,
        }

        results = {}
        
        for attempt in range(retry_attempt):
            try:
                search = GoogleSearch(params)
                rst = search.get_dict()
                docs = rst.get('image_results', [])
                
                search_data = [
                    {
                        "image_path": item.get("favicon", ""),
                        "snippet": item.get("snippet", ""),
                        "url": item.get("link", ""),
                        # "source": item.get("source", "")
                    }
                    for item in docs
                ]

                # return self.download_upload(search_data[:5], img_save_path, byte)
                return search_data[:5]
            except requests.exceptions.Timeout:
                print(f"请求超时（尝试 {attempt + 1}/{retry_attempt}）: {download_url}")
            except Exception as e:
                print(f"Error searching image via URL: {str(e)}. Retrying...")
        
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
                content = response.content if method == 'requests' else response

                if method == 'curl':
                    # ============== 新增：解析base64数据 ==============
                    try:
                        response_json = json.loads(content.decode('utf-8'))
                        if isinstance(response_json, dict) and response_json.get('data').get('type') == 'b64':
                            image_b64 = response_json['data']['file_b64']
                            image_bytes = base64.b64decode(image_b64)
                            return image_bytes  # 返回解码后的二进制图片
                        else:
                            raise ValueError("Invalid response format")
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        return content
                        
                if method == 'requests' and response.status_code != 200:
                    raise Exception(f"HTTP status code: {response.status_code}")
                    
                return content
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
        target_path = f'zhili.zl/qwenvl_rft/image/{image_name}'

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

    def download_upload(self, search_data, img_save_path, byte, max_retries=5, max_time=10):

        upload_data = []
        for item in tqdm(search_data, desc="download"):
            image_path = item.get("image_path", '')
            # breakpoint()
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

                # ========= 新增：保存到本地 =========
                if not byte:
                    # 生成唯一文件名（基于URL的MD5）
                    url_md5 = hashlib.md5(item['url'].encode()).hexdigest()
                    filename = f"{url_md5}.jpeg"
                    save_path = os.path.join(img_save_path, filename)
                    
                    # 写入文件
                    try:
                        with open(save_path, 'wb') as f:
                            f.write(image_data)
                    
                        image_url = save_path
                    except Exception as e:
                        logging.error(f"Save image failed: {str(e)}")
                        
                # ========= 结束新增 =========
                else:
                    image_url_ = self.upload(image_data, byte=byte)
                    if image_url_:
                        image_url = self.clean_image_url(image_url_)
                        # item['image_path'] = image_url
            else:
                logging.error(f"All download attempts failed for {image_path}")
                return [{"caption": item['snippet'], "image_url":None}]

            data_dict = {
                "image_url": image_url,
                "caption": item['snippet'],
                # "source": item['source'],
                "ori_url": item['url'],
                "requests_success": True
            }
        
            upload_data.append(data_dict)

        return upload_data

    def parse_image_search_result(self, search_result):
        search_images = [item.get('image_url', '') for item in search_result]
        search_texts = [item.get('snippet', '') for item in search_result]
        search_urls = [item.get('url', '') for item in search_result]

        if all(img == '' for img in search_images) or search_images[0] == None:
            if all(text == '' for text in search_texts) or search_texts[0] == None:
                return ([],[],[])
            return ([], search_texts, search_urls)

        return (
            search_images,
            search_texts,
            search_urls
        )

    def call(self, items, messages=None, img_save_path=None, byte=False, **kwargs):
        try:
            assert isinstance(items, dict)
            image_urls = items.get("image_urls") or items.get("images")
        except:
            return "[VL Search] Invalid request format: Input must be a DICT containing 'image_urls' field"

        if len(image_urls) == 0:
            return "[VL Search] Empty search images."
        

        ctxs = []

        for image_url in image_urls:
            search_result = self.search_image_by_image_url(image_url, img_save_path, byte)
            
            search_images, search_texts, search_urls = self.parse_image_search_result(search_result)
           

            if len(search_images) > 0:
                lines = []
                for img, txt, url in zip(search_images, search_texts, search_urls):
                    entry = f"Image: {img}, Text: {txt}, Webpage Url: {url}"
                    lines.append(entry)
                contents = "```\n" + '\n\n'.join(lines) + "\n```"

            elif len(search_images) == 0 and len(search_texts) > 0:
                lines = []
                for txt, url in zip(search_texts, search_urls):
                    entry = f"Text: {txt}, Webpage Url: {url}"
                    lines.append(entry)
                contents = "```\n" + '\n\n'.join(lines) + "\n```"
            
            else:
                contents = f"[VLSearchImage] No image found: {search_images}, {search_texts}, {search_result}"
            
            ctxs.append(contents)
            
        print(ctxs[0])            
        return ctxs[0]      # 只支持一次搜一张图片！



if __name__ == '__main__':

    image_url1 = ["https://mitalinlp.oss-cn-hangzhou.aliyuncs.com/rallm/deep_research_vl_image/hle_image/1.jpg"]
    image_url2 = ["https://th.bing.com/th/id/OIP.sCzpnScidvdeSPyy-0Rd2wHaDt?o=7rm=3&rs=1&pid=ImgDetMain&o=7&rm=3"]
    
    results = VLSearchImage().call({"images": ["https://mitalinlp.oss-cn-hangzhou.aliyuncs.com/rallm/deep_research_vl_image/hle_image/290.jpg"] })
    print(results)