import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
# from prompt import EXTRACTOR_PROMPT 
import os 
from openai import OpenAI
import random
from topsdk.client import TopApiClient,TopException
from topsdk.defaultability.defaultability import Defaultability
from topsdk.defaultability.request.alibaba_aidata_aignite_application_run_request import AlibabaAidataAigniteApplicationRunAigniteApplicationExecuteReqDTO,AlibabaAidataAigniteApplicationRunRequest
from urllib.parse import urlparse, unquote
import time 

import litellm
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
import tiktoken
from transformers import AutoTokenizer 


def count_tokens(messages, model="gpt-4o"):
        try: 
            tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path) 
        except Exception as e: 
            tokenizer = tiktoken.encoding_for_model(model)
        
        if isinstance(messages, list):
            full_message = [Message(**x) for x in messages]
        elif isinstance(messages, str):
            return len(tokenizer.encode(messages))
        else:
            raise ValueError("Invalid message type")
        full_prompt = build_text_completion_prompt(full_message, allow_special=True)
        
        return len(tokenizer.encode(full_prompt))


class ScraperReader:
    """Scraper内容读取器"""
    
    def __init__(self):
        self.scraper_key = os.getenv("SCRAPER_API_KEY")
        self.scraper_country_code = "us"
        self.scraper_time_out = 40
    def readpage(self, url: str) -> str:
        payload = {'api_key': self.scraper_key, 
                    'url': url, 
                    'output_format': 'markdown', 
                    'country_code': self.scraper_country_code }
        max_retries = 2
        for attempt in range(max_retries):
            try:
                r = requests.get('https://api.scraperapi.com/', params=payload, timeout=self.scraper_time_out)
                content = r.text
                return content
            except requests.exceptions.Timeout:
                # 超时情况下返回默认内容
                content = "[visit] Request timed out after {} seconds. Using default content.".format(self.scraper_time_out)
            except Exception as e:
                content ="[visit] Failed to read page."
        return content


@register_tool('retrieve', allow_overwrite=True)
class Retrieve(BaseTool):
    """
    Retrieve webpages associated with the given id and return a summary.
    The mapping between URL and id is supplied via the url2id dictionary.
    """
    name = "retrieve"
    description = "Read the webpage(s) whose id matches the given id and return the summary."

    parameters = {
    "type": "object",
    "properties": {
        "url_id": {
            "type": ["string", "array"],
            "items": {
                "type": "string"
                },
            "minItems": 1,
            "description": "The URL ID(s) of the webpage(s) to visit. Can be a single URL ID or an array of URL IDs."
      },
      "goal": {
            "type": "string",
            "description": "The goal of the visit for webpage(s)."
      }
    },
    "required": ["url_id", "goal"]
  }

    # ---------------------------
    # Main entry point
    # ---------------------------
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        params must contain:
          • url2id : dict mapping URL → id
          • url_id     : target id
          • goal   : reading goal (passed through to readpage)
          if "url2page" is present, it is used to retrieve the page content
        """
        max_tokens = 20000
        try:
            url2id = params["url2id"]
            target_id = params["url_id"]
            if isinstance(target_id, list):
                target_id = [int(id_.replace("id_", "")) for id_ in target_id]
            elif isinstance(target_id, str):
                target_id = [int(target_id.replace("id_", ""))]
            else:
                raise ValueError("Invalid url_id type")
            goal = params["goal"]
            assert isinstance(url2id, dict)
        except Exception:
            return "[retrieve] Invalid request format: must contain 'url2id', 'url_id', and 'goal'."

        # Find every URL whose associated id equals the requested id
        urls = [url for url, _id in url2id.items() if _id in target_id]

        if not urls:
            return f"[retrieve] id '{target_id}' not found in url2id."

        # One URL → call readpage directly
        # if len(urls) == 1:
        #     return self.readpage(urls[0], goal).strip()

        
        # Several URLs share the same id → read them in parallel
        url_page: List[dict] = []
        non_existed_urls = []
        url2page = None
        if "url2page" in params:
            url2page = params["url2page"]
            for url in urls:
                if url in url2page:
                    if isinstance(url2page[url], list):
                        page_content = ""
                        for item in url2page[url]:
                            page_content += str(item)
                        # page_content = "".join(url2page[url])
                    else:
                        page_content = str(url2page[url])
                    url_page.append({url: page_content})
                else:
                    non_existed_urls.append(url)
        else:
            non_existed_urls = urls

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(self.readpage, u, goal): u for u in non_existed_urls}
            for future in as_completed(futures):
                try:
                    url_page.append(future.result())
                except Exception as e:
                    url_page.append(f"[retrieve] Error fetching {futures[future]}: {str(e)}")

        output_str = ""
        for item in url_page:
            for key, value in item.items():
                output_str += f"<id_{url2id[key]}>\nContent:\n{value}\n</id_{url2id[key]}>\n\n"
        try:
            if count_tokens(output_str) > max_tokens:
                output_str = output_str[:max_tokens]
        except Exception as e:
            print(f"[retrieve] Error counting tokens: {str(e)}")

        return output_str
    
    def call_server(self, msgs, max_tries=10):
        # 设置 OpenAI 的 API 密钥和 API 基础 URL 使用 vLLM 的 API 服务器。
        openai_api_key = "EMPTY"
        openai_api_base = "http://127.0.0.1:6002/v1"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(
                    # TODO: [Important] If you change the summary model, you need to change the model path correspondingly.
                    model=os.getenv("SUMMARY_MODEL_PATH"),
                    messages=msgs,
                    temperature=0.7
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string 
                        left = content.find('{')
                        right = content.rfind('}') 
                        if left != -1 and right != -1 and left <= right: 
                            content = content[left:right+1]
                    return content
            except Exception as e:
                # print(e)
                if attempt == (max_tries - 1):
                    return ""
                continue

    def scraper_readpage(self, url: str) -> str:
        max_retries = 3

        scraper_reader = ScraperReader()
        for attempt in range(max_retries):
            page_content = scraper_reader.readpage(url)
            if page_content:
                return page_content
        return "[visit] Failed to read page."

    def readpage(self, url: str, goal=None, max_token=15000) -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        if isinstance(url, list):
            url = url[0]
            
        max_attempts = 10
        for attempt in range(max_attempts):
            content = self.scraper_readpage(url)
            sevice = "scraper"
            print(sevice, "length", len(content))

            # Check if we got valid content
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                content = content[:int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))]

                return {url: content}
            else:
                return {url: ""}

