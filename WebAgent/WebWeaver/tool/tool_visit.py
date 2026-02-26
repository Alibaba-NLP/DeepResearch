import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Optional
import requests
from qwen_agent.tools.base import BaseTool, register_tool
import os 
from openai import OpenAI
import random
import time
import sys 
sys.path.append("../")
sys.path.append("./")

from topsdk.client import TopApiClient,TopException
from topsdk.defaultability.defaultability import Defaultability
from topsdk.defaultability.request.alibaba_aidata_aignite_application_run_request import AlibabaAidataAigniteApplicationRunAigniteApplicationExecuteReqDTO,AlibabaAidataAigniteApplicationRunRequest
from urllib.parse import urlparse, unquote
from transformers import AutoTokenizer
from transformers import AutoTokenizer
import tiktoken
try:
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
except:
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
os.environ['TOKENIZERS_PARALLELISM'] = "false"

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you need to maintain details as much as possible, output the **full original context** of the content as far as possible, it can be more than three paragraphs. You should maintain the important original tables and diagrams.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""


WEBCONTENT_MAXLENGTH = 24000
# Visit Tool (Using Jina Reader)
JINA_READER_URL_PREFIX = "https://r.jina.ai/"



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


@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    # The `description` tells the agent the functionality of this tool.
    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = {
    "type": "object",
    "properties": {
        "url": {
            "type": ["string", "array"],
            "items": {
                "type": "string"
                },
            "minItems": 1,
            "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
        },
        "goal": {
            "type": "string",
            "description": "The goal of the visit for webpage(s)."
        },
        "parse_type": {
            "type": "string",
            "enum": ["html", "pdf"],
            "default": "html",
            "description": "Specify whether to visit a HTML webpage or a PDF paper. Must be either 'html' or 'pdf'."
        }
    },
    "required": ["url", "goal"]
  }
    def __init__(self, cfg: Optional[dict] = None):
        super().__init__()
        
    @staticmethod
    def parse_json_output(raw):
        triple_match = re.search(r'```json\s*\n(.*?)\n```', raw, re.DOTALL)

        if triple_match:
            json_str = triple_match.group(1)
            try:
                return json5.loads(json_str)
            except Exception as e:
                print(f"Error parsing JSON5: {e}")
                return None
        else:
            try:
                return json5.loads(raw)
            except Exception as e:
                print(f"Error parsing raw string as JSON5: {e}")
                return None

    def call(self, params: Union[str, dict], **kwargs) -> Union[dict, List[dict]]:
        """
        Orchestrates visiting and processing web pages.
        If a single URL is provided, it's processed directly.
        If a list of URLs is provided, they are processed concurrently.
        """
        try:
            urls = params["url"]
            goal = params["goal"]
            parse_type = params.get("parse_type", 'html')
        except KeyError:
            return {
                "execution_status": "failed",
                "summary": """[Visit] Invalid request format. Required format:
                            {
                                "url": "string_or_array_of_strings",
                                "goal": "string",
                                "parse_type": "html" or "pdf" (optional)
                            }"""
            }

        if isinstance(urls, str):
            # Process a single URL
            response = self.readpage(urls, goal, parse_type)
        elif isinstance(urls, list):
            # Process a list of URLs concurrently
            response = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit all readpage tasks to the executor
                future_to_url = {executor.submit(self.readpage, u, goal, parse_type): u for u in urls}
                
                # Process results as they are completed
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        result = future.result()
                        response.append(result)
                    except Exception as e:
                        print(f"An exception occurred while processing {url}: {e}")
                        # Append a structured error message for the failed URL
                        response.append({
                            "url": url,
                            "goal": goal,
                            "execution_status": "failed",
                            "summary": f"Failed to process the page at {url}. Error: {e}"
                        })
        else:
            return {
                "execution_status": "failed",
                "summary": "[Visit] The 'url' parameter must be a string or a list of strings."
            }

        return response

    def call_server_align(self, msgs, max_retries=2): 

        pairs = [
                ["http://localhost:6002/v1/chat/completions", "EMPTY"],
                ["http://localhost:6002/v1/chat/completions", "EMPTY"],
                ["http://localhost:6002/v1/chat/completions", "EMPTY"]
            ]

        response = None
        for i in range(len(pairs)):
            url_llm = pairs[i][0]
            Authorization = pairs[i][1]

            headers = {
                'Content-Type': 'application/json',
                'Authorization': Authorization
            }

            payload = json.dumps({
                "model": os.environ.get("SUMMARY_MODEL_PATH"),
                "temperature": 0.7,
                "top_p": 0.8,
                "max_tokens": 20000,   ## 16000
                "presence_penalty": 1.5,
                "messages": msgs,
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            })
            try:
                response = requests.request("POST", url_llm, headers=headers, data=payload, timeout=240)
                data = response.json()
                if "error" in data and data["error"]["message"] == "Provider returned error":
                    print("error in message")
                    return ""
                
                content = data["choices"][0]["message"]["content"].replace("```json", "").replace("```", "")
                try:
                    content = json.loads(content)
                except: 
                    # left_pos, right_pos = content.index("{"), content.rindex("}")
                    left_pos = content.find("{")
                    right_pos = content.rfind("}")
                    if left_pos > 0 and right_pos > left_pos:
                        content = content[left_pos:right_pos+1]

                return content
            except Exception as e:
                print("Visit Tool Error", e)
                print("Use another EAS Service.")
                print(response.text)
        return ""
    
    def scraper_readpage(self, url: str) -> str:
        max_retries = 3

        scraper_reader = ScraperReader()
        for attempt in range(max_retries):
            page_content = scraper_reader.readpage(url)
            if page_content:
                return page_content
        return "[visit] Failed to read page."


    def html_readpage(self, url: str) -> str:
        max_attempts = 10
        for attempt in range(max_attempts):
            content = self.scraper_readpage(url)
            sevice = "scraper"
            
            print(sevice)
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                return content
        return "[visit] Failed to read page."

    def readpage(self, url: str, goal: str, parse_type: str = "html") -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """

        if parse_type == "html":
            content = self.html_readpage(url)
        else:
            return {"execution_status": "failed", "summary": "The provided webpage content could not be accessed."}

        # Check if we got valid content
        if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
            content = tokenizer.decode(tokenizer.encode(content)[:WEBCONTENT_MAXLENGTH])

            messages = [{"role":"user","content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
            parse_retry_times = 0
            raw = self.call_server_align(messages)

            parse_retry_times = 0
            while parse_retry_times < 3:
                try:
                    if isinstance(raw, str):
                        # raw = parse_json_output(raw)
                        raw = json.loads(raw)
                    raw["url"] = url
                    raw["goal"] = goal
                    break
                except:
                    raw = self.call_server_align(messages)
                    raw["url"] = url
                    raw["goal"] = goal
                    parse_retry_times += 1
            # parse failed
            if parse_retry_times >= 3:
                raw = {"execution_status": "failed", "summary": "The provided webpage content could not be accessed."}
            # parse successful
            else:
                raw["execution_status"] = "successful"
            return raw
              
        else:
            return {"execution_status": "failed", "summary": "The provided webpage content could not be accessed."}
