import os
import requests
import time
import random
import asyncio
import json
import re
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from uniform_eval.network.server.rm_model.judge_model import judge_model
from requests.exceptions import RequestException, Timeout
from urllib.parse import urlparse, unquote
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.tools.private.topsdk.client import TopApiClient,TopException
from qwen_agent.tools.private.topsdk.defaultability.defaultability import Defaultability
from qwen_agent.tools.private.topsdk.defaultability.request.alibaba_aidata_aignite_application_run_request import AlibabaAidataAigniteApplicationRunAigniteApplicationExecuteReqDTO,AlibabaAidataAigniteApplicationRunRequest

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
GPT_API_KEY = os.getenv("GPT_API_KEY", "")
JUDGE_MODEL_N = os.getenv("JUDGE_MODEL_N", "5")
JUDGE_SUMMARY_MODEL = os.getenv("JUDGE_SUMMARY_NAME", "Qwen2.5-72B-Instruct-SummaryModel-lw-32b")
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))



extractor_prompt = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""


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
      }
    },
    "required": ["url", "goal"]
  }
    # The `call` method is the main function of the tool.
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
            url = params["url"]
            goal = params["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"
        

        if isinstance(url, str):
            response = self.readpage(url, goal)
        else:
            response = []
            assert isinstance(url, List)
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self.readpage, u, goal): u for u in url}
                for future in as_completed(futures):
                    try:
                        response.append(future.result())
                    except Exception as e:
                        response.append(f"Error fetching {futures[future]}: {str(e)}")
            response = "\n=======\n".join(response)
        return response.strip()
    
    async def llm(self, messages):
        judge_name = JUDGE_SUMMARY_MODEL
        judge_func = judge_model(judge_name)
        json_schema = {
            "type": "object",
            "properties": {
                "rational": {"type": "string"},
                "evidence": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["rational", "evidence", "summary"]
        }
        # print("json_schema",json_schema)
        compute_llm_res = await judge_func(messages, top_k=-1, json_schema=json_schema)
        response = compute_llm_res.get('output', "")
        return response


    def jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina service.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        headers = {
            "Authorization": f"Bearer {JINA_API_KEY}",
        }
        max_retries = 3
        timeout = 10
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=timeout
                )
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content
                else:
                    print(response.text)
                    raise ValueError("jina readpage error")
            except Exception as e:
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."
                
        return "[visit] Failed to read page."

    def aidata_readpage(self, url: str, only_cache=False) -> str:
        """
        Read webpage content using Aidata service.
        
        Args:
            url: The URL to read
            
        Returns:
            str: The webpage content or error message
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                
                # Create Client
                client = TopApiClient(
                    appkey='34689424',
                    app_sercet="9a9c1bdda60cc53fda4a8cbb3343ed4d",
                    top_gateway_url='http://gw.api.taobao.com/router/rest',
                    verify_ssl=False
                )
                ability = Defaultability(client=client)

                # Create request DTO
                req_dto = AlibabaAidataAigniteApplicationRunAigniteApplicationExecuteReqDTO()
                req_dto.inputs = {
                    "urls": [{
                        "channel": "Google",
                        "tag": "",
                        "url": url,
                    }],
                    "cache": only_cache,
                }
                req_dto.application_id = 2627

                # Create and send request
                request = AlibabaAidataAigniteApplicationRunRequest()
                request.token = 'mMloh5kZyvOm8hgSfYx3izdmiXQehI'
                request.aignite_application_execute_req_dto = req_dto
                
                response = ability.alibaba_aidata_aignite_application_run(request)
                response_data = json.loads(response['data']['outputs'])["data"]['data'][0]
                
                content = response_data.get("content", "")
                title = response_data.get("title", "")
                
                if not content:
                    return "[visit] Empty content."
                    
                return f"{title}\n\n{content}" if title else content
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."
                    
        return "[visit] Failed to read page."

    def extract_wiki_keyword(self, url):
        parsed = urlparse(url)
        path = parsed.path
        keyword = path.split('/')[-1]
        keyword = unquote(keyword)
        keyword = keyword.replace(' ', '_').replace('+', '_')
        return keyword
    def query_wiki_dict_service(self,url):
        api_url = "http://1748815774914563.cn-beijing.pai-eas.aliyuncs.com/api/predict/wiki_dict_service_zh_en"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "YTg3NDg2YThhZDU0ZTI0NjZiMzY0NTQ2OTI5MzgyZGIwYTkyZTNiMA=="
        }
        keyword = self.extract_wiki_keyword(url)
        print(keyword)
        response = requests.post(api_url, headers=headers, data=keyword.encode("utf-8"))
        return response.text 
    def readpage(self, url: str, goal: str) -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        max_attempts = 10
        for attempt in range(max_attempts):
            if ('wikipedia.org' in url or 'wikipedia.com' in url or '') and attempt <=1:
                sevice = 'self-wiki'
                content = self.query_wiki_dict_service(url)
            elif attempt <=3:
                sevice = 'aidata-cache'
                content = self.aidata_readpage(url, only_cache=True)
            elif attempt <=6:
                sevice = "aidata-online"
                content = self.aidata_readpage(url, only_cache=False)
            else: 
                content = self.jina_readpage(url)
                sevice = "jina"
            print(sevice)

            # print(content)
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                content = content[:WEBCONTENT_MAXLENGTH]
                messages = [{"role":"user","content": extractor_prompt.format(webpage_content=content, goal=goal)}]
                parse_retry_times = 0
                raw = asyncio.run(self.llm(messages))

                # 如果网页超长，返回结果是 {\n 这种形式
                summary_retries = 3
                while len(raw) < 10 and summary_retries >= 0:
                    truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                    status_msg = (
                        f"[visit] Summary url[{url}] " 
                        f"attempt {3 - summary_retries + 1}/3, "
                        f"content length: {len(content)}, "
                        f"truncating to {truncate_length} chars"
                    ) if summary_retries > 0 else (
                        f"[visit] Summary url[{url}] failed after 3 attempts, "
                        f"final truncation to 25000 chars"
                    )
                    print(status_msg)
                    content = content[:truncate_length]
                    extraction_prompt = extractor_prompt.format(
                        webpage_content=content,
                        goal=goal
                    )
                    messages = [{"role": "user", "content": extraction_prompt}]
                    raw = asyncio.run(self.llm(messages))
                    summary_retries -= 1
                # 说明 raw 的长度大于10或者已经retry 超出了 
                parse_retry_times = 0
                while parse_retry_times < 3:
                    try:
                        # 尝试 parse json
                        raw = json.loads(raw)
                        break
                    except:
                        raw = asyncio.run(self.llm(messages))
                        parse_retry_times += 1
                # parse 失败
                if parse_retry_times >= 3:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                # parse 成功
                else:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + raw["evidence"] + "\n\n"
                    useful_information += "Summary: \n" + raw["summary"] + "\n\n"

                    summary_retries -= 1

                if len(useful_information) < 10 and summary_retries < 0:
                    print("[visit] Could not generate valid summary after maximum retries")
                    useful_information = "[visit] Failed to read page"
                return useful_information
                
            # If we're on the last attempt, return the last result
            if attempt == max_attempts - 1:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                return useful_information

    
if __name__ == "__main__":

    visit = Visit()
    print(visit.readpage("https://en.wikipedia.org/wiki/Japanese_submarine_I-19", "介绍一下日本一艘潜艇I-19"))
    print("===="*10)
    print(visit.readpage("https://www.novartis.com/news/media-releases/novartis-lutathera-significantly-reduced-risk-disease-progression-or-death-72-first-line-treatment-patients-advanced-gastroenteropancreatic-neuroendocrine-tumors", "介绍主要内容"))