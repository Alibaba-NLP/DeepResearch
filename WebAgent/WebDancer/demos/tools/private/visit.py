import os
import json
import requests
from openai import OpenAI
from qwen_agent.tools.base import BaseTool, register_tool
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_MULTIQUERY_NUM = os.getenv("MAX_MULTIQUERY_NUM", 3)
JINA_API_KEY = os.getenv("JINA_API_KEY")
DASHSCOPE_KEY = os.getenv('DASHSCOPE_API_KEY')

extractor_prompt = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning**: Locate the **specific sections/data** directly related to the user's goal within the webpage content.
2. **Key Extraction**: Identify and extract the **most relevant information** from the content, you never miss any important information
3. **Summary Output**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.


**Final Output Format using JSON format**:
{{
  "rational": "string",
  "evidence": "string",
  "summary": "string",
}}
"""

def jina_readpage(url: str) -> str:
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

@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
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
    def call(self, params: str, **kwargs) -> str:
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
            with ThreadPoolExecutor(max_workers=MAX_MULTIQUERY_NUM) as executor:
                futures = {executor.submit(self.readpage, u, goal): u for u in url}
                for future in as_completed(futures):
                    try:
                        response.append(future.result())
                    except Exception as e:
                        response.append(f"Error fetching {futures[future]}: {str(e)}")
            response = "\n=======\n".join(response)
        return response.strip()
    

    def llm(self, messages):
        client = OpenAI(api_key=DASHSCOPE_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        max_retries = 10
        for attempt in range(max_retries):
            response = client.chat.completions.create(
                model="qwen2.5-72b-instruct", 
                messages=messages,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        return ""


    def readpage(self, url: str, goal: str) -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        max_attempts = 3
        for attempt in range(max_attempts):
            content = jina_readpage(url)
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                messages = [{"role":"user","content": extractor_prompt.format(webpage_content=content, goal=goal)}]
                raw = self.llm(messages).replace("```json\n", "").replace("\n```", "").strip()
                parse_retry_times = 0
                try:
                    rawjson = json.loads(raw)
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + rawjson.get("evidence", "The provided webpage content is not in json.") + "\n\n"
                    useful_information += "Summary: \n" + rawjson.get("summary", "The webpage content is not processed in json") + "\n\n"
                    if useful_information != "":
                        print("useful_information:",useful_information)
                        return useful_information
                except Exception as e:
                    print("[visit] Failed to parse json:", e)
                    raw = self.llm(messages).replace("```json\n", "").replace("\n```", "")
                    parse_retry_times += 1

                
            # If we're on the last attempt, return the last result
            if attempt == max_attempts - 1:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                return useful_information



if __name__ == '__main__':
    print(Visit().readpage("https://github.com/callanwu/WebWalker-1?tab=readme-ov-file", "who are you?"))