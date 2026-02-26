import os
import json
import requests
from typing import Union, List
from qwen_agent.tools.base import BaseTool, register_tool
from concurrent.futures import ThreadPoolExecutor

SELECT_URL_PROMPT = """Please process the following search results and user goal to extract all relevant urls:

## **Search Results** 
{search_results}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate all the relevant **urls** directly related to the user's goal according to the titles, snippets, and url format in the search content.
2. **Extract relevant urls for goals**: Identify and extract all the **relevant urls** from the content, you never miss any important urls, output the **full original urls** of the content as far as possible. Ensure the urls are valid and complete.

**Final Output Format using JSON format has "rational", "urls" feilds**
Output example:
```json{{
"rational": "The rational is...",
"urls": ["url1", "url2", "url3"]
}}
'''
"""


@register_tool('select_related_url')
class SelectURL(BaseTool):

    def call_server_align(self, msgs, max_retries=2):
        url_llm = "http://localhost:6002/v1/chat/completions"

        headers = {
            'Content-Type': 'application/json',
            'Authorization': "EMPTY"
        }
        payload = json.dumps({
            "model": os.environ.get("SUMMARY_MODEL_PATH"),
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 80000,   ## 16000
            "presence_penalty": 1.5,
            "messages": msgs,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        })
        
        response = None
        for _ in range(max_retries):
            try:
                response = requests.request("POST", url_llm, headers=headers, data=payload, timeout=240)
                data = response.json()
                if "error" in data and data["error"]["message"] == "Provider returned error":
                    print("error in message")
                    return ""
                
                content = data["choices"][0]["message"]["content"].replace("```json", "").replace("```", "")
                try:
                    content = json.loads(content)
                    return content
                except: 
                    print("Error happens in url selection. Retry again.")
                    
                    left_pos, right_pos = content.index("{"), content.rindex("}")
                    if left_pos > 0 and right_pos > left_pos:
                        content = content[left_pos:right_pos+1]
                    return json.loads(content)
                return content
            except Exception as e:
                print("URL Select Error, retry again", e)
                continue
                # print(response.text)
        return ""

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
            search_results = params["search_results"]
            goal = params["goal"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        messages = [{"role":"user","content": SELECT_URL_PROMPT.format(search_results=search_results, goal=goal)}]
        select_url_json = self.call_server_align(messages)
        
        return select_url_json


if __name__ == "__main__":
    # tool = Search()
    search_content = f"""
    ## Web Results
1. [The Hong Kong University of Science and Technology ｜World's ...](https://hkust.edu.hk/)

Hong Kong University of Science and Technology (HKUST) is a world-class international research university, leading research excellence in science, ...

2. [Hong Kong University of Science and Technology - Wikipedia](https://en.wikipedia.org/wiki/Hong_Kong_University_of_Science_and_Technology)

HKUST is a public research university in Sai Kung District, New Territories, Hong Kong. Founded in 1991, it was the territory's third institution to be granted ...

3. [The Hong Kong University of Science and Technology](https://www.topuniversities.com/universities/hong-kong-university-science-technology)

We offer interdisciplinary programs across science, technology, engineering, business, and humanities, complemented by a strong foundation in AI, data science, ...

4. [Hong Kong University of Science and Technology (HKUST)](https://experience.cornell.edu/opportunities/hong-kong-university-science-and-technology-hkust)

HKUST is a leading research university with English courses, guaranteed housing, and a diverse student body. It is a Cornell Global Hub partner with a 150-acre ...

5. [The Hong Kong University of Science and Technology (Guangzhou)](https://www.hkust-gz.edu.cn/)

HKUST(GZ) will explore new frontiers in cross-disciplinary education and innovate pedagogies. By doing so, HKUST(GZ) will play a constructive role in ...

6. [The Hong Kong University of Science and Technology (@hkust)](https://www.instagram.com/hkust/?hl=en)

The Hong Kong University of Science and Technology (HKUST) Official Account. Tag us with #HKUST to let us repost your content on our channels!
"""
    # print(tool.call({"query": ["When Do LLMs Help With Node Classification? A Comprehensive Analysis", "CUHK DB Group"]}))
    select_url = SelectURL()
    # print(select_url.call({"search_results": search_content, "goal": "introduce the university HKUST"}))
    print(select_url.call({"search_results": "hello", "goal": "introduce the university HKUST"}))
