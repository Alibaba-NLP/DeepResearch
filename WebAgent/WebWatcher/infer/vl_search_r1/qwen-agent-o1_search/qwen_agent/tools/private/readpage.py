import os
import json
import random
import requests
import asyncio
from requests.exceptions import RequestException, Timeout
from qwen_agent.tools.base import BaseTool, register_tool
from uniform_eval.network.server.rm_model.judge_model import judge_model
from qwen_agent.tools.private.topsdk.client import TopApiClient,TopException
from qwen_agent.tools.private.topsdk.defaultability.defaultability import Defaultability
from qwen_agent.tools.private.topsdk.defaultability.request.alibaba_aidata_aignite_application_run_request import AlibabaAidataAigniteApplicationRunAigniteApplicationExecuteReqDTO,AlibabaAidataAigniteApplicationRunRequest

JINA_API_KEY = os.getenv("JINA_API_KEY")
JUDGE_MODEL = os.getenv("JUDGE_NAME")
READPAGE_SOURCE = os.getenv('READPAGE_SOURCE', 'aidata')


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

def jina_readpage(url):
    assert JINA_API_KEY, "Please set the JINA_API_KEY environment variable."
    headers = {
            "Authorization": f"Bearer {JINA_API_KEY}"
        }
    response = requests.get("https://r.jina.ai/"+url, headers=headers)
    if response.status_code != 200:
        print(str(url)+"\n"+str(response.status_code))
    return response.text


def aidata_readpage(url):
    only_cache = os.getenv('NLP_WEB_SEARCH_ONLY_CACHE', 'true').lower() in ('y', 'yes', 't', 'true', '1', 'on')
    # create Client
    client = TopApiClient(appkey='34689424', app_sercet="9a9c1bdda60cc53fda4a8cbb3343ed4d",top_gateway_url='http://gw.api.taobao.com/router/rest',verify_ssl=False)
    ability = Defaultability(client=client)

    # create domain
    alibabaAidataAigniteApplicationRunAigniteApplicationExecuteReqDTO = AlibabaAidataAigniteApplicationRunAigniteApplicationExecuteReqDTO()

    alibabaAidataAigniteApplicationRunAigniteApplicationExecuteReqDTO.inputs = \
        {
            "urls":[
                {
                    "channel":"Google",
                    "tag":"",
                    "url":url,
                }
            ],
            "cache": only_cache,
        }
    alibabaAidataAigniteApplicationRunAigniteApplicationExecuteReqDTO.application_id = 2627

    # create request
    request = AlibabaAidataAigniteApplicationRunRequest()
    request.token = 'mMloh5kZyvOm8hgSfYx3izdmiXQehI'
    request.aignite_application_execute_req_dto = alibabaAidataAigniteApplicationRunAigniteApplicationExecuteReqDTO
    response = ability.alibaba_aidata_aignite_application_run(request)
    response = json.loads(response['data']['outputs'])["data"]['data'][0]
    return "[visit] Empty content." if response.get("content","") == "" else response.get("title","") + "\n\n" + response.get("content","")


# @register_tool('visit', allow_overwrite=True)
# class Visit(BaseTool):
#     # The `description` tells the agent the functionality of this tool.
#     name = 'visit'
#     description = 'Visit a webpage and return the content of webpage.'
#     # The `parameters` tell the agent what input parameters the tool has.
#     parameters = {
#         "type": "object",
#         "properties": {
#             "url": {
#                 "type": "string",
#                 "description": "The url you want to explore.",
#             }
#         },
#         "required": ["url"],
#     }
#     # The `call` method is the main function of the tool.
#     def call(self, params: Union[str, dict], **kwargs) -> str:
#         try:
#             params = self._verify_json_format_args(params)
#             url = params['url']
#         except:
#             return "[visit] Invalid request format: Input must be a JSON object"
#         if len(url) == 0:
#             return "[visit] Empty url."

#         if READPAGE_SOURCE == 'jina':
#             response = jina_readpage(url)
#         else:
#             response = aidata_readpage(url)
#         return response



@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    # The `description` tells the agent the functionality of this tool.
    name = 'visit'
    description = 'Visit a webpage and return the summary of webpage.'
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = [
        {
            'name': 'url',
            'type': 'string',
            'description': 'the url you want to explore',
            'required': True
        },
        {
            'name': 'goal',
            'type': 'string',
            'description': 'the goal of the visit for the webpage',
            'required': True
        }
    ]
    # The `call` method is the main function of the tool.
    def call(self, params: str, **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
            url = params["url"]
            goal = params["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"
        response = self.readpage(url, goal)
        return response
    
    async def llm(self, messages):
        json_schema = {
            "type": "object",
            "properties": {
                "rational": {"type": "string"},
                "evidence": {"type": "string"},
                "summary": {"type": "string"}
            },
            "required": ["rational", "evidence", "summary"]
        }
        assert JUDGE_MODEL is not None, "JUDGE_MODEL is not set"
        judge_name_all = JUDGE_MODEL.split(",")
        max_retries = 10
        for attempt in range(max_retries):
            judge_name = random.choice(judge_name_all)
            print(judge_name)
            judge_func = judge_model(judge_name)
            compute_llm_res = await judge_func(messages, top_k=-1, json_schema=json_schema)
            response = compute_llm_res.get('output', "")
            if response == "":
                print(f"Summary model is down!! Retry {attempt}", compute_llm_res)
                continue
            return response
        return ""


    def readpage(self, url, goal):
        max_retries = 10
        if url is None or type(url)!=str:
            return "[visit] Url is None! Failed to read page."
        for attempt in range(max_retries):
            try:
                if READPAGE_SOURCE == 'jina':
                    webpage_content = jina_readpage(url)
                else:
                    webpage_content = aidata_readpage(url)
                messages = [{"role":"user","content": extractor_prompt.format(webpage_content=webpage_content, goal=goal)}]
                useful_information = asyncio.run(self.llm(messages))
                return useful_information
            except (RequestException, Timeout, Exception) as e:
                print(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."
        return "[visit] Failed to read page."



if __name__ == '__main__':
    print(aidata_readpage("https://www.baidu.com"))