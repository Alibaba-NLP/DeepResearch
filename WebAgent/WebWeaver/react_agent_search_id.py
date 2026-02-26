import json
import os
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
import litellm
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI
import tiktoken
from transformers import AutoTokenizer 
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
import traceback
import copy
import re
import ast
from dashscope_api import call_dashscope

from prompt.search_user_prompt_id_3 import SEARCH_USER_PROMPT


MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 40))
print(f'Running with MAX_LLM_CALL_PER_RUN = {MAX_LLM_CALL_PER_RUN}')


class MultiTurnReactAgentSearch(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 **kwargs):
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description,
                         files=files,
                         **kwargs)
        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        self.page_info = []

    def call_server(self, msgs, max_tries=10):
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        openai_api_base = os.environ.get("OPENAI_API_BASE")

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                )
                content = chat_response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                if attempt == (max_tries - 1):
                    print(f"SGLang server error {e}")
                    return f"SGLang server error"
                error_str = traceback.format_exc()
                print("SGLang server error trace back", error_str)

                continue
        
        return "SGLang server empty response"

    def count_tokens(self, messages, model="gpt-4o"):
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

    def parse_visit_result(self, content_json_list, url2id):
        output_str = "<material>"
        for content_json in content_json_list:
            execution_status = content_json['execution_status']
            if execution_status == "successful":
                if "url" not in content_json or "goal" not in content_json or "summary" not in content_json:
                    continue
                url = content_json['url']
                idx = url2id[url]
                goal = content_json['goal']
                useful_information = f"\n<id_{idx}>\n"
                useful_information += "Summary: \n" + content_json['summary'] + "\n"
                useful_information += f"\n</id_{idx}>\n"
                output_str += useful_information
        
        output_str += "</material>"
        
        return output_str

    def save_page_info(self, page_info, content_json):
        url_list = []
        for item in page_info:
            if "url" not in item or "goal" not in item or "summary" not in item or "evidence" not in item:
                continue
            url_list.append(item['url'])
        for new_content in content_json:
            if "url" not in new_content or "goal" not in new_content or "summary" not in new_content or "evidence" not in new_content:
                continue
            if new_content['url'] not in url_list:
                page_info.append(new_content)
        return page_info
    def save_url2id(self, url2id, content_json):
        for new_content in content_json:
            if "url" not in new_content or "goal" not in new_content or "summary" not in new_content or "evidence" not in new_content:
                continue
            if new_content["url"] in url2id:
                continue
            url2id[new_content["url"]] = len(url2id) + 1
        
        return url2id

    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        self.model=model
        page_info = []
        url2id = {}
        search_num = 0
        max_output_tokens = 40000
        try:
            question = data['item']['question']
            answer = data['item'].get('answer', '')
            
        except: 
            raw_msg = data['item']['messages'][1]["content"] 
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 
        
        user_prompt = SEARCH_USER_PROMPT
        self.user_prompt = user_prompt
        self.user_prompt = self.user_prompt + question + "\nPlease start to think and tool call."
        messages = [{"role": "system", "content": self.system_message}, {"role": "user", "content": self.user_prompt}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        outline = ""
        outline_first_part = ""
        token_num_list = []
        round = 0
        while num_llm_calls_available > 0:
            round += 1
            num_llm_calls_available -= 1
            if self.llm_generate_cfg["if_infer"]:
                content = call_dashscope(
                model=model,
                messages=messages,
                stop=["\n<tool_response>", "<tool_response>"],
                temperature=self.llm_generate_cfg.get('temperature', 0.6),
                top_p=self.llm_generate_cfg.get('top_p', 0.95),
                max_tokens=max_output_tokens)
            else:
                content = self.call_server(messages)
            print(f'Round {round}: {content}')
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            messages.append({"role": "assistant", "content": content.strip()})
            if "<write_outline>" in content and "</write_outline>" not in content:
                print("Warning: <write_outline> tag is not closed")
                outline_first_part = messages[-1]["content"].split('<write_outline>')[1]
            if "<write_outline>" not in content and "</write_outline>" in content:
                if len(outline_first_part) > 100: 
                    outline = outline_first_part + messages[-1]["content"].split('</write_outline>')[0]
                    messages[-1]["content"] = "<write_outline>" +  outline + "</write_outline>"
                else:
                    messages[-1]["content"] =  " Please generate outline within tags <write_outline> and </write_outline>!"
            if '<write_outline>' in content and '</write_outline>' in content:
                segments = re.findall(r'<write_outline>(.*?)</write_outline>', content, flags=re.S)
                new_writting_content =  ''.join(segments)
                outline = new_writting_content

                messages[-1]["content"] += "\nTry to make the outline more comprehensive and ensure the citation for each subsection."

            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                try:
                    tool_call = json.loads(tool_call)
                    tool_name = tool_call.get('name', '')
                    if "arguments" in tool_call:
                        tool_args = tool_call.get('arguments', {})
                        tool_args["page_info"] = page_info
                    elif "goal" in tool_call:
                        tool_args = {}
                        tool_args["goal"] = tool_call.get("goal", "")
                        tool_args["page_info"] = page_info
                    else:
                        raise ValueError("Invalid tool call format")
                    result = self._call_tool(tool_name, tool_args)
                    result = json.loads(result)
                    search_num += 1
                except:
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                
                if isinstance(result, list):
                    if len(result) > 0:
                        ### save page content
                        page_info = self.save_page_info(page_info, result)
                        url2id = self.save_url2id(url2id, result)
                        # page_info.extend(result)
                        result = "<tool_response>\n" + self.parse_visit_result(result, url2id) + "\n</tool_response>"
                    else:
                        result = "<tool_response>\n" + "No useful information found." + "\n</tool_response>"
                else:
                    result = "<tool_response>\n" + result + "\n</tool_response>"

                messages.append({"role": "user", "content": result})
            if '<answer>' in content and '</answer>' in content or "<terminate>" in content:
                termination = 'answer'
                break
              
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            max_tokens = 84000 #31 * 1024 - 500
            token_count = self.count_tokens(messages)
            token_num_list.append(token_count)
            print(f"round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"token limit reached: {token_count} > {max_tokens}")
                termination = 'token limit reached'

                result = {
                    "question": question,
                    "answer": answer,
                    "search_messages": messages,
                    "outline": outline,
                    "outline_token": 0,
                    "termination": termination,
                    "output_token": 0,
                    "token_num_list": token_num_list,
                    "writer_reasoning_content": "",
                    "writer_model": self.model,
                    "page_info": page_info,
                    "url2id": url2id,
                    "search_num": search_num,
                }
                return result

        token_count = self.count_tokens(outline)
        print(f"Number of outline: {token_count}")
        if token_count == 0:
            print("No answer found.")
            print("Message: ", messages)
        
        result = {
            "question": question,
            "answer": answer,
            "search_messages": messages,
            "outline": outline,
            "outline_token": token_count,
            "termination": termination,
            "output_token": 0,
            "token_num_list": token_num_list,
            "writer_reasoning_content": "",
            "writer_model": self.model,
            "page_info": page_info,
            "url2id": url2id,
            "search_num": search_num,
        }
        return result
