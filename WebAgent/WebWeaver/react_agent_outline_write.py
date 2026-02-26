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
from urllib.parse import unquote	
from utils.utils import read_jsonl, save_jsonl	
from prompt.user_prompt import USER_PROMPT_INST, USER_PROMPT_EXAMPLE	
from dashscope_api import call_dashscope


MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 40))	
print(f'Running with MAX_LLM_CALL_PER_RUN = {MAX_LLM_CALL_PER_RUN}')	
class MultiTurnReactAgentWrite(FnCallAgent):	
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
                if "Input data may contain inappropriate content" in error_str:	
                    return "Input data may contain inappropriate content"	
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
    def get_url2page(self, page_info):	
        '''	
        parse the page_info to url2page	
        '''	
        url2page = {}	
        for page in page_info:	
            ### 保持所有材料 url统一	
            if "summary" not in page or "goal" not in page or "evidence" not in page or "url" not in page:	
                continue	
            # if "evidence" in page:	
            evidence = page['evidence']	
            if isinstance(evidence, list):	
                for i in range(len(evidence)):	
                    if evidence[i]:	
                        evidence += str(evidence[i]) + "\n"	
                # evidence = "\n\n".join(evidence)	
            url2page[page['url']] = evidence	
        	
        return url2page	
    def get_url2id(self, page_info):	
        '''	
        get the url2id from page_ifo	
        '''	
        url2id = {}	
        for i in range(len(page_info)):	
            if "summary" not in page_info[i] or "goal" not in page_info[i] or "evidence" not in page_info[i] or "url" not in page_info[i]:	
                continue	
            url2id[page_info[i]["url"] ] = i + 1	
        return url2id	
    def get_url2summary(self, page_info):	
        '''	
        get the url2summary from page_info	
        '''	
        url2summary = {}	
        for i in range(len(page_info)):	
            if "summary" not in page_info[i] or "goal" not in page_info[i] or "evidence" not in page_info[i] or "url" not in page_info[i]:	
                continue	
            url2summary[page_info[i]["url"]] = "Goal: " + page_info[i]["goal"] + "\nSummary: " + page_info[i]["summary"]	
        return url2summary	
    def process_page_info(self, page_info):	
        '''	
        1. change the url to id	
        2. get the url2id	
        3. parse the page_info to url2page	
        '''	
        url2id = self.get_url2id(page_info)	
        url2page = self.get_url2page(page_info)	
        url2summary = self.get_url2summary(page_info)	
        	
        return url2id, url2page, url2summary	
        	
    def add_reference(self, url2id, output_content):	
        output_content += "\n\nReferences:\n"	
        for url, id in url2id.items():	
            output_content += f"[{id}]. {url}\n"	
        return output_content	
    def get_user_prompt(self, url2id, url2summary, url2page, query, outline=None):	
        user_prompt = f"""We have explored some subqueries related to the query "{query}". To write a comprehensive and informative article on this topic, we also provide url_id, title, and some statements with corresponding evidence related to the query and the subqueries. Please write a comprehensive and informative article for the query based on the provided information.	
The collected materials are as follows:	
<material>\n"""	
        for url, summary in url2summary.items():	
            if url not in url2page or url not in url2id:	
                print(f"[visit] url[{url}] not in url2page")	
            try:	
                if isinstance(summary, list):	
                    summary = "".join(summary)	
            except:	
                summary = str(summary)	
            page_content = url2page[url]	
            try:	
                if isinstance(page_content, list):	
                    page_content = "".join(page_content)	
            except:	
                page_content = str(page_content)	
                	
            user_prompt += f'''<id_{url2id[url]}>\n{summary}\n</id_{url2id[url]}>\n'''	

            if url not in url2id and url in outline:	
                print(f"[visit] url[{url}] not in url2id")	
            if outline:	
                decoded_url = unquote(url)	
                outline = outline.replace(url, f"<id_{url2id[url]}>").replace(decoded_url, f"<id_{url2id[url]}>")	
        user_prompt += f"""\n</material>\n"""	
	
        if outline:	
            user_prompt += f"You must strictly follow the outline and fill in the contents as detailed as possible.\n<outline>\n\n{outline}</outline>\n\n"	
        user_prompt += f"User query:\n{query}"	
        return user_prompt	
    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:	
        '''	
        url2page:  {url: page content}	
        url2id: {url: id}	
        '''	
        max_tokens = 100000
        max_output_tokens = 30000
        self.model=model	
        try:	
            question = data['item']['question']	
            	
        except: 	
            raw_msg = data['item']['messages'][1]["content"] 	
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 	
        	
        page_info = data['item']['page_info']	
        url2id, url2page, url2summary = self.process_page_info(page_info)	
        if "url2id" in data['item']:	
            url2id = data['item']["url2id"]	
        answer = data['item'].get('answer', '')  # for open-ended questions, answer is empty	
        user_prompt = self.get_user_prompt(url2id, url2summary, url2page, question, data['item']['outline'])	
        user_prompt = USER_PROMPT_INST + user_prompt + USER_PROMPT_EXAMPLE + question	
	
        messages = [{"role": "system", "content": self.system_message}, {"role": "user", "content": user_prompt}]	
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN	
        writer_prediction = ""	
        token_num_list = []	
        termination = "Failed"	
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
            if "Input data may contain inappropriate content" in content:	
                print("[Warning] Input data may contain inappropriate content")	
                writer_prediction = ""	
                break	
            if '<tool_response>' in content:	
                pos = content.find('<tool_response>')	
                content = content[:pos]	
            messages.append({"role": "assistant", "content": content.strip()})	
            if '<write>' in content and '</write>' not in content:	
                messages[-1]['content'] = f" Please write within tags <write> and </write>!"	
            if '<write>' in content and '</write>' in content:	
                ### remove the retrieved content of the previous step	
                tool_response_content = messages[-2]['content']	
                if "<tool_response>" in tool_response_content and "</tool_response>" in tool_response_content:	
                    mask_content = tool_response_content.split("<tool_response>")[1].split("</tool_response>")[0]	
                    tool_response_content = tool_response_content.replace(mask_content, "The page content for the previous section has been masked for saving the space.")	
	
                messages[-2]['content'] = tool_response_content	

                segments = re.findall(r'<write>(.*?)</write>', content, flags=re.S)	
                new_writting_content =  ''.join(segments)	
                writer_prediction += new_writting_content	
            if '<tool_call>' in content and '</tool_call>' in content:	
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]	
                try:	
                    tool_call = json.loads(tool_call)	
                    tool_name = tool_call.get('name', '')	
                    if "arguments" in tool_call:	
                        tool_args = tool_call.get('arguments', {})	
                    elif "url_id" in tool_call and "goal" in tool_call:	
                        tool_args = {}	
                        tool_args["url_id"] = tool_call.get("url_id", [])	
                        tool_args["goal"] = tool_call.get("goal", "")	
                    else:	
                        raise ValueError("Invalid tool call format")	
                    # tool_args = tool_call.get('arguments', {})	
                    tool_args["url2id"] = url2id	
                    if url2page:	
                        tool_args["url2page"] = url2page	
                    result = self._call_tool(tool_name, tool_args)	
                except:	
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'	
                result = "<tool_response>\n" + result + "\n</tool_response>" + "\nThink about what insight information can be got from the tool response, and then write starting with <write> and ending with </write>."	
                messages.append({"role": "user", "content": result})	
            if '<answer>' in content and '</answer>' in content or "<terminate>" in content:	
                termination = 'answer'	
                break	
              	
            if num_llm_calls_available <= 0 and '<answer>' not in content:	
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'	
            
            token_count = self.count_tokens(messages)	
            token_num_list.append(token_count)	
            print(f"round: {round}, token count: {token_count}")	
            if token_count > max_tokens:	
                print(f"token limit reached: {token_count} > {max_tokens}")	
                termination = 'token limit reached'	
                # prediction = 'No answer found.'	
                infer_messages = copy.deepcopy(messages)	
                result = {	
                    "question": question,	
                    "answer": answer,	
                    "infer_messages": infer_messages,	
                    "writer_prediction": "",	
                    "termination": termination,	
                    "url2id": url2id,	
                    "url2summary": url2summary,	
                    "url2page": url2page,	
                    "outline": data['item']['outline'],	
                    "output_token": 0,	
                    "token_num_list": token_num_list,	
                    "writer_reasoning_content": "",	
                    "writer_model": self.model,	
                }	
                return result	
        ### add reference	
        writer_prediction = self.add_reference(url2id, writer_prediction)	
        token_count = self.count_tokens(writer_prediction)	
        print(f"Number of writer_prediction: {token_count}")	
        if token_count == 0:	
            print("No answer found.")	
            print("Message: ", messages)	
        	
        infer_messages = copy.deepcopy(messages)	

        result = {	
            "question": question,	
            "answer": answer,	
            "infer_messages": infer_messages,	
            "writer_prediction": writer_prediction,	
            "termination": termination,	
            "url2id": url2id,	
            "url2summary": url2summary,	
            "url2page": url2page,	
            "outline": data['item']['outline'],	
            "output_token": token_count,	
            "token_num_list": token_num_list,	
            "writer_reasoning_content": "",	
            "writer_model": self.model,	
        }	
        return result