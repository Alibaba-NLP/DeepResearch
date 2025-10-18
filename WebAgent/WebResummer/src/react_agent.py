import json
import os
from typing import Dict, List, Optional, Union
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI
import tiktoken
from transformers import AutoTokenizer 
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from summary_utils import summarize_conversation


MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 60))
print(f'Running with MAX_LLM_CALL_PER_RUN = {MAX_LLM_CALL_PER_RUN}')
RESUM = os.getenv('RESUM', 'False').lower() == 'true'
print(f'ReSum Mode: {RESUM}')
MAX_CONTEXT = int(os.getenv('MAX_CONTEXT', 32))
print(f"Maximum Context: {MAX_CONTEXT}k")


class MultiTurnReactAgent(FnCallAgent):
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
        # Set OpenAI API key and base URL using vLLM API server
        openai_api_key = "EMPTY"
        openai_api_base = "http://127.0.0.1:6001/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    stop=["\n<tool_responseF", "<tool_response>"],
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
                continue
        
        return "SGLang server empty response"

    def count_tokens(self, messages, model="gpt-4o"):
        try: 
            tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path) 
        except Exception as e: 
            tokenizer = tiktoken.encoding_for_model(model)
        
        full_message = [Message(**x) for x in messages]
        full_prompt = build_text_completion_prompt(full_message, allow_special=True)
        
        return len(tokenizer.encode(full_prompt))

    def _run(self, data: str, model: str, summary_iteration: int, **kwargs) -> List[List[Message]]:
        self.model = model
        question = data['item']['question']
        answer = data['item']['answer']

        messages = [
            {"role": "system", "content": self.system_message}, 
            {"role": "user", "content": question}
        ]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        last_summary, full_trajectory = None, messages.copy() 
        while num_llm_calls_available > 0:
            round += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages)
            print(f'round {round}: {content}')

            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]

            messages.append({"role": "assistant", "content": content.strip()})
            full_trajectory.append({"role": "assistant", "content": content.strip()}) 

            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                try:
                    tool_call = json.loads(tool_call)
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('arguments', {})
                    result = self._call_tool(tool_name, tool_args)
                    print(f"Tool call {tool_name} invocation success with length {len(result)}")
                except Exception as e:
                    print(f"Tool call error: {e}")
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                result = "<tool_response>" + result + "</tool_response>"
                messages.append({"role": "user", "content": result})
                full_trajectory.append({"role": "user", "content": result}) 

            elif '<answer>' in content and '</answer>' in content:
                answer_content = content.split('<answer>')[1].split('</answer>')[0].strip()
                if len(answer_content):
                    termination = 'answer'
                    prediction = answer_content 
                    break
      
            max_tokens = MAX_CONTEXT * 1024 - 1000
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            should_summarize = ((RESUM and token_count >= max_tokens * 0.9) or round % summary_iteration == 0) and num_llm_calls_available 
            if should_summarize: 
                recent_messages = messages[2:].copy() 

                try:
                    summary_response = summarize_conversation(question, recent_messages, last_summary)
                    print(f"[Summary Tool] ReSum-Tool-30B Invocation success (len: {len(summary_response)}): {summary_response}")
                except Exception as e: 
                    print(f"[Summary Tool] ReSum-Tool-30B Invocation failed: {e}")
                    summary_response = "" 
                
                if summary_response:  
                    last_summary = summary_response  
                    new_observation = "Question: " + question + "\nBelow is a summary of the previous conversation. This summary condenses key information from earlier steps, so please consider it carefully. Assess whether the summary provides enough information to answer the question and use it as the basis for further reasoning and information gathering to answer the question.\n" \
                                     + "Summary: " + summary_response + "\n"
                    messages = [
                        {"role": "system", "content": self.system_message}, 
                        {"role": "user", "content": new_observation}
                    ]
                    full_trajectory.append({"role": "user", "content": new_observation})  
                    token_count = self.count_tokens(messages) 
                    print(f"round {round}, token count after summary: {token_count}")
            
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            if token_count > max_tokens:
                print(f"Token count exceeds limit: {token_count} > {max_tokens}")
                
                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop invoking tools and, based on all the information above, think again and provide what you consider the most likely answer in the following format: <think> your final thinking </think>\n <answer> your answer </answer>"
                content = self.call_server(messages)

                messages.append({"role": "assistant", "content": content.strip()})
                full_trajectory.append({"role": "assistant", "content": content.strip()}) 

                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'
                result = {
                    "question": question,
                    "answer": answer,
                    "rollout_id": data['rollout_id'],
                    "messages": full_trajectory,
                    "prediction": prediction,
                    "termination": termination
                }
                return result

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        result = {
            "question": question,
            "answer": answer,
            "rollout_id": data['rollout_id'],
            "messages": full_trajectory,
            "prediction": prediction,
            "termination": termination
        }
        return result
