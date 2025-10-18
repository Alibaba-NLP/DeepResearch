import json 
import requests 
from prompt import QUERY_SUMMARY_PROMPT, QUERY_SUMMARY_PROMPT_LAST
import os 
import re
import time 


RESUM_TOOL_NAME = os.getenv("RESUM_TOOL_NAME", "")
RESUM_TOOL_URL = os.getenv("RESUM_TOOL_URL", "") 


def call_resum_server(query, max_retries=10):
    data = {
        "model": RESUM_TOOL_NAME,
        "messages": [{ "role": "user",  "content": query}], 
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
    }
    header = {
        "Content-Type": "application/json", 
        "Authorization": ""
    }
    
    for _ in range(max_retries):
        try: 
            response = requests.post(RESUM_TOOL_URL, headers=header, json=data, timeout=360)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            # print(content)
            if content: 
                pattern = r'<think>.*?</think>' 
                content = re.sub(pattern, '', content, flags=re.DOTALL).strip()
                try:
                    content = content.split("<summary>")[1].split("</summary>")[0]
                except:
                    content = content 
                return "<summary>" + content + "</summary>"
            else: 
                return ""
        except Exception as e:
            time.sleep(0.2)
            if _ == max_retries - 1: 
                print(f"[Summary Tool] Server attempt {_+1}/{max_retries} failed: {str(e)}")
                return ""
    return ""


def summarize_conversation(question, recent_history_messages, last_summary, max_retries=10):
    recent_history_str = "\n".join([str(msg) for msg in recent_history_messages])
    
    if not last_summary:
        query_prompt = QUERY_SUMMARY_PROMPT.replace("{{{question}}}", question).replace("{{{recent_history_messages}}}", recent_history_str)
    else:
        query_prompt = QUERY_SUMMARY_PROMPT_LAST.replace("{{{question}}}", question).replace("{{{recent_history_messages}}}", recent_history_str).replace("{{{last_summary}}}", last_summary)
    
    response = call_resum_server(query_prompt, max_retries=max_retries)
    return response


###### Test Code ###### 
if __name__ == "__main__": 
    query = "Please give me a simple three-day travel plan for Kitakyushu, Japan."
    print(call_resum_server(query))
    print(summarize_conversation(query, [], ""))
