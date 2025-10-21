import json
import os
import time
from typing import List, Union
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

import requests
import tiktoken
from openai import OpenAI
from prompt import EXTRACTOR_PROMPT
from qwen_agent.tools.base import BaseTool, register_tool

import trafilatura
import html2text
from readability import Document

#print (EXTRACTOR_PROMPT)
# this is the code for importing everything and making sure everything works

VISIT_SERVER_TIMEOUT = 200
WEBCONTENT_MAXLENGHT = 150000
# the limits to how long our content can be, techinally should keep this in the env file but it doesn't really matter

DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123 Safari/537.36"
)
#this is the default user agent that we are using, so websites dont think we are bots

READPAGE_USE_JS = False
LOCAL_REQUEST_TIMEOUT = 25
# we can toggle readpage w JS true or false, it is false right now, and the timeout for the requests is 25 seconds

if __name__ == "__main__":
    '''    
    print("Imports OK. Prompt starts with:")
    print(EXTRACTOR_PROMPT.splitlines()[0])
    print("UA:", DEFAULT_UA[:60] + "…")
    print("READPAGE_USE_JS=", READPAGE_USE_JS, " LOCAL_REQUEST_TIMEOUT=", LOCAL_REQUEST_TIMEOUT)
    '''
    


# this is the code that truncates to tokens, inspired from the one in tool_visit.py
def truncate_to_tokens (text: str, max_tokens: int = 95_000) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    if len(toks) <= max_tokens:
        return text
    return enc.decode(toks[:max_tokens])


# this is the main function to get the information from the webpage

def local_readpage(url:str) -> str:
    #uses trafilatura to fetch the webpage and extract the main content as markdown
    #returns markdown on success or failed to read page error on failure
    
    try:
        html = trafilatura.fetch_url(url , no_ssl = True)
        if not html or not html.strip():
            return "[visit] Failed to read page."

        md = trafilatura.extract(
            html,
            url = url,
            output_format = "markdown",
            include_comments = False,
            include_links = False,
            include_images = False,
        )
                    
        if md and len(md.strip()) > 200:
            return md
        
    except Exception:
        pass
    #if this fails, then you go to the next thing
    
    # that was just the code for trafilatura, now we can also implement another library to give us some fallback incase something doesnt work.
    try:
        r = requests.get(url, headers={"User-Agent" : DEFAULT_UA}, timeout = LOCAL_REQUEST_TIMEOUT)
        r.raise_for_status()
        doc = Document(r.text)
        cleaned_html = doc.summary(html_partial=True)
        
        md2 = trafilatura.extract(cleaned_html, output_format ="markdown")
        if not md2:
            md2 = html2text.html2text(cleaned_html)
        
        if md2 and len(md2.strip()) > 200:
            return md2
    except Exception:
       pass
        
    return "[visit] Failed to read page"


# some helper functions for the output

def _ok(content: str) -> bool:
    return (
        bool(content)
        and not content.startswith("[visit] Failed")
        and content.strip() != "[visit] Empty content."
        and not content.startswith("[document_parser]")
    )
    
def _fallback_payload(url: str, goal: str) -> str:
    return (
        f"The useful information in {url} for user goal {goal} as follows: \n\n"
        "Evidence in page: \nThe provided webpage content could not be accessed. "
        "Please check the URL or file format.\n\n"
        "Summary: \nThe webpage content could not be processed, and therefore, "
        "no information is available.\n\n"
    )
 
def _build_messages(content: str, goal: str):
    return [{
        "role": "user",
        "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
    }]   



#this is the main function for actually getting the summaries
def _call_summary_server(messages, max_retries: int = 3) -> str:
    #this is the final summarizing thing, which we return the output of.
    
    client = OpenAI(
        api_key = os.environ.get("API_KEY"),
        base_url = os.environ.get("API_BASE")
    )
    
    model = os.environ.get("SUMMARY_MODEL_NAME", "")
    
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model = model,
                messages = messages,
                temperature = 0.7
            )
            content = resp.choices[0].message.content or ""
            try:
                json.loads(content)
                return content
            except Exception:
                left, right = content.find("{"), content.rfind("}")
                if 0 <= left < right:
                    return content[left:right+1]
                return content
            
        except Exception:
            if attempt == max_retries -1:
                return ""
            continue
    
    return ""


#Now we need to register the tool into orchestrator like it except

@register_tool("visit", allow_overwrite = True)
class Visit(BaseTool):
    name = "visit"
    description = " Visit webpages(s) and return the summary of the content"
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit."
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s)."
            },
        },
        "required": ["url", "goal"],
    }
    
#after the tool has been registered, we can actually write the final code for calling the pipeline

    def call(self, params: Union[str,dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except Exception:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"
        
        start = time.time()
        
        # this is the case that we only have one url, if we have more urls, then we need to do the case for multiple urls    
        if isinstance(url, str):
            return self._read_and_summarize(url,goal)        
        
        assert isinstance(url,list)
        chunks = []
        for innerUrl in url:
            if time.time() - start > 900: # this is the fifteen minute cap on the summarization, we can change it to more later
                chunks.append(_fallback_payload(innerUrl,goal))
            else:
                try:
                    chunks.append(self._read_and_summarize(innerUrl,goal))
                except Exception as e:
                    chunks.append(f"Error fetching {innerUrl}: {e}")
        return "\n===========\n".join(chunks)

    def _read_and_summarize(self , url: str, goal: str) -> str:
        content = local_readpage(url)
        #this is incase the output is not good
        if not _ok(content):
            return _fallback_payload(url,goal)
        
        content = truncate_to_tokens(content, 95_000)
        messages = _build_messages(content, goal)
        
        raw = _call_summary_server(messages,max_retries=int(os.getenv("VISIT_SERVER_MAX_RETRIES",2)))
        summary_retries = 3
        while (not raw or len(raw) < 10) and summary_retries >= 0:
            trunc_len = int(0.7 * len(content)) if summary_retries > 0 else 25_000
            print(f"[visit] Summary url[{url}] attempt {3 - summary_retries + 1}/3; "
                    f"content_len={len(content)} → truncating to {trunc_len}")
            content = content[:trunc_len]
            messages = _build_messages(content, goal)
            raw = _call_summary_server(messages, max_retries=1)
            summary_retries -= 1

    # parse JSON (re-ask a couple times if needed)
        if isinstance(raw, str):
            raw = raw.replace("```json", "").replace("```", "").strip()

        parse_retry_times = 0
        while parse_retry_times < 2:
            try:
                obj = json.loads(raw)
                break
            except Exception:
                raw = _call_summary_server(messages, max_retries=1)
                if isinstance(raw, str):
                    raw = raw.replace("```json", "").replace("```", "").strip()
                parse_retry_times += 1
        else:
            return _fallback_payload(url, goal)

        evidence = str(obj.get("evidence", ""))
        summary  = str(obj.get("summary", ""))

        result = (
            f"The useful information in {url} for user goal {goal} as follows: \n\n"
            f"Evidence in page: \n{evidence}\n\n"
            f"Summary: \n{summary}\n\n"
        )
        if len(result) < 10 and summary_retries < 0:
            print("[visit] Could not generate valid summary after maximum retries")
            return "[visit] Failed to read page"
        return result
        

            


