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
from io import BytesIO

import re
import mimetypes
import secrets
from urllib.parse import urlparse
from datetime import datetime, timezone



# you can either hardcode this or you can set your own
WORKING_DOCS_DIR = Path(os.environ.get("WORKING_DOCS_DIR", "workingDocuments")).resolve()
WORKING_DOCS_DIR.mkdir(parents=True, exist_ok=True)



#print (EXTRACTOR_PROMPT)
# this is the code for importing everything and making sure everything works
# this is the code to make sure it is able to save every document 
'''
The idea with the saving is that we want to save the raw markdown file for everywebpage that we visit
the folder will is called workingDocuments

workingDocuments /
<run-folder>/
index.json  <----- this has a catalog of everything that is saved
<basename>__raw.<ext>     <------- this is the original bytes that we have saved (html/pdf.ect)
<basename>__extracted.md  <------- this is the cleaned markdown used for summarization
<basename>__summary.json  <------- this is the evidence  + summary that we extracted

'''



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



def local_readpage(url: str) -> str:
    # 1) Try trafilatura first (works great on HTML)
    try:
        html = trafilatura.fetch_url(url, no_ssl=True)
        if html and html.strip():
            md = trafilatura.extract(
                html, url=url, output_format="markdown",
                include_comments=False, include_links=False, include_images=False
            )
            if md and len(md.strip()) > 200:
                return md
    except Exception:
        pass

    # 2) Manual fetch + type-aware extraction
    try:
        r = requests.get(url, headers={"User-Agent": DEFAULT_UA}, timeout=LOCAL_REQUEST_TIMEOUT)
        r.raise_for_status()
        ct = (r.headers.get("content-type") or "").lower()
        lowurl = url.lower()

        # --- PDF ---
        if "pdf" in ct or lowurl.endswith(".pdf"):
            try:
                # optional dependency
                from pdfminer.high_level import extract_text
                text = extract_text(BytesIO(r.content)) or ""
                text = text.strip()
                if len(text) > 200:
                    return "# PDF Extracted Text\n\n" + text
            except Exception:
                # no pdfminer or parse failed
                return "[visit] Empty content."
            return "[visit] Empty content."

        # --- Excel (xlsx/xls) ---
        if "spreadsheetml" in ct or lowurl.endswith(".xlsx") or lowurl.endswith(".xls"):
            try:
                import pandas as pd
                
                xls = pd.ExcelFile(BytesIO(r.content))
                # take first sheet preview (up to 50 rows)
                df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
                preview = df.head(50).to_markdown(index=False)
                if len(preview.strip()) > 50:
                    return f"# Spreadsheet Preview (first 50 rows)\n\n{preview}"
            except Exception:
                return "[visit] Empty content."
            return "[visit] Empty content."

        # --- Plain HTML/text ---
        if "html" in ct or ct.startswith("text/"):
            from readability import Document
            doc = Document(r.text)
            cleaned_html = doc.summary(html_partial=True)
            md2 = trafilatura.extract(cleaned_html, output_format="markdown")
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
    
# we can add some helper functions right here

# this function just "slugifies" the text, which means it removes all the special characters from the 
# query
def _slugify(text: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "")).strip("-").lower()
    return s[:maxlen] or "q"

#gets time
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

#makes folder for the run 
def _make_run_folder(goal: str, question_id) -> Path:
    
    """
    If question_id is provided, use it as the folder name (slugified).
    Otherwise, create a unique run folder.
    """
    if question_id is not None and str(question_id).strip():
        folder = _slugify(str(question_id), 64)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        rand  = secrets.token_hex(3)
        folder = f"run-{stamp}-{rand}-{_slugify(goal, 28)}"
    qdir = WORKING_DOCS_DIR / folder
    qdir.mkdir(parents=True, exist_ok=True)
    return qdir


def _basename_for_url(url: str) -> str:
    p = urlparse(url)
    tail = Path(p.path).name or "index"
    return f"{_slugify(tail, 40)}-{secrets.token_hex(4)}"

def _guess_ext(url: str, content_type: str | None) -> str:
    p = urlparse(url)
    name = Path(p.path).name
    if "." in name and len(name.split(".")[-1]) <= 5:
        return "." + name.split(".")[-1].lower()
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            return ext
    return ".html"

def _save_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def _update_index(qdir: Path, record: dict) -> None:
    idx = qdir / "index.json"
    data = []
    if idx.exists():
        try:
            data = json.loads(idx.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    data.append(record)
    _save_text(idx, json.dumps(data, ensure_ascii=False, indent=2))
    
def _download_raw(url: str) -> dict:
    """Best-effort raw fetch for archival."""
    try:
        r = requests.get(url, headers={"User-Agent": DEFAULT_UA}, timeout=LOCAL_REQUEST_TIMEOUT, allow_redirects=True)
        return {
            "ok": r.ok,
            "status": r.status_code,
            "content_type": r.headers.get("content-type", ""),
            "final_url": str(r.url),
            "content_bytes": (r.content if r.ok else b""),
        }
    except Exception as e:
        return {"ok": False, "status": None, "content_type": "", "final_url": url, "error": str(e)}



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
            #new
            "question_id":{
                "type":"string",
                "description":"If provided, all saved files go under this folder name"
            }
        },
        "required": ["url", "goal"],
    }
    
#after the tool has been registered, we can actually write the final code for calling the pipeline

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
            question_id = params.get("question_id")
        except Exception:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        # ensure string if provided (you already had this)
        question_id = str(question_id) if question_id is not None else None
        qdir = _make_run_folder(goal, question_id)
        print(f"[visit] saving to {qdir}")

        start = time.time()

        if isinstance(url, str):
            # single URL
            try:
                return self._read_and_summarize(url, goal, qdir)
            except Exception as e:
                return f"[visit] Failed on {url}: {e}"

        # list of URLs
        out: List[str] = []
        for u in url:
            try:
                if time.time() - start > VISIT_SERVER_TIMEOUT:
                    out.append(_fallback_payload(u, goal))
                else:
                    out.append(self._read_and_summarize(u, goal, qdir))
            except Exception as e:
                out.append(f"[visit] Failed on {u}: {e}")

        return "\n===========\n".join(out)

    
    def _read_and_summarize(self , url: str, goal: str, qdir: Path) -> str:
        # 0) archive RAW bytes up front
        
        
        # raw_info = _download_raw(url)
        # base = _basename_for_url(url)
        # ts = _utc_now_iso()

        # raw_path = None
        # if raw_info.get("ok") and raw_info.get("content_bytes"):
        #     ext = _guess_ext(url, raw_info.get("content_type"))
        #     raw_path = qdir / f"{base}__raw{ext}"
        #     try:
        #         _save_bytes(raw_path, raw_info["content_bytes"])
        #     except Exception:
        #         raw_path = None
        
        base = _basename_for_url(url)
        ts = _utc_now_iso()
        raw_info = {"status": None, "content_type": None, "final_url": url}
        raw_path = None

        # 1) extract page
        content = local_readpage(url)

        # 1a) save extracted markdown (if any)
        extracted_path = None
        if _ok(content):
            extracted_path = qdir / f"{base}__extracted.md"
            try:
                header = f"# Extracted content\n\nURL: {url}\nSaved: {ts}\n\n---\n\n"
                _save_text(extracted_path, header + content)
            except Exception:
                extracted_path = None

        # 2) bail early if extraction failed (still keep raw + index)
        if not _ok(content):
            _update_index(qdir, {
                "url": url,
                "saved_at": ts,
                "goal": goal,
                "raw_file": (str(raw_path) if raw_path else None),
                "extracted_file": None,
                "summary_file": None,
                "status": "fallback",
                "http_status": raw_info.get("status"),
                "content_type": raw_info.get("content_type"),
                "final_url": raw_info.get("final_url", url),
            })
            return _fallback_payload(url, goal)

        # 3) summarize (your existing logic)
        content = truncate_to_tokens(content, 95_000)
        messages = _build_messages(content, goal)

        raw = _call_summary_server(messages, max_retries=int(os.getenv("VISIT_SERVER_MAX_RETRIES", 2)))
        summary_retries = 3
        while (not raw or len(raw) < 10) and summary_retries >= 0:
            trunc_len = int(0.7 * len(content)) if summary_retries > 0 else 25_000
            print(f"[visit] Summary url[{url}] attempt {3 - summary_retries + 1}/3; content_len={len(content)} → truncating to {trunc_len}")
            content = content[:trunc_len]
            messages = _build_messages(content, goal)
            raw = _call_summary_server(messages, max_retries=1)
            summary_retries -= 1

        # parse JSON
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
            _update_index(qdir, {
                "url": url, "saved_at": ts, "goal": goal,
                "raw_file": (str(raw_path) if raw_path else None),
                "extracted_file": (str(extracted_path) if extracted_path else None),
                "summary_file": None,
                "status": "parse_failed",
                "http_status": raw_info.get("status"),
                "content_type": raw_info.get("content_type"),
                "final_url": raw_info.get("final_url", url),
            })
            return _fallback_payload(url, goal)

        evidence = str(obj.get("evidence", ""))
        summary  = str(obj.get("summary", ""))


        # # 4) save summary JSON
        # summary_path = qdir / f"{base}__summary.json"
        # try:
        #     _save_text(summary_path, json.dumps(
        #         {"url": url, "goal": goal, "evidence": evidence, "summary": summary},
        #         ensure_ascii=False, indent=2
        #     ))
        # except Exception:
        #     summary_path = None

        # 5) update index
        _update_index(qdir, {
            "url": url,
            "saved_at": ts,
            "goal": goal,
            "raw_file": (str(raw_path) if raw_path else None),
            "extracted_file": (str(extracted_path) if extracted_path else None),
            "summary_file": (str(summary_path) if summary_path else None),
            "status": "ok",
            "http_status": raw_info.get("status"),
            "content_type": raw_info.get("content_type"),
            "final_url": raw_info.get("final_url", url),
        })

        # 6) return same shape the agent expects
        result = (
            f"The useful information in {url} for user goal {goal} as follows: \n\n"
            f"Evidence in page: \n{evidence}\n\n"
            f"Summary: \n{summary}\n\n"
        )
        if len(result) < 10 and summary_retries < 0:
            print("[visit] Could not generate valid summary after maximum retries")
            return "[visit] Failed to read page"
        return result


'''


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
        
'''

