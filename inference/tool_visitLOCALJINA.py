# tool_visitLOCALNEW.py
import json
import os
import time
from typing import List, Union, Optional
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

import requests
import tiktoken
from openai import OpenAI
from prompt import EXTRACTOR_PROMPT
from qwen_agent.tools.base import BaseTool, register_tool

import re
import mimetypes
import secrets
from urllib.parse import urlparse
from datetime import datetime, timezone
from io import BytesIO  # kept for parity; optional

# ---- optional: DB integration (safely no-op if missing) ----
try:
    from db_min import save_document  # returns numeric doc id
except Exception:
    def save_document(content: str) -> Optional[int]:
        return None

# ====== CONFIG ======
VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))
DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123 Safari/537.36"
)

# Jina Reader token (optional; r.jina.ai works unauthenticated for most pages)
JINA_API_KEYS = os.getenv("JINA_API_KEYS", "").strip()

# Where we save artifacts
WORKING_DOCS_DIR = Path(os.environ.get("WORKING_DOCS_DIR", "workingDocuments")).resolve()
WORKING_DOCS_DIR.mkdir(parents=True, exist_ok=True)


# ====== SMALL HELPERS ======
def truncate_to_tokens(text: str, max_tokens: int = 95_000) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text or "")
    if len(toks) <= max_tokens:
        return text
    return enc.decode(toks[:max_tokens])

def _ok(content: str) -> bool:
    return (
        bool(content)
        and not content.startswith("[visit] Failed")
        and content.strip() != "[visit] Empty content."
        and not content.startswith("[document_parser]")
    )

def _fallback_payload(url: str, goal: str, doc_id: Optional[int] = None) -> str:
    tag = f"[[DOC_ID:{doc_id}]]\n" if doc_id else ""
    return tag + (
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

def _slugify(text: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "")).strip("-").lower()
    return s[:maxlen] or "q"

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _make_run_folder(goal: str, question_id: Optional[str]) -> Path:
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

def _guess_ext(url: str, content_type: Optional[str]) -> str:
    p = urlparse(url)
    name = Path(p.path).name
    if "." in name and len(name.split(".")[-1]) <= 5:
        return "." + name.split(".")[-1].lower()
    if content_type:
        ext = mimetypes.guess_extension((content_type or "").split(";")[0].strip())
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


# ====== JINA FETCH ======
def jina_readpage(url: str) -> str:
    """
    Fetch a cleaned representation of the page via Jina's reader gateway.
    Returns markdown-like text for most HTML sites (and many PDFs).
    """
    max_retries = 3
    timeout = 50
    headers = {"Authorization": f"Bearer {JINA_API_KEYS}"} if JINA_API_KEYS else {}

    for attempt in range(max_retries):
        try:
            r = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.text or ""
            # non-200 → try again
        except Exception:
            pass
        time.sleep(0.5)
    return "[visit] Failed to read page."

def html_readpage_jina(url: str) -> str:
    for _ in range(8):
        content = jina_readpage(url)
        if _ok(content) and content != "[visit] Empty content.":
            return content
    return "[visit] Failed to read page."


# ====== SUMMARIZER CALL ======
def _call_summary_server(messages, max_retries: int = 3) -> str:
    client = OpenAI(
        api_key = os.environ.get("API_KEY"),
        base_url = os.environ.get("API_BASE")
    )
    model = os.environ.get("SUMMARY_MODEL_NAME", "")

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7
            )
            content = (resp.choices[0].message.content or "").strip()
            # try to return clean JSON
            try:
                json.loads(content)
                return content
            except Exception:
                left, right = content.find("{"), content.rfind("}")
                if 0 <= left < right:
                    return content[left:right+1]
                return content
        except Exception:
            if attempt == max_retries - 1:
                return ""
            time.sleep(0.3)
    return ""


# ====== THE TOOL ======
@register_tool("visit", allow_overwrite=True)
class Visit(BaseTool):
    name = "visit"
    description = "Visit webpage(s) using Jina reader, summarize to evidence/summary, and save artifacts + doc_id."
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
            "question_id": {
                "type": "string",
                "description": "If provided, all saved files go under this folder name."
            }
        },
        "required": ["url", "goal"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
            question_id = params.get("question_id")
        except Exception:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        question_id = str(question_id) if question_id is not None else None
        qdir = _make_run_folder(goal, question_id)
        print(f"[visit] saving to {qdir}")

        start = time.time()

        if isinstance(url, str):
            try:
                return self._read_and_summarize(url, goal, qdir)
            except Exception as e:
                return f"[visit] Failed on {url}: {e}"

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

    def _read_and_summarize(self, url: str, goal: str, qdir: Path) -> str:
        base = _basename_for_url(url)
        ts = _utc_now_iso()
        raw_path = None  # raw archive off by default (can add if you want)
        doc_id: Optional[int] = None

        # 1) fetch content via Jina
        content = html_readpage_jina(url)

        # 1a) save extracted content (and DB doc_id)
        extracted_path = None
        if _ok(content):
            # Truncate for safety before saving/DB
            clipped = truncate_to_tokens(content, WEBCONTENT_MAXLENGTH // 2)
            extracted_path = qdir / f"{base}__extracted.md"
            try:
                header = f"# Extracted content (Jina)\n\nURL: {url}\nSaved: {ts}\n\n---\n\n"
                _save_text(extracted_path, header + clipped)
                # save only the cleaned markdown to DB; return numeric id
                doc_id = save_document(clipped)
            except Exception:
                extracted_path = None

        # 2) bail early if extraction failed
        if not _ok(content):
            _update_index(qdir, {
                "url": url,
                "saved_at": ts,
                "goal": goal,
                "raw_file": (str(raw_path) if raw_path else None),
                "extracted_file": None,
                "summary_file": None,
                "status": "fallback",
                "final_url": url,
                "doc_id": doc_id,
            })
            return _fallback_payload(url, goal, doc_id=doc_id)

        # 3) summarize to JSON {evidence, summary}
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

        # parse JSON (retry a little)
        if isinstance(raw, str):
            raw = raw.replace("```json", "").replace("```", "").strip()

        parse_retry_times = 0
        obj = None
        while parse_retry_times < 2:
            try:
                obj = json.loads(raw)
                break
            except Exception:
                raw = _call_summary_server(messages, max_retries=1)
                if isinstance(raw, str):
                    raw = raw.replace("```json", "").replace("```", "").strip()
                parse_retry_times += 1

        if obj is None:
            _update_index(qdir, {
                "url": url,
                "saved_at": ts,
                "goal": goal,
                "raw_file": (str(raw_path) if raw_path else None),
                "extracted_file": (str(extracted_path) if extracted_path else None),
                "summary_file": None,
                "status": "parse_failed",
                "final_url": url,
                "doc_id": doc_id,
            })
            return _fallback_payload(url, goal, doc_id=doc_id)

        evidence = str(obj.get("evidence", ""))
        summary  = str(obj.get("summary", ""))

        # (summary JSON save intentionally disabled to keep disk light)
        summary_path = None

        # 5) update index
        _update_index(qdir, {
            "url": url,
            "saved_at": ts,
            "goal": goal,
            "raw_file": (str(raw_path) if raw_path else None),
            "extracted_file": (str(extracted_path) if extracted_path else None),
            "summary_file": (str(summary_path) if summary_path else None),
            "status": "ok",
            "final_url": url,
            "doc_id": doc_id,
        })

        # 6) return the orchestrator-friendly text block (with DOC_ID tag up front)
        result = (
            f"The useful information in {url} for user goal {goal} as follows: \n\n"
            f"Evidence in page: \n{evidence}\n\n"
            f"Summary: \n{summary}\n\n"
        )
        if len(result) < 10 and summary_retries < 0:
            return "[visit] Failed to read page"

        tag = f"[[DOC_ID:{doc_id}]]\n" if doc_id else ""
        return tag + result
