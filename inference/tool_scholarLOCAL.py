# tool_scholarLOCAL.py
import time
import requests
from typing import List, Optional, Union
from qwen_agent.tools.base import BaseTool, register_tool

from tool_visitLOCAL import (
    WORKING_DOCS_DIR,
    _make_run_folder,
    _update_index,
)

# small local helpers we need for saving
from tool_visitLOCAL import _utc_now_iso  # already there in your visit file
from pathlib import Path
import json, re, secrets

S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
CR_URL = "https://api.crossref.org/works"

@register_tool("google_scholar", allow_overwrite=True)
class Scholar(BaseTool):
    name = "google_scholar"
    description = "Leverage scholarly indexes to retrieve relevant information from academic publications. Accepts multiple queries."
    parameters = {
    "type": "object",
    "properties": {
        "query": {
            "type": ["string", "array"],
            "items": {"type": "string"},
            "description": "Query string or array of query strings for scholarly search."
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "description": "How many results per query (default 10)."
        },
        "question_id": {  # NEW
            "type": "string",
            "description": "If provided, all saved files go under this folder name (shared with 'visit')."
        }
    },
    "required": ["query"],
    }
    
    '''
    def _save_query_artifacts(self, qdir: Path, query: str, text: str, s2_raw: list, cr_raw: list) -> None:
    # filename base (like visit does for URLs)
        safe = re.sub(r"[^a-zA-Z0-9]+", "-", query).strip("-").lower()[:36]
        base = f"scholar-{safe}-{secrets.token_hex(4)}"
        ts = _utc_now_iso()

        # save a JSON bundle
        json_path = qdir / f"{base}__scholar.json"
        bundle = {
            "type": "scholar",
            "query": query,
            "saved_at": ts,
            "providers": ["SemanticScholar", "Crossref"],
            "counts": {
                "semanticscholar": len(s2_raw or []),
                "crossref": len(cr_raw or []),
                "total": (len(s2_raw or []) + len(cr_raw or [])),
            },
            "results": {
                "semanticscholar": s2_raw or [],
                "crossref": cr_raw or [],
            },
            "formatted_text": text or "",
        }
        json_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

        # save a readable markdown alongside
        md_path = qdir / f"{base}__scholar.md"
        md = (
            f"# Scholarly Search Results\n\n"
            f"**Query:** {query}\n\n"
            f"**Saved:** {ts}\n\n"
            f"---\n\n"
            f"{text or ''}\n"
        )
        md_path.write_text(md, encoding="utf-8")

        # append a single record into the SAME index.json used by visit
        _update_index(qdir, {
            "type": "scholar",
            "query": query,
            "saved_at": ts,
            "json_file": str(json_path),
            "md_file": str(md_path),
            "status": "ok",
            "counts": bundle["counts"],
            "preview": (text or "")[:300],
        })

'''
    def _save_query_artifacts(self, qdir: Path, query: str, text: str, s2_raw: list, cr_raw: list) -> None:
        # filename base (like visit does for URLs)
        safe = re.sub(r"[^a-zA-Z0-9]+", "-", query).strip("-").lower()[:36]
        base = f"scholar-{safe}-{secrets.token_hex(4)}"
        ts = _utc_now_iso()

        #  Save a readable MARKDOWN file only
        md_path = qdir / f"{base}__scholar.md"
        md = (
            f"# Scholarly Search Results\n\n"
            f"**Query:** {query}\n\n"
            f"**Saved:** {ts}\n\n"
            f"---\n\n"
            f"{text or ''}\n"
        )
        md_path.write_text(md, encoding="utf-8")

        #  Append a compact record to index.json (no json_file, no big payloads)
        _update_index(qdir, {
            "type": "scholar",
            "query": query,
            "saved_at": ts,
            "md_file": str(md_path),
            "status": "ok",
            "counts": {
                "semanticscholar": len(s2_raw or []),
                "crossref": len(cr_raw or []),
                "total": (len(s2_raw or []) + len(cr_raw or [])),
            },
            "preview": (text or "")[:300],
        })

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self._s = requests.Session()
        self._s.headers.update({"User-Agent": "scholarLOCAL/1.0"})

    def _search_semanticscholar(self, q: str, k: int) -> List[dict]:
        params = {
            "query": q,
            "limit": min(k, 20),     # S2 returns up to 20 per page
            "fields": "title,url,authors,year,venue,abstract"
        }
        r = self._s.get(S2_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json() or {}
        return data.get("data", []) or []

    def _search_crossref(self, q: str, k: int) -> List[dict]:
        params = {"query": q, "rows": min(k, 20)}
        r = self._s.get(CR_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json() or {}
        return (data.get("message", {}) or {}).get("items", []) or []

    def _format_item_s2(self, i: int, rec: dict) -> str:
        title = (rec.get("title") or "Untitled").strip()
        url = (rec.get("url") or "").strip()
        year = rec.get("year") or ""
        venue = rec.get("venue") or ""
        authors = rec.get("authors") or []
        snippet = rec.get("abstract") or ""
        date_line = f"\nDate published: {year}" if year else ""
        source_line = "\nSource: Semantic Scholar"
        snippet_line = f"\n{snippet.strip()}" if snippet else ""
        return f"{i}. [{title}]({url}){date_line}{source_line}\n{snippet_line}"

    def _format_item_cr(self, i: int, rec: dict) -> str:
        title_list = rec.get("title") or []
        title = (title_list[0] if title_list else "Untitled").strip()
        url = ""
        for l in rec.get("link") or []:
            if l.get("URL"):
                url = l["URL"]
                break
        if not url:
            url = rec.get("URL") or ""
        year = ""
        issued = rec.get("issued", {}).get("date-parts") or []
        if issued and issued[0]:
            year = issued[0][0]
        container = rec.get("container-title") or []
        venue = container[0] if container else ""
        abstract = rec.get("abstract") or ""
        date_line = f"\nDate published: {year}" if year else ""
        source_line = "\nSource: Crossref"
        snippet_line = f"\n{abstract.strip()}" if abstract else ""
        return f"{i}. [{title}]({url}){date_line}{source_line}\n{snippet_line}"

    def _one(self, query: str, max_results: int = 10) -> str:
        # Try S2 first (fast, free), then fill from Crossref if needed.
        collected: List[str] = []
        # retry semantics similar to your other tools
        for attempt in range(5):
            try:
                s2 = self._search_semanticscholar(query, max_results)
                break
            except Exception:
                if attempt == 4:
                    return "Google search Timeout, return None, Please try again later."
                time.sleep(0.3)

        for i, rec in enumerate(s2[:max_results], 1):
            collected.append(self._format_item_s2(i, rec))

        if len(collected) < max_results:
            need = max_results - len(collected)
            try:
                cr = self._search_crossref(query, need)
                start_idx = len(collected) + 1
                for j, rec in enumerate(cr[:need], start_idx):
                    collected.append(self._format_item_cr(j, rec))
            except Exception:
                pass  # fine—return whatever we have

        if not collected:
            return f"No results found for '{query}'. Try with a more general query."

        return (
            f"A Google search for '{query}' found {len(collected)} results:\n\n"
            "## Web Results\n" + "\n\n".join(collected)
        )

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            qv = params["query"]
            k = params.get("max_results", 10)
            question_id = params.get("question_id")  # NEW
        except Exception:
            return "[google_scholar] Invalid request format: Input must be a JSON object containing 'query' field"

        # SAME DIRECTORY as `visit`:
        # visit's _make_run_folder(goal, question_id) – if question_id is present, it wins.
        qdir = _make_run_folder("scholar", question_id)
        print(f"[scholar] saving to {qdir}")

        def run_one(q: str) -> str:
            # do your current search logic, but capture raw lists
            s2_raw = []
            cr_raw = []
            text = ""

            # >>> your existing _search_semanticscholar + _search_crossref code,
            # but keep the raw arrays in s2_raw / cr_raw, and build the final
            # formatted 'text' exactly like before.

            # 1) S2
            for attempt in range(5):
                try:
                    s2_raw = self._search_semanticscholar(q, k)
                    break
                except Exception:
                    if attempt == 4:
                        text = "Google search Timeout, return None, Please try again later."
                        self._save_query_artifacts(qdir, q, text, [], [])
                        return text
                    time.sleep(0.3)

            collected = []
            for i, rec in enumerate(s2_raw[:k], 1):
                collected.append(self._format_item_s2(i, rec))

            # 2) Crossref fill
            if len(collected) < k:
                need = k - len(collected)
                try:
                    cr_raw = self._search_crossref(q, need)
                    start_idx = len(collected) + 1
                    for j, rec in enumerate(cr_raw[:need], start_idx):
                        collected.append(self._format_item_cr(j, rec))
                except Exception:
                    pass

            if not collected:
                text = f"No results found for '{q}'. Try with a more general query."
                self._save_query_artifacts(qdir, q, text, s2_raw, cr_raw)
                return text

            text = (
                f"A Google search for '{q}' found {len(collected)} results:\n\n"
                "## Web Results\n" + "\n\n".join(collected)
            )

            # WRITE into the SAME index.json inside qdir
            self._save_query_artifacts(qdir, q, text, s2_raw, cr_raw)
            return text

        if isinstance(qv, str):
            return run_one(qv)

        assert isinstance(qv, list)
        blocks = [run_one(q) for q in qv]
        return "\n=======\n".join(blocks)

