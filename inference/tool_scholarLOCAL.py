# tool_scholarLOCAL.py
import time
import requests
from typing import List, Optional, Union
from qwen_agent.tools.base import BaseTool, register_tool

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
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of query strings for scholarly search."
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "How many results per query (default 10)."
                }
        },
        "required": ["query"],
    }

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
        except Exception:
            return "[google_scholar] Invalid request format: Input must be a JSON object containing 'query' field"

        if isinstance(qv, str):
            return self._one(qv, max_results=k)

        assert isinstance(qv, List)
        blocks = [self._one(q, max_results=k) for q in qv]
        return "\n=======\n".join(blocks)
