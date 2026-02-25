import os
import time
import requests
from typing import List, Optional, Union
from qwen_agent.tools.base import BaseTool, register_tool


SEARXNG_URL = "http://127.0.0.1:8080".rstrip("/")

@register_tool("search",allow_overwrite = True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self._s = requests.Session()
        self._s.headers.update({"User-Agent": "searchLOCAL/1.0"})

    def _one(self, query: str) -> str:
        # Match your old failure message if the backend isn't reachable
        if not SEARXNG_URL:
            return "Google search Timeout, return None, Please try again later."

        params = {
            "q": query,
            "format": "json",
            # stick to friendly engines; harmless if some are disabled in SearXNG
            "engines": "duckduckgo,brave,startpage,wikipedia",
            "language": "en",
        }

        # Same retry semantics (5 tries, same final timeout string)
        for attempt in range(5):
            try:
                r = self._s.get(f"{SEARXNG_URL}/search", params=params, timeout=20)
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                if attempt == 4:
                    return "Google search Timeout, return None, Please try again later."
                time.sleep(0.2)

        results = data.get("results") or []
        if not results:
            return f"No results found for '{query}'. Try with a more general query."

        rows: List[str] = []
        for i, res in enumerate(results[:10], 1):
            title = (res.get("title") or "Untitled").strip()
            link = (res.get("url") or "").strip()

            # Best-effort mapping to your old "Date published" / "Source" / snippet lines
            date = (res.get("publishedDate") or res.get("date") or "").strip()
            date_line = f"\nDate published: {date}" if date else ""
            source = (res.get("engine") or "").strip()
            source_line = f"\nSource: {source}" if source else ""
            snippet = (res.get("content") or "").strip()
            snippet_line = f"\n{snippet}" if snippet else ""

            line = f"{i}. [{title}]({link}){date_line}{source_line}\n{snippet_line}"
            line = line.replace("Your browser can't play this video.", "")
            rows.append(line)

        return (
            f"A Google search for '{query}' found {len(rows)} results:\n\n"
            "## Web Results\n" + "\n\n".join(rows)
        )

    def call(self, params: Union[str, dict], **kwargs) -> str:
        # Same input contract/error text as your original tool
        try:
            query = params["query"]
        except Exception:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        if isinstance(query, str):
            return self._one(query)

        assert isinstance(query, List)
        blocks = [self._one(q) for q in query]  # sequential to mirror old behavior
        return "\n=======\n".join(blocks)
