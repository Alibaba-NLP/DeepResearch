import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
import asyncio
from typing import Dict, List, Optional, Union
import uuid
import http.client
import json

import os


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; returns source links and snippets for each query."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self._search_config = cfg or {}
        self._parallel_session_id = uuid.uuid4().hex

    def search_provider(self):
        provider = self._search_config.get('provider', os.getenv('SEARCH_PROVIDER', '')).strip().lower()
        if provider:
            return provider
        key = os.getenv('SERPER_KEY_ID', '').strip()
        return 'serper' if key and key != 'your_key' else 'parallel'

    def search_with_parallel(self, query: str, session_id: str):
        # Keep the optional transport import out of the existing Serper path.
        from tool_parallel_search import parallel_search
        try:
            return asyncio.run(parallel_search(query, session_id))
        except Exception:
            # Transport errors may contain headers or credentials. Do not expose them
            # in model observations, and never switch providers after a failure.
            return '[Search] Parallel search failed or timed out. Please try again later.'

    def google_search_with_serp(self, query: str):
        serper_key = os.getenv('SERPER_KEY_ID', '').strip()
        if not serper_key or serper_key == 'your_key':
            return '[Search] Serper requires SERPER_KEY_ID.'
        def contains_chinese_basic(text: str) -> bool:
            return any('\u4E00' <= char <= '\u9FFF' for char in text)
        conn = http.client.HTTPSConnection("google.serper.dev")
        if contains_chinese_basic(query):
            payload = json.dumps({
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn"
            })
            
        else:
            payload = json.dumps({
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en"
            })
        headers = {
                'X-API-KEY': serper_key,
                'Content-Type': 'application/json'
            }
        
        
        for i in range(5):
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                print(e)
                if i == 4:
                    return f"Google search Timeout, return None, Please try again later."
                continue
    
        data = res.read()
        results = json.loads(data.decode("utf-8"))

        try:
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            idx = 0
            if "organic" in results:
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

            content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            return content
        except:
            return f"No results found for '{query}'. Try with a more general query."


    
    def search_with_serp(self, query: str):
        result = self.google_search_with_serp(query)
        return result

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            if isinstance(params, str):
                params = json.loads(params)
            query = params["query"]
        except (ValueError, KeyError, TypeError):
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        queries = [query] if isinstance(query, str) else query
        if (not isinstance(queries, list) or not queries or
                any(not isinstance(q, str) or not q.strip() for q in queries)):
            return '[Search] query must be a nonempty string or array of nonempty strings.'
        provider = self.search_provider()
        if provider == 'none':
            return '[Search] Web search is disabled.'
        if provider not in ('serper', 'parallel'):
            return '[Search] Unknown SEARCH_PROVIDER. Choose parallel, serper or none.'
        session_id = kwargs.get('search_session_id') or self._parallel_session_id
        responses = []
        for q in queries:
            if provider == 'parallel':
                responses.append(self.search_with_parallel(q, session_id))
            else:
                responses.append(self.search_with_serp(q))
        return '\n=======\n'.join(responses)

