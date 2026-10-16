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
import time


SERPER_KEY=os.environ.get('SERPER_KEY_ID')
# Optional You.com backend: set SEARCH_PROVIDER=you (default: serper).
# YDC_API_KEY is optional; without it the keyless endpoint is used (100 searches/day per IP).
SEARCH_PROVIDER=os.environ.get('SEARCH_PROVIDER', 'serper').lower()
YDC_API_KEY=os.environ.get('YDC_API_KEY')
YDC_USER_AGENT="youdotcom-integration/alibaba-nlp-deepresearch"


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
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
    def google_search_with_serp(self, query: str):
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
                'X-API-KEY': SERPER_KEY,
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


    
    def search_with_youcom(self, query: str):
        headers = {'User-Agent': YDC_USER_AGENT}
        if YDC_API_KEY:
            url = "https://ydc-index.io/v1/search"
            headers['X-API-Key'] = YDC_API_KEY
            kwargs = {'json': {"query": query, "count": 10}}
        else:
            url = "https://api.you.com/v1/agents/search"
            kwargs = {'params': {"query": query, "count": 10}}

        for i in range(5):
            try:
                res = requests.request("POST" if YDC_API_KEY else "GET", url, headers=headers, timeout=30, **kwargs)
                if res.status_code == 429 and i < 4:
                    retry_after = res.headers.get("Retry-After", "")
                    time.sleep(min(int(retry_after), 60) if retry_after.isdigit() else 2 ** i)
                    continue
                if res.status_code in (401, 402, 403, 429):
                    reason = {401: "invalid YDC_API_KEY", 402: "out of credits",
                              403: "key lacks scope or wrong host", 429: "rate limited"}[res.status_code]
                    print(f"You.com search error {res.status_code}: {reason}")
                    return f"You.com search failed ({res.status_code}: {reason}), Please try again later."
                res.raise_for_status()
                results = res.json()
                break
            except Exception as e:
                print(e)
                if i == 4:
                    return f"Google search Timeout, return None, Please try again later."
                continue

        try:
            pages = (results.get("results") or {}).get("web") or []
            if not pages:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            idx = 0
            for page in pages:
                idx += 1
                date_published = ""
                if page.get("page_age"):
                    date_published = "\nDate published: " + str(page["page_age"]).split("T")[0]

                snippet = ""
                if page.get("description"):
                    snippet = "\n" + page["description"]

                redacted_version = f"{idx}. [{page.get('title', '')}]({page.get('url', '')}){date_published}\n{snippet}"
                web_snippets.append(redacted_version)

            content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            return content
        except:
            return f"No results found for '{query}'. Try with a more general query."

    def search_with_serp(self, query: str):
        if SEARCH_PROVIDER == "you":
            return self.search_with_youcom(query)
        result = self.google_search_with_serp(query)
        return result

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            # 单个查询
            response = self.search_with_serp(query)
        else:
            # 多个查询
            assert isinstance(query, List)
            responses = []
            for q in query:
                responses.append(self.search_with_serp(q))
            response = "\n=======\n".join(responses)
            
        return response

