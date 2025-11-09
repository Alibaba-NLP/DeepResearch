import os
import re
import json
import time
import random
import requests
import datetime
from typing import Union, List
from dataclasses import dataclass
from functools import wraps
import atexit
import uuid

from qwen_agent.tools.private.cache_utils import JSONLCache
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.tools.private.sfilter import multi_call_sfilter
from qwen_agent.log import logger

MAX_CHAR = int(os.getenv("MAX_CHAR", default=28000))
SEARCH_ENGINE = os.getenv("SEARCH_ENGINE", "google")
SEARCH_STRATEGY = os.getenv("SEARCH_STRATEGY", "rerank")
TEXT_SEARCH_KEY = os.getenv("TEXT_SEARCH_KEY")

# knowledge
KNOWLEDGE_SNIPPET = """## 来自 {source} 的内容：

```
{content}
```"""

KNOWLEDGE_PROMPT = """# 知识库

{knowledge_snippets}"""



@register_tool("web_search", allow_overwrite=True)
class WebSearch(BaseTool):
    name = "web_search"
    description = "Call this tool to interact with the web_search API. You will receive the top 10 text excerpts from Google's text search engine using text as the search query."
    parameters = {
        "type": "object",
            "properties": {
            "queries": {
                "type": "array",
                "items": {
                "type": "string",
                "description": "The search query."
                },
                "description": "The list of search queries."
                }
            },
            "required": [
            "queries"
            ]
    }

    def google_search(self, query: str):
        url = 'https://google.serper.dev/search'
        headers = {
            'X-API-KEY': TEXT_SEARCH_KEY,
            'Content-Type': 'application/json',
        }
        data = {
            "q": query,
            "num": 10,
            "extendParams": {
                "country": "en",
                "page": 1,
            },
        }

        for i in range(5):
            try:
                response = requests.post(url, headers=headers, data=json.dumps(data))
                results = response.json()
            except Exception as e:
                print(e)
                if i == 4:
                    return f"Google search Timeout, return None, Please try again later."
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} - {response.text}")

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
            return f"No results found for '{query}'. Try with a more general query, or remove the year filter."


    def call(self, params: Union[str, dict], **kwargs) -> str:
        assert TEXT_SEARCH_KEY is not None, "Please set the TEXT_SEARCH_KEY environment variable."
        try:
            query = params["queries"][0]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'queries' field"
        
        if isinstance(query, str):
            response = self.google_search(query)
        else:
            assert isinstance(query, List)
            with ThreadPoolExecutor(max_workers=3) as executor:
                response = list(executor.map(self.google_search, query))
            response = "\n=======\n".join(response)
        return response


if __name__ == "__main__":
    # os.environ['NLP_WEB_SEARCH_ONLY_CACHE'] = 'false'
    # os.environ['NLP_WEB_SEARCH_ENABLE_READPAGE'] = 'true'
    # os.environ['NLP_WEB_SEARCH_ENABLE_SFILTER'] = 'true'
    print(WebSearch().call({"queries": ['Boston Terrier dog black and white short face compact build']}))