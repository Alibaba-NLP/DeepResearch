from qwen_agent.tools.base import BaseTool, register_tool 
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
import os

SEARCH_API_URL = os.getenv("SEARCH_API_URL")
GOOGLE_SEARCH_KEY = os.getenv("GOOGLE_SEARCH_KEY")


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

    def google_search(self, query: str):
        url = 'https://google.serper.dev/search'
        headers = {
            'X-API-KEY': GOOGLE_SEARCH_KEY,
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
        assert GOOGLE_SEARCH_KEY is not None, "Please set the GOOGLE_SEARCH_KEY environment variable."
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            response = self.google_search(query)
        else:
            assert isinstance(query, List)
            with ThreadPoolExecutor(max_workers=3) as executor:
                response = list(executor.map(self.google_search, query))
            response = "\n=======\n".join(response)
        return response
