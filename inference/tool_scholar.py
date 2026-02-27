import os
import json
import requests
from typing import Union, List
from qwen_agent.tools.base import BaseTool, register_tool
from concurrent.futures import ThreadPoolExecutor
import http.client


from tavily import TavilyClient


SERPER_KEY=os.environ.get('SERPER_KEY_ID')
SEARCH_PROVIDER=os.environ.get('SEARCH_PROVIDER', 'serper').lower()


@register_tool("google_scholar", allow_overwrite=True)
class Scholar(BaseTool):
    name = "google_scholar"
    description = "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries."
    parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string", "description": "The search query."},
                    "minItems": 1,
                    "description": "The list of search queries for Google Scholar."
                },
            },
        "required": ["query"],
    }

    def google_scholar_with_serp(self, query: str):
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
        "q": query,
        })
        headers = {
        'X-API-KEY': SERPER_KEY,
        'Content-Type': 'application/json'
        }
        for i in range(5):
            try:
                conn.request("POST", "/scholar", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                print(e)
                if i == 4:
                    return f"Google Scholar Timeout, return None, Please try again later."
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
                    if "year" in page:
                        date_published = "\nDate published: " + str(page["year"])

                    publicationInfo = ""
                    if "publicationInfo" in page:
                        publicationInfo = "\npublicationInfo: " + page["publicationInfo"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]
                    
                    link_info = "no available link"
                    if "pdfUrl" in page: 
                        link_info = "pdfUrl: " + page["pdfUrl"]
                    
                    citedBy = ""
                    if "citedBy" in page:
                        citedBy = "\ncitedBy: " + str(page["citedBy"])
                    
                    redacted_version = f"{idx}. [{page['title']}]({link_info}){publicationInfo}{date_published}{citedBy}\n{snippet}"

                    redacted_version = redacted_version.replace("Your browser can't play this video.", "") 
                    web_snippets.append(redacted_version)

            content = f"A Google scholar for '{query}' found {len(web_snippets)} results:\n\n## Scholar Results\n" + "\n\n".join(web_snippets)
            return content
        except:
            return f"No results found for '{query}'. Try with a more general query."


    def scholar_with_tavily(self, query: str):
        academic_domains = [
            "scholar.google.com",
            "arxiv.org",
            "pubmed.ncbi.nlm.nih.gov",
            "semanticscholar.org",
            "ieee.org",
            "acm.org",
            "springer.com",
            "sciencedirect.com",
            "nature.com",
            "wiley.com",
        ]
        try:
            client = TavilyClient()
            response = client.search(
                query=query,
                max_results=10,
                search_depth="advanced",
                include_domains=academic_domains,
            )
        except Exception as e:
            print(e)
            return f"Tavily Scholar Timeout, return None, Please try again later."

        try:
            results = response.get("results", [])
            if not results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            for idx, page in enumerate(results, 1):
                date_published = ""
                if page.get("published_date"):
                    date_published = "\nDate published: " + page["published_date"]

                snippet = ""
                if page.get("content"):
                    snippet = "\n" + page["content"]

                redacted_version = f"{idx}. [{page.get('title', '')}]({page.get('url', '')}){date_published}\n{snippet}"
                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)

            content = f"A Tavily scholar search for '{query}' found {len(web_snippets)} results:\n\n## Scholar Results\n" + "\n\n".join(web_snippets)
            return content
        except:
            return f"No results found for '{query}'. Try with a more general query."

    def _do_scholar_search(self, query: str):
        if SEARCH_PROVIDER == 'tavily':
            return self.scholar_with_tavily(query)
        return self.google_scholar_with_serp(query)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        # assert GOOGLE_SEARCH_KEY is not None, "Please set the IDEALAB_SEARCH_KEY environment variable."
        try:
            params = self._verify_json_format_args(params)
            query = params["query"]
        except:
            return "[google_scholar] Invalid request format: Input must be a JSON object containing 'query' field"

        if isinstance(query, str):
            response = self._do_scholar_search(query)
        else:
            assert isinstance(query, List)
            with ThreadPoolExecutor(max_workers=3) as executor:

                response = list(executor.map(self._do_scholar_search, query))
            response = "\n=======\n".join(response)
        return response
