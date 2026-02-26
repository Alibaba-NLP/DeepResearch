import os
import json
import requests
from typing import Union, List
from qwen_agent.tools.base import BaseTool, register_tool
from concurrent.futures import ThreadPoolExecutor

import sys 
sys.path.append("../")
sys.path.append("./")
try:
    from .tool_select_url import SelectURL
except:
    from tool_select_url import SelectURL

try:
    from .tool_visit import Visit
    from .tool_search import Search
except:
    from tool_visit import Visit
    from tool_search import Search


@register_tool('search_and_visit')
class SearchAndVisit(BaseTool):
    name = "search_and_visit"
    description = "Perform Google web searches, select related pages, visit them and output relevant statements for the query. Accepts multiple queries."
    parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string", "description": "The search query."},
                    "minItems": 1,
                    "description": "The list of search queries."
                },
                "goal": {
                    "type": "string",
                    "description": "The goal of the search."
                },
            },
        "required": ["query", "goal"],
    }

    def parse_existing_page_info(self, page_info):
        url_page = {}
        for item in page_info:
            url_page[item["url"]] = item

        return url_page

    def call(self, params: Union[str, dict], **kwargs) -> list:
        # assert GOOGLE_SEARCH_KEY is not None, "Please set the IDEALAB_SEARCH_KEY environment variable."
        try:
            params = self._verify_json_format_args(params)
            query = params["query"]
            goal = params["goal"]
            page_info = params.get("page_info")
            url_page = self.parse_existing_page_info(page_info)
            # filter_year = params.get("filter_year", None)
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        search_tool = Search()
        params_search = {
                "query": query
            }
        response = search_tool.call(params_search)
        
        select_url_class = SelectURL()
        selected_url_json = select_url_class.call({"search_results": response, "goal": goal})

        ### filter existing page
        existing_page_info = []
        non_existing_urls = []
        if isinstance(selected_url_json, str) or "urls" not in selected_url_json:
            print(f"error in url selection: ", selected_url_json)
            selected_url_json = {"urls": []}

        if isinstance(selected_url_json["urls"], str):
            selected_url_json["urls"] = [selected_url_json["urls"]]  
        for url in selected_url_json["urls"]:
            if url in url_page:
                existing_page_info.append(url_page[url])
            else:
                non_existing_urls.append(url)

        visit_class = Visit()
        visit_results = []
        if len(non_existing_urls) > 0:
            visit_results = visit_class.call({"url": non_existing_urls, "goal": goal})

        existing_page_info.extend(visit_results)


        return existing_page_info


if __name__ == "__main__":
    tool = SearchAndVisit()
    print(tool.call({"query": ["HKUST", "The university in Hong Kong", "rank of university"], "goal": "introduce the university HKUST", "page_info": []}))
