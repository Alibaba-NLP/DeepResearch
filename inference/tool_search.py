from typing import List, Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool

from inference.firecrawl_client import get_firecrawl_client


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
        
    def search_with_firecrawl(self, query: str) -> str:
        client = get_firecrawl_client()

        try:
            response = client.search(query, limit=10, scrape_options={
                "formats": ["summary"]
            })
        except Exception as error:
            return f"[Search] Firecrawl search request failed: {error}"
        
        web_search_results = getattr(response, "web", None) or []

        if not web_search_results:
            return f"No results found for '{query}'. Try with a more general query."

        web_snippets = []
        for idx, web_search_result in enumerate(web_search_results, start=1):

            try:
                title = getattr(web_search_result.metadata, "title", "Untitled")
                url = getattr(web_search_result.metadata, "url", "")
                description = getattr(web_search_result.metadata, "description", "")
                summary = getattr(web_search_result, "summary", "")
            except:
                title = "Untitled"
                url = ""
                description = ""
                summary = ""

            entry = f"{idx}. [{title}]({url})"
            if description:
                entry += f"\nDescription: {description}"
            if summary:
                entry += f"\nContent Summary: {summary}"

            web_snippets.append(entry.strip())
        
        content = (
            f"A Firecrawl search for '{query}' found {len(web_snippets)} results:\n\n"
            "## Web Results\n" + "\n\n".join(web_snippets)
        )

        return content

    
    def search_with_serp(self, query: str):
        result = self.search_with_firecrawl(query)
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

