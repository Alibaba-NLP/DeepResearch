"""
Exa.ai Search Tool for DeepResearch
AI-native semantic search with neural embeddings for superior research results.

Exa.ai advantages:
- Neural/semantic search (understands meaning, not just keywords)
- Can retrieve full page contents directly
- Better for research and complex queries
- Built-in query optimization (autoprompt)
- Supports date filtering and domain restrictions
- Category filtering (research papers, news, company info, etc.)
- AI-generated highlights for quick comprehension
"""

import json
import os
from typing import Any, Dict, Optional, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool

EXA_API_KEY = os.environ.get('EXA_API_KEY')
EXA_BASE_URL = "https://api.exa.ai"

# Valid Exa categories for filtering results
VALID_CATEGORIES = [
    "company", "research paper", "news", "pdf", 
    "github", "tweet", "personal site", "linkedin profile"
]


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs semantic web searches using Exa.ai: supply an array 'query'; retrieves top results with AI-powered understanding. Supports category filtering for research papers, news, etc."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings. Exa understands natural language queries well."
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results per query (default: 10, max: 100)",
                "default": 10
            },
            "include_contents": {
                "type": "boolean",
                "description": "Whether to include page text content and highlights",
                "default": False
            },
            "category": {
                "type": "string",
                "description": "Filter by category: 'research paper', 'news', 'company', 'pdf', 'github', 'tweet', 'personal site', 'linkedin profile'",
                "enum": ["company", "research paper", "news", "pdf", "github", "tweet", "personal site", "linkedin profile"]
            }
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.api_key = EXA_API_KEY
        if not self.api_key:
            raise ValueError("EXA_API_KEY environment variable not set. Get your key from https://exa.ai/")

    def exa_search(
        self, 
        query: str, 
        num_results: int = 10, 
        include_contents: bool = False,
        category: Optional[str] = None
    ) -> str:
        """
        Perform a search using Exa.ai API.
        
        Exa supports multiple search types:
        - "auto": Intelligently combines neural and other methods (default)
        - "neural": AI-powered semantic search
        - "deep": Comprehensive search with query expansion
        
        Categories available:
        - "research paper": Academic papers and publications
        - "news": News articles
        - "company": Company websites and info
        - "pdf": PDF documents
        - "github": GitHub repositories
        - "tweet": Twitter/X posts
        - "personal site": Personal websites/blogs
        - "linkedin profile": LinkedIn profiles
        """
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        payload: Dict[str, Any] = {
            "query": query,
            "numResults": num_results,
            "type": "auto",
            "useAutoprompt": True,
        }
        
        # Add category filter if specified
        if category and category in VALID_CATEGORIES:
            payload["category"] = category
        
        if include_contents:
            payload["contents"] = {
                "text": {"maxCharacters": 2000},
                "highlights": True
            }
        
        response = None
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{EXA_BASE_URL}/search",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                if response is not None and response.status_code == 429:
                    return f"Exa search rate limited. Please wait and try again."
                if response is not None and response.status_code == 401:
                    return f"Exa API key invalid. Check your EXA_API_KEY environment variable."
                if attempt == 2:
                    return f"Exa search failed after 3 attempts: {str(e)}"
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    return f"Exa search failed after 3 attempts: {str(e)}"
                continue
        
        if response is None:
            return "Exa search failed: no response received"
        
        results = response.json()
        
        if "results" not in results or not results["results"]:
            return f"No results found for '{query}'. Try a different query."
        
        web_snippets = []
        for idx, result in enumerate(results["results"], 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            published_date = result.get("publishedDate", "")
            author = result.get("author", "")
            
            snippet_parts = [f"{idx}. [{title}]({url})"]
            
            if author:
                snippet_parts.append(f"Author: {author}")
            if published_date:
                snippet_parts.append(f"Date: {published_date[:10]}")
            
            # Prefer highlights (AI-generated key points), then text, then snippet
            if include_contents:
                highlights = result.get("highlights", [])
                if highlights:
                    snippet_parts.append("\nKey points:")
                    for h in highlights[:3]:
                        snippet_parts.append(f"  • {h}")
                elif "text" in result:
                    text = result["text"][:500]
                    snippet_parts.append(f"\n{text}...")
            elif "snippet" in result:
                snippet_parts.append(f"\n{result['snippet']}")
            
            web_snippets.append("\n".join(snippet_parts))
        
        search_type = results.get("resolvedSearchType", "neural")
        category_info = f" (category: {category})" if category else ""
        content = f"Exa {search_type} search{category_info} for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n\n" + "\n\n".join(web_snippets)
        return content

    def call(self, params: Union[str, dict], **kwargs: Any) -> str:
        params_dict: Dict[str, Any]
        if isinstance(params, str):
            try:
                params_dict = json.loads(params)
            except json.JSONDecodeError:
                return "[Search] Invalid JSON input"
        else:
            params_dict = dict(params)
        
        query = params_dict.get("query")
        if not query:
            return "[Search] Invalid request: 'query' field is required"
        
        raw_num = params_dict.get("num_results", 10)
        num_results = int(raw_num) if raw_num is not None else 10
        include_contents = bool(params_dict.get("include_contents", False))
        category = params_dict.get("category")
        
        # Validate category if provided
        if category and category not in VALID_CATEGORIES:
            category = None
        
        if isinstance(query, str):
            return self.exa_search(query, num_results, include_contents, category)
        
        if isinstance(query, list):
            responses = []
            for q in query:
                responses.append(self.exa_search(q, num_results, include_contents, category))
            return "\n=======\n".join(responses)
        
        return "[Search] Invalid query format: must be string or array of strings"


if __name__ == "__main__":
    from dotenv import load_dotenv
    
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(env_path)
    
    searcher = Search()
    result = searcher.call({"query": ["What is retrieval augmented generation?"]})
    print(result)
