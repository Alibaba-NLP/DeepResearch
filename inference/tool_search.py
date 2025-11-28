"""
Multi-Provider Search Tool for DeepResearch
Implements a fallback chain: Exa.ai → Serper → DuckDuckGo

Provider priority:
1. Exa.ai (best quality, semantic search with neural embeddings)
2. Serper.dev (Google results, reliable fallback)
3. DuckDuckGo (free, always available)

The tool automatically falls back to the next provider when:
- API key is not configured
- Rate limit is hit
- API errors occur
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool

# API Keys from environment
EXA_API_KEY = os.environ.get('EXA_API_KEY')
SERPER_API_KEY = os.environ.get('SERPER_API_KEY')

# API endpoints
EXA_BASE_URL = "https://api.exa.ai"
SERPER_BASE_URL = "https://google.serper.dev"

# Valid Exa categories
VALID_CATEGORIES = [
    "company", "research paper", "news", "pdf", 
    "github", "tweet", "personal site", "linkedin profile"
]


class SearchProviderError(Exception):
    """Raised when a search provider fails and should fallback."""
    pass


class ExaSearch:
    """Exa.ai semantic search provider."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def search(
        self,
        query: str,
        num_results: int = 10,
        include_contents: bool = False,
        category: Optional[str] = None
    ) -> str:
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
                if response is not None:
                    if response.status_code == 429:
                        raise SearchProviderError("Exa rate limited")
                    if response.status_code == 401:
                        raise SearchProviderError("Exa API key invalid")
                    if response.status_code == 402:
                        raise SearchProviderError("Exa credits exhausted")
                if attempt == 2:
                    raise SearchProviderError(f"Exa failed: {e}")
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    raise SearchProviderError(f"Exa failed: {e}")
                time.sleep(1)
        
        if response is None:
            raise SearchProviderError("Exa: no response")
        
        results = response.json()
        
        if "results" not in results or not results["results"]:
            return f"No results found for '{query}'."
        
        snippets = []
        for idx, r in enumerate(results["results"], 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            date = r.get("publishedDate", "")[:10] if r.get("publishedDate") else ""
            
            parts = [f"{idx}. [{title}]({url})"]
            if date:
                parts.append(f"Date: {date}")
            
            if include_contents:
                highlights = r.get("highlights", [])
                if highlights:
                    parts.append("Key points:")
                    for h in highlights[:3]:
                        parts.append(f"  • {h}")
                elif r.get("text"):
                    parts.append(r["text"][:500] + "...")
            elif r.get("snippet"):
                parts.append(r["snippet"])
            
            snippets.append("\n".join(parts))
        
        search_type = results.get("resolvedSearchType", "neural")
        cat_info = f" (category: {category})" if category else ""
        return f"[Exa {search_type}]{cat_info} '{query}' - {len(snippets)} results:\n\n" + "\n\n".join(snippets)


class SerperSearch:
    """Serper.dev Google search provider."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def search(self, query: str, num_results: int = 10) -> str:
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": min(num_results, 100)
        }
        
        response = None
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{SERPER_BASE_URL}/search",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                if response is not None:
                    if response.status_code == 429:
                        raise SearchProviderError("Serper rate limited")
                    if response.status_code in (401, 403):
                        raise SearchProviderError("Serper API key invalid")
                if attempt == 2:
                    raise SearchProviderError(f"Serper failed: {e}")
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    raise SearchProviderError(f"Serper failed: {e}")
                time.sleep(1)
        
        if response is None:
            raise SearchProviderError("Serper: no response")
        
        data = response.json()
        organic = data.get("organic", [])
        
        if not organic:
            return f"No results found for '{query}'."
        
        snippets = []
        for idx, r in enumerate(organic[:num_results], 1):
            title = r.get("title", "No title")
            url = r.get("link", "")
            snippet = r.get("snippet", "")
            
            parts = [f"{idx}. [{title}]({url})"]
            if snippet:
                parts.append(snippet)
            snippets.append("\n".join(parts))
        
        return f"[Serper/Google] '{query}' - {len(snippets)} results:\n\n" + "\n\n".join(snippets)


class DuckDuckGoSearch:
    """DuckDuckGo search provider (free, no API key needed)."""
    
    def search(self, query: str, num_results: int = 10) -> str:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise SearchProviderError("DuckDuckGo: duckduckgo_search not installed")
        
        results: List[Dict[str, Any]] = []
        for attempt in range(3):
            try:
                with DDGS() as ddg:
                    results = list(ddg.text(query, max_results=num_results))
                break
            except Exception as e:
                err = str(e).lower()
                if "ratelimit" in err or "429" in err:
                    if attempt < 2:
                        wait = 2 ** attempt
                        time.sleep(wait)
                        continue
                    raise SearchProviderError(f"DuckDuckGo rate limited after {attempt + 1} attempts")
                if attempt == 2:
                    raise SearchProviderError(f"DuckDuckGo failed: {e}")
                time.sleep(1)
        
        if not results:
            return f"No results found for '{query}'."
        
        snippets = []
        for idx, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("href", "")
            body = r.get("body", "")
            
            parts = [f"{idx}. [{title}]({url})"]
            if body:
                parts.append(body)
            snippets.append("\n".join(parts))
        
        return f"[DuckDuckGo] '{query}' - {len(snippets)} results:\n\n" + "\n\n".join(snippets)


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    """
    Multi-provider search tool with automatic fallback.
    
    Fallback chain: Exa.ai → Serper → DuckDuckGo
    """
    
    name = "search"
    description = "Search the web. Tries Exa.ai (semantic), then Serper (Google), then DuckDuckGo. Supply 'query' as string or array."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Query string or array of queries"
            },
            "num_results": {
                "type": "integer",
                "description": "Results per query (default: 10)",
                "default": 10
            },
            "include_contents": {
                "type": "boolean",
                "description": "Include page contents (Exa only)",
                "default": False
            },
            "category": {
                "type": "string",
                "description": "Category filter (Exa only): 'research paper', 'news', 'company', 'pdf', 'github'",
                "enum": VALID_CATEGORIES
            }
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.providers: List[tuple] = []
        
        # Build provider list based on available API keys
        if EXA_API_KEY:
            self.providers.append(("exa", ExaSearch(EXA_API_KEY)))
        if SERPER_API_KEY:
            self.providers.append(("serper", SerperSearch(SERPER_API_KEY)))
        # DuckDuckGo always available (no API key needed)
        self.providers.append(("duckduckgo", DuckDuckGoSearch()))
        
        if not self.providers:
            raise ValueError("No search providers available")

    def _search_with_fallback(
        self,
        query: str,
        num_results: int = 10,
        include_contents: bool = False,
        category: Optional[str] = None
    ) -> str:
        """Try each provider in order until one succeeds."""
        errors = []
        
        for name, provider in self.providers:
            try:
                if name == "exa":
                    return provider.search(query, num_results, include_contents, category)
                elif name == "serper":
                    return provider.search(query, num_results)
                else:  # duckduckgo
                    return provider.search(query, num_results)
            except SearchProviderError as e:
                errors.append(f"{name}: {e}")
                continue
        
        return f"[Search] All providers failed for '{query}':\n" + "\n".join(f"  - {e}" for e in errors)

    def call(self, params: Union[str, dict], **kwargs: Any) -> str:
        params_dict: Dict[str, Any]
        if isinstance(params, str):
            try:
                params_dict = json.loads(params)
            except json.JSONDecodeError:
                return "[Search] Invalid JSON"
        else:
            params_dict = dict(params)
        
        query = params_dict.get("query")
        if not query:
            return "[Search] 'query' is required"
        
        num_results = int(params_dict.get("num_results", 10) or 10)
        include_contents = bool(params_dict.get("include_contents", False))
        category = params_dict.get("category")
        
        if category and category not in VALID_CATEGORIES:
            category = None
        
        if isinstance(query, str):
            return self._search_with_fallback(query, num_results, include_contents, category)
        
        if isinstance(query, list):
            results = []
            for q in query:
                results.append(self._search_with_fallback(q, num_results, include_contents, category))
            return "\n=======\n".join(results)
        
        return "[Search] query must be string or array"


if __name__ == "__main__":
    from dotenv import load_dotenv
    
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(env_path)
    
    # Check available providers
    exa_key = os.environ.get('EXA_API_KEY')
    serper_key = os.environ.get('SERPER_API_KEY')
    
    print("Available providers:")
    if exa_key:
        print("  ✓ Exa.ai")
    if serper_key:
        print("  ✓ Serper.dev")
    print("  ✓ DuckDuckGo (always available)")
    print()
    
    searcher = Search()
    result = searcher.call({"query": ["What is retrieval augmented generation?"]})
    print(result)
