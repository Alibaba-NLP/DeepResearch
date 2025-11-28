"""
Multi-Provider Web Search Tool
==============================

Implements a robust search fallback chain optimized for AI research:
  1. Exa.ai     - Best semantic/neural search, $10 free credits
  2. Tavily    - Purpose-built for RAG/LLMs, 1,000 free requests/month
  3. Serper    - Google SERP results, 2,500 free queries
  4. DuckDuckGo - Free forever, final fallback (no API key needed)

Each provider is tried in order. If one fails (rate limit, error, no key),
the next provider is attempted automatically.

Environment Variables:
  EXA_API_KEY      - Exa.ai API key (https://exa.ai/)
  TAVILY_API_KEY   - Tavily API key (https://tavily.com/)
  SERPER_KEY_ID    - Serper API key (https://serper.dev/)

If no API keys are set, DuckDuckGo is used as the default (free, no key needed).
"""

import http.client
import json
import os
import time
from typing import Dict, List, Optional, Union

import requests
from qwen_agent.tools.base import BaseTool, register_tool


# API Keys from environment
EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
SERPER_KEY = os.environ.get("SERPER_KEY_ID", "")


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return any("\u4E00" <= char <= "\u9FFF" for char in text)


# =============================================================================
# Search Providers
# =============================================================================

def search_exa(query: str, num_results: int = 10) -> Optional[str]:
    """
    Exa.ai - Neural/semantic search engine.
    Best for finding conceptually relevant results, not just keyword matches.
    """
    if not EXA_API_KEY:
        return None
    
    try:
        response = requests.post(
            "https://api.exa.ai/search",
            headers={
                "x-api-key": EXA_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "numResults": num_results,
                "useAutoprompt": True,
                "type": "neural",
            },
            timeout=30,
        )
        
        if response.status_code == 401:
            print("[Exa] Invalid API key")
            return None
        if response.status_code == 429:
            print("[Exa] Rate limited")
            return None
        if response.status_code != 200:
            print(f"[Exa] Error {response.status_code}: {response.text[:200]}")
            return None
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            return None
        
        snippets = []
        for idx, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            text = r.get("text", "")[:300] if r.get("text") else ""
            published = r.get("publishedDate", "")
            
            snippet = f"{idx}. [{title}]({url})"
            if published:
                snippet += f"\nDate published: {published[:10]}"
            if text:
                snippet += f"\n{text}"
            snippets.append(snippet)
        
        return f"A search for '{query}' found {len(snippets)} results:\n\n## Web Results\n\n" + "\n\n".join(snippets)
    
    except requests.Timeout:
        print("[Exa] Request timeout")
        return None
    except Exception as e:
        print(f"[Exa] Error: {e}")
        return None


def search_tavily(query: str, num_results: int = 10) -> Optional[str]:
    """
    Tavily - Search API designed specifically for RAG and LLM applications.
    Returns AI-optimized snippets and supports advanced filtering.
    """
    if not TAVILY_API_KEY:
        return None
    
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": num_results,
                "search_depth": "advanced",
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=30,
        )
        
        if response.status_code == 401:
            print("[Tavily] Invalid API key")
            return None
        if response.status_code == 429:
            print("[Tavily] Rate limited")
            return None
        if response.status_code != 200:
            print(f"[Tavily] Error {response.status_code}: {response.text[:200]}")
            return None
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            return None
        
        snippets = []
        for idx, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            content = r.get("content", "")[:300]
            score = r.get("score", 0)
            
            snippet = f"{idx}. [{title}]({url})"
            if content:
                snippet += f"\n{content}"
            snippets.append(snippet)
        
        return f"A search for '{query}' found {len(snippets)} results:\n\n## Web Results\n\n" + "\n\n".join(snippets)
    
    except requests.Timeout:
        print("[Tavily] Request timeout")
        return None
    except Exception as e:
        print(f"[Tavily] Error: {e}")
        return None


def search_serper(query: str, num_results: int = 10) -> Optional[str]:
    """
    Serper - Google Search API (SERP results).
    Fast and reliable Google search results.
    """
    if not SERPER_KEY:
        return None
    
    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        
        if contains_chinese(query):
            payload = json.dumps({
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn",
                "num": num_results,
            })
        else:
            payload = json.dumps({
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en",
                "num": num_results,
            })
        
        headers = {
            "X-API-KEY": SERPER_KEY,
            "Content-Type": "application/json",
        }
        
        res = None
        for attempt in range(3):
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"[Serper] Connection error: {e}")
                    return None
                time.sleep(1)
                continue
        
        if res is None:
            return None
        
        data = json.loads(res.read().decode("utf-8"))
        
        if "error" in data:
            print(f"[Serper] API error: {data['error']}")
            return None
        
        if "organic" not in data:
            return None
        
        snippets = []
        for idx, page in enumerate(data["organic"], 1):
            title = page.get("title", "No title")
            url = page.get("link", "")
            snippet_text = page.get("snippet", "")
            date = page.get("date", "")
            source = page.get("source", "")
            
            result = f"{idx}. [{title}]({url})"
            if date:
                result += f"\nDate published: {date}"
            if source:
                result += f"\nSource: {source}"
            if snippet_text:
                result += f"\n{snippet_text}"
            
            result = result.replace("Your browser can't play this video.", "")
            snippets.append(result)
        
        return f"A search for '{query}' found {len(snippets)} results:\n\n## Web Results\n\n" + "\n\n".join(snippets)
    
    except Exception as e:
        print(f"[Serper] Error: {e}")
        return None


def search_duckduckgo(query: str, num_results: int = 10) -> Optional[str]:
    """
    DuckDuckGo - Free search with no API key required.
    Rate limited but reliable as a final fallback.
    """
    try:
        from duckduckgo_search import DDGS
        from duckduckgo_search.exceptions import RatelimitException
    except ImportError:
        print("[DuckDuckGo] duckduckgo-search package not installed")
        return None
    
    retries = 3
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
            
            if not results:
                return None
            
            snippets = []
            for idx, r in enumerate(results, 1):
                title = r.get("title", "No title")
                url = r.get("href", r.get("link", ""))
                body = r.get("body", "")[:300]
                
                snippet = f"{idx}. [{title}]({url})"
                if body:
                    snippet += f"\n{body}"
                snippets.append(snippet)
            
            return f"A search for '{query}' found {len(snippets)} results:\n\n## Web Results\n\n" + "\n\n".join(snippets)
        
        except RatelimitException:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            print("[DuckDuckGo] Rate limited after retries")
            return None
        except Exception as e:
            print(f"[DuckDuckGo] Error: {e}")
            return None
    
    return None


# =============================================================================
# Multi-Provider Search with Fallback
# =============================================================================

def multi_provider_search(query: str, num_results: int = 10) -> str:
    """
    Search using multiple providers with automatic fallback.
    
    Provider priority (by quality):
      1. Exa.ai     - Best semantic search
      2. Tavily     - Purpose-built for LLMs
      3. Serper     - Google SERP results
      4. DuckDuckGo - Free fallback
    
    Returns the first successful result or an error message.
    """
    providers = [
        ("Exa", search_exa),
        ("Tavily", search_tavily),
        ("Serper", search_serper),
        ("DuckDuckGo", search_duckduckgo),
    ]
    
    errors = []
    
    for name, search_fn in providers:
        result = search_fn(query, num_results)
        if result:
            return result
        errors.append(name)
    
    return f"No results found for '{query}'. All providers failed: {', '.join(errors)}. Try a different query."


# =============================================================================
# Qwen Agent Tool Registration
# =============================================================================

@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    """Web search tool with multi-provider fallback."""
    
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings. Include multiple complementary search queries in a single call.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        
        # Log which providers are available
        available = []
        if EXA_API_KEY:
            available.append("Exa")
        if TAVILY_API_KEY:
            available.append("Tavily")
        if SERPER_KEY:
            available.append("Serper")
        available.append("DuckDuckGo")
        
        print(f"[Search] Available providers: {', '.join(available)}")

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        params_dict: dict = params
        query = params_dict.get("query")
        if query is None:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            return multi_provider_search(query)
        
        if not isinstance(query, list):
            return "[Search] Invalid query format: 'query' must be a string or array of strings"
        
        responses = []
        for q in query:
            responses.append(multi_provider_search(q))
        
        return "\n=======\n".join(responses)
