"""
Multi-Provider Web Search Tool
==============================

Implements a robust search fallback chain optimized for AI research:
  1. Exa.ai      - Best semantic/neural search ($10 free credits)
  2. Tavily      - Purpose-built for RAG/LLMs (1,000 free requests/month)
  3. Serper      - Google SERP results (2,500 free queries)
  4. DuckDuckGo  - Free forever, final fallback (no API key needed)

Each provider is tried in order. If one fails (rate limit, error, no key),
the next provider is attempted automatically.

Environment Variables:
  EXA_API_KEY      - Exa.ai API key (https://exa.ai/)
  TAVILY_API_KEY   - Tavily API key (https://tavily.com/)
  SERPER_KEY_ID    - Serper API key (https://serper.dev/)

If no API keys are set, DuckDuckGo is used as the default (free, no key needed).
"""

import json
import os
import time
from typing import List, Optional, Union

import requests
from qwen_agent.tools.base import BaseTool, register_tool


# API Keys from environment
EXA_API_KEY = os.environ.get("EXA_API_KEY", "").strip()
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
SERPER_KEY = os.environ.get("SERPER_KEY_ID", "").strip()

# Request timeouts (seconds)
REQUEST_TIMEOUT = 30


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    if not text:
        return False
    return any("\u4E00" <= char <= "\u9FFF" for char in text)


def sanitize_query(query: str) -> str:
    """Sanitize and validate a search query."""
    if not query:
        return ""
    # Strip whitespace and limit length
    return query.strip()[:500]


def format_results(query: str, results: List[dict], provider: str) -> str:
    """Format search results into a consistent markdown format."""
    if not results:
        return ""
    
    snippets = []
    for idx, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        date = r.get("date", "")
        
        # Build result entry
        entry = f"{idx}. [{title}]({url})"
        if date:
            entry += f"\nDate: {date}"
        if snippet:
            entry += f"\n{snippet}"
        snippets.append(entry)
    
    header = f"A search for '{query}' found {len(snippets)} results:\n\n## Web Results\n\n"
    return header + "\n\n".join(snippets)


# =============================================================================
# Search Providers
# =============================================================================

def search_exa(query: str, num_results: int = 10) -> Optional[str]:
    """
    Exa.ai - Neural/semantic search engine.
    Best for finding conceptually relevant results, not just keyword matches.
    
    API Docs: https://docs.exa.ai/reference/search
    """
    if not EXA_API_KEY:
        return None
    
    query = sanitize_query(query)
    if not query:
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
                "numResults": min(num_results, 100),  # API max is 100
                "type": "auto",  # Let Exa choose best search type
            },
            timeout=REQUEST_TIMEOUT,
        )
        
        # Handle error responses
        if response.status_code == 401:
            print("[Exa] Invalid or expired API key")
            return None
        if response.status_code == 429:
            print("[Exa] Rate limited - too many requests")
            return None
        if response.status_code == 402:
            print("[Exa] Payment required - credits exhausted")
            return None
        if response.status_code != 200:
            error_msg = response.text[:200] if response.text else "Unknown error"
            print(f"[Exa] Error {response.status_code}: {error_msg}")
            return None
        
        data = response.json()
        api_results = data.get("results", [])
        
        if not api_results:
            return None
        
        # Normalize results
        results = []
        for r in api_results:
            title = r.get("title") or "No title"
            url = r.get("url", "")
            text = r.get("text", "")
            published = r.get("publishedDate", "")
            
            # Truncate text for snippet
            snippet = text[:300] + "..." if len(text) > 300 else text
            date = published[:10] if published else ""
            
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "date": date,
            })
        
        return format_results(query, results, "Exa")
    
    except requests.Timeout:
        print("[Exa] Request timeout")
        return None
    except requests.ConnectionError:
        print("[Exa] Connection error - network issue")
        return None
    except json.JSONDecodeError:
        print("[Exa] Invalid JSON response")
        return None
    except Exception as e:
        print(f"[Exa] Unexpected error: {type(e).__name__}: {e}")
        return None


def search_tavily(query: str, num_results: int = 10) -> Optional[str]:
    """
    Tavily - Search API designed specifically for RAG and LLM applications.
    Returns AI-optimized snippets and supports advanced filtering.
    
    API Docs: https://docs.tavily.com/documentation/api-reference/endpoint/search
    """
    if not TAVILY_API_KEY:
        return None
    
    query = sanitize_query(query)
    if not query:
        return None
    
    try:
        # Tavily supports both Bearer token and api_key in body
        # Using Bearer token as it's more standard
        response = requests.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {TAVILY_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "max_results": min(num_results, 20),  # API max is 20
                "search_depth": "basic",  # Use basic (1 credit) vs advanced (2 credits)
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=REQUEST_TIMEOUT,
        )
        
        # Handle error responses
        if response.status_code == 401:
            print("[Tavily] Invalid or expired API key")
            return None
        if response.status_code == 429:
            print("[Tavily] Rate limited - too many requests")
            return None
        if response.status_code == 432:
            print("[Tavily] Plan limit exceeded - upgrade required")
            return None
        if response.status_code == 433:
            print("[Tavily] Pay-as-you-go limit exceeded")
            return None
        if response.status_code != 200:
            error_msg = response.text[:200] if response.text else "Unknown error"
            print(f"[Tavily] Error {response.status_code}: {error_msg}")
            return None
        
        data = response.json()
        api_results = data.get("results", [])
        
        if not api_results:
            return None
        
        # Normalize results
        results = []
        for r in api_results:
            title = r.get("title") or "No title"
            url = r.get("url", "")
            content = r.get("content", "")
            
            # Truncate content for snippet
            snippet = content[:300] + "..." if len(content) > 300 else content
            
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "date": "",
            })
        
        return format_results(query, results, "Tavily")
    
    except requests.Timeout:
        print("[Tavily] Request timeout")
        return None
    except requests.ConnectionError:
        print("[Tavily] Connection error - network issue")
        return None
    except json.JSONDecodeError:
        print("[Tavily] Invalid JSON response")
        return None
    except Exception as e:
        print(f"[Tavily] Unexpected error: {type(e).__name__}: {e}")
        return None


def search_serper(query: str, num_results: int = 10) -> Optional[str]:
    """
    Serper - Google Search API (SERP results).
    Fast and reliable Google search results.
    
    API Docs: https://serper.dev/
    """
    if not SERPER_KEY:
        return None
    
    query = sanitize_query(query)
    if not query:
        return None
    
    try:
        # Determine locale based on query content
        if contains_chinese(query):
            payload = {
                "q": query,
                "gl": "cn",
                "hl": "zh-cn",
                "num": min(num_results, 100),
            }
        else:
            payload = {
                "q": query,
                "gl": "us",
                "hl": "en",
                "num": min(num_results, 100),
            }
        
        # Use requests instead of http.client for consistency and better error handling
        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": SERPER_KEY,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        
        # Handle error responses
        if response.status_code == 401:
            print("[Serper] Invalid API key")
            return None
        if response.status_code == 429:
            print("[Serper] Rate limited")
            return None
        if response.status_code != 200:
            error_msg = response.text[:200] if response.text else "Unknown error"
            print(f"[Serper] Error {response.status_code}: {error_msg}")
            return None
        
        data = response.json()
        
        # Check for API-level errors
        if "error" in data:
            print(f"[Serper] API error: {data['error']}")
            return None
        
        organic = data.get("organic", [])
        if not organic:
            return None
        
        # Normalize results
        results = []
        for page in organic:
            title = page.get("title") or "No title"
            url = page.get("link", "")
            snippet_text = page.get("snippet", "")
            date = page.get("date", "")
            
            # Clean up snippet
            snippet = snippet_text.replace("Your browser can't play this video.", "").strip()
            
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "date": date,
            })
        
        return format_results(query, results, "Serper")
    
    except requests.Timeout:
        print("[Serper] Request timeout")
        return None
    except requests.ConnectionError:
        print("[Serper] Connection error - network issue")
        return None
    except json.JSONDecodeError:
        print("[Serper] Invalid JSON response")
        return None
    except Exception as e:
        print(f"[Serper] Unexpected error: {type(e).__name__}: {e}")
        return None


def search_duckduckgo(query: str, num_results: int = 10) -> Optional[str]:
    """
    DuckDuckGo - Free search with no API key required.
    Rate limited but reliable as a final fallback.
    """
    try:
        from duckduckgo_search import DDGS
        from duckduckgo_search.exceptions import RatelimitException, DuckDuckGoSearchException
    except ImportError:
        print("[DuckDuckGo] duckduckgo-search package not installed. Run: pip install duckduckgo-search")
        return None
    
    query = sanitize_query(query)
    if not query:
        return None
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                api_results = list(ddgs.text(query, max_results=min(num_results, 25)))
            
            if not api_results:
                return None
            
            # Normalize results
            results = []
            for r in api_results:
                title = r.get("title") or "No title"
                url = r.get("href") or r.get("link", "")
                body = r.get("body", "")
                
                # Truncate body for snippet
                snippet = body[:300] + "..." if len(body) > 300 else body
                
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "date": "",
                })
            
            return format_results(query, results, "DuckDuckGo")
        
        except RatelimitException:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"[DuckDuckGo] Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            print("[DuckDuckGo] Rate limited after all retries")
            return None
        except DuckDuckGoSearchException as e:
            print(f"[DuckDuckGo] Search error: {e}")
            return None
        except Exception as e:
            print(f"[DuckDuckGo] Unexpected error: {type(e).__name__}: {e}")
            return None
    
    return None


# =============================================================================
# Multi-Provider Search with Fallback
# =============================================================================

def multi_provider_search(query: str, num_results: int = 10) -> str:
    """
    Search using multiple providers with automatic fallback.
    
    Provider priority (by quality):
      1. Exa.ai      - Best semantic search
      2. Tavily      - Purpose-built for LLMs
      3. Serper      - Google SERP results
      4. DuckDuckGo  - Free fallback
    
    Returns the first successful result or an error message.
    """
    # Validate query
    query = sanitize_query(query)
    if not query:
        return "[Search] Empty query provided. Please provide a search term."
    
    providers = [
        ("Exa", search_exa),
        ("Tavily", search_tavily),
        ("Serper", search_serper),
        ("DuckDuckGo", search_duckduckgo),
    ]
    
    failed_providers = []
    
    for name, search_fn in providers:
        result = search_fn(query, num_results)
        if result:
            return result
        failed_providers.append(name)
    
    # All providers failed
    return f"No results found for '{query}'. Providers attempted: {', '.join(failed_providers)}. Try a different or simpler query."


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
        
        # Log which providers are available at initialization
        available = []
        if EXA_API_KEY:
            available.append("Exa")
        if TAVILY_API_KEY:
            available.append("Tavily")
        if SERPER_KEY:
            available.append("Serper")
        available.append("DuckDuckGo")  # Always available
        
        print(f"[Search] Initialized with providers: {', '.join(available)}")

    def call(self, params: Union[str, dict], **kwargs) -> str:
        # Handle string input (invalid)
        if isinstance(params, str):
            return "[Search] Invalid request: Input must be a JSON object with 'query' field, not a string."
        
        # Handle None or non-dict
        if not isinstance(params, dict):
            return "[Search] Invalid request: Input must be a JSON object with 'query' field."
        
        query = params.get("query")
        
        # Handle missing query
        if query is None:
            return "[Search] Missing 'query' field in request."
        
        # Handle single string query
        if isinstance(query, str):
            query = query.strip()
            if not query:
                return "[Search] Empty query string provided."
            return multi_provider_search(query)
        
        # Handle list of queries
        if isinstance(query, list):
            if not query:
                return "[Search] Empty query list provided."
            
            # Filter out empty strings
            valid_queries = [q.strip() for q in query if isinstance(q, str) and q.strip()]
            
            if not valid_queries:
                return "[Search] No valid queries in list (all empty or non-string)."
            
            responses = []
            for q in valid_queries:
                responses.append(multi_provider_search(q))
            
            return "\n=======\n".join(responses)
        
        # Invalid query type
        return f"[Search] Invalid 'query' type: expected string or array, got {type(query).__name__}."
