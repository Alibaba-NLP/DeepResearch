#!/usr/bin/env python3
"""
DeepResearch Interactive CLI - llama.cpp Server Edition
=========================================================

A powerful local research assistant that runs on YOUR machine.
Zero API costs. Full privacy. Complete control.

This script connects to a local llama.cpp server running the 
Tongyi-DeepResearch model and provides a full ReAct agent loop
with web search and page visiting capabilities.

Architecture:
    ┌─────────────────┐     HTTP      ┌──────────────────┐
    │  This Script    │ ────────────> │  llama.cpp       │
    │  (Agent Logic)  │               │  Server          │
    │  - Tool calls   │ <──────────── │  (Model loaded)  │
    │  - Web search   │     JSON      │  - Metal GPU     │
    │  - Page visits  │               │  - 32K context   │
    └─────────────────┘               └──────────────────┘

Usage:
    # Terminal 1: Start the server (one-time, stays running)
    ./scripts/start_llama_server.sh
    
    # Terminal 2: Run research queries
    python inference/interactive_llamacpp.py

Requirements:
    pip install requests duckduckgo-search python-dotenv

The server must be running before starting this script.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

import requests

from urllib.parse import urlparse

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass

# =============================================================================
# Configuration
# =============================================================================

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
JINA_API_KEY = os.environ.get("JINA_API_KEYS", "") or os.environ.get("JINA_API_KEY", "")

MAX_ROUNDS = 30
MAX_TOKENS = 4096
TEMPERATURE = 0.7
TOP_P = 0.95
REQUEST_TIMEOUT = 300  # 5 minutes for long generations

# Stop sequences for the ReAct loop
STOP_SEQUENCES = [
    "<tool_response>",
    "\n<tool_response>",
]

# =============================================================================
# System Prompt - Optimized for DeepResearch ReAct Agent
# =============================================================================

def get_system_prompt() -> str:
    return f"""You are a deep research assistant. Your task is to answer questions by searching the web and synthesizing information from credible sources.

# CRITICAL RULES

1. **Think deeply**: Use <think></think> tags to reason about what you know and what you need to find
2. **Search strategically**: Use multiple targeted searches to gather comprehensive information
3. **Verify information**: Cross-reference facts across multiple sources
4. **Synthesize thoroughly**: Combine information from multiple sources into a coherent answer
5. **NEVER visit the same URL twice**: Each URL can only be visited once
6. **Always conclude**: After gathering sufficient info (typically 5-15 sources), provide your answer in <answer></answer> tags
7. **Be efficient**: Aim to answer in 10-20 rounds

# Response Format

When you need to search, respond with:
<think>What I need to find and why</think>
<tool_call>
{{"name": "search", "arguments": {{"query": ["your search query"]}}}}
</tool_call>

When you need to visit a page for details:
<think>Why I need to visit this specific page</think>
<tool_call>
{{"name": "visit", "arguments": {{"url": "https://example.com", "goal": "what specific info you need"}}}}
</tool_call>

When you have enough information, respond with:
<think>Summary of what I found and my analysis</think>
<answer>Your comprehensive, well-researched answer with citations where appropriate</answer>

# Tools

<tools>
{{"type": "function", "function": {{"name": "search", "description": "Web search. Returns titles, URLs, and snippets.", "parameters": {{"type": "object", "properties": {{"query": {{"type": "array", "items": {{"type": "string"}}, "description": "1-3 search queries"}}}}, "required": ["query"]}}}}}}
{{"type": "function", "function": {{"name": "visit", "description": "Visit a URL to get full page content. Each URL can only be visited ONCE.", "parameters": {{"type": "object", "properties": {{"url": {{"type": "string", "description": "URL to visit"}}, "goal": {{"type": "string", "description": "What info you need"}}}}, "required": ["url", "goal"]}}}}}}
</tools>

# Important Notes
- The visit tool returns the COMPLETE page content in one response
- After 8-10 successful source visits, you likely have enough information to answer
- Prefer quality over quantity - don't just collect sources, synthesize them

Current date: {datetime.now().strftime("%Y-%m-%d")}"""


# =============================================================================
# Tools - DuckDuckGo Search (FREE, no API key needed!)
# =============================================================================

def duckduckgo_search(queries: list, num_results: int = 10) -> str:
    """Search using DuckDuckGo - completely free, no API key needed."""
    try:
        from duckduckgo_search import DDGS
        from duckduckgo_search.exceptions import RatelimitException
    except ImportError:
        return "[Search Error] duckduckgo-search not installed. Run: pip install duckduckgo-search"
    
    results = []
    for query in queries[:3]:
        retries = 3
        for attempt in range(retries):
            try:
                with DDGS() as ddgs:
                    search_results = list(ddgs.text(query, max_results=num_results))
                
                output = [f"\n## Search: '{query}'\n"]
                for idx, r in enumerate(search_results, 1):
                    title = r.get("title", "No title")
                    url = r.get("href", r.get("link", ""))
                    snippet = r.get("body", "")[:300]
                    output.append(f"{idx}. [{title}]({url})")
                    if snippet:
                        output.append(f"   {snippet}...")
                
                results.append("\n".join(output))
                break  # Success, exit retry loop
            except RatelimitException:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    continue
                results.append(f"[Search Error for '{query}']: Rate limited. Try again in a few seconds.")
            except Exception as e:
                results.append(f"[Search Error for '{query}']: {e}")
                break
    
    return "\n".join(results) if results else "No results found"


def is_valid_url(url: str) -> bool:
    """Check if URL is valid and uses http/https scheme."""
    try:
        result = urlparse(url)
        return all([result.scheme in ('http', 'https'), result.netloc])
    except Exception:
        return False


def visit_page(url: str, goal: str) -> str:
    """Fetch webpage content using Jina Reader (free tier) or direct fetch."""
    if isinstance(url, list):
        url = url[0] if url else ""
    
    if not url:
        return "[Visit Error] No URL provided"
    
    if not is_valid_url(url):
        return f"[Visit Error] Invalid URL: {url}. Must be a valid http/https URL."
    
    # Try Jina Reader first (free tier available)
    try:
        headers = {"Accept": "text/plain"}
        if JINA_API_KEY:
            headers["Authorization"] = f"Bearer {JINA_API_KEY}"
        
        jina_url = f"https://r.jina.ai/{url}"
        response = requests.get(jina_url, headers=headers, timeout=30, allow_redirects=True)
        
        if response.status_code == 200 and len(response.text) > 100:
            content = response.text[:12000]  # Increased limit for more context
            return f"**Content from {url}** (goal: {goal}):\n\n{content}"
    except Exception:
        pass
    
    # Fallback to direct fetch
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        text = response.text
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) > 100:
            return f"**Content from {url}** (goal: {goal}):\n\n{text[:12000]}"
        return f"[Visit Error] Page content too short or blocked: {url}"
    except requests.Timeout:
        return f"[Visit Error] Timeout fetching {url}"
    except Exception as e:
        return f"[Visit Error] Could not fetch {url}: {type(e).__name__}: {e}"


# =============================================================================
# llama.cpp Server Client
# =============================================================================

class LlamaCppClient:
    """Client for the llama.cpp OpenAI-compatible API."""
    
    def __init__(self, base_url: str = LLAMA_SERVER_URL):
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/v1/chat/completions"
        self.session = requests.Session()
    
    def check_server(self) -> bool:
        """Check if the server is running and responsive."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate(self, messages: List[Dict[str, str]], 
                 max_tokens: int = MAX_TOKENS,
                 temperature: float = TEMPERATURE,
                 top_p: float = TOP_P,
                 stop: Optional[List[str]] = None) -> str:
        """Generate a response from the llama.cpp server."""
        
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop or STOP_SEQUENCES,
            "stream": False,
        }
        
        try:
            response = self.session.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300,  # 5 minute timeout for long generations
            )
            
            if response.status_code != 200:
                error_text = response.text[:500]
                return f"[Server Error] Status {response.status_code}: {error_text}"
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return content.strip()
            
        except requests.Timeout:
            return "[Error] Request timed out. The model may be processing a complex query."
        except requests.ConnectionError:
            return "[Error] Cannot connect to llama.cpp server. Is it running?"
        except Exception as e:
            return f"[Error] API call failed: {e}"


# =============================================================================
# Research Agent
# =============================================================================

def research(client: LlamaCppClient, question: str, verbose: bool = True, 
             max_rounds: int = MAX_ROUNDS, temperature: float = TEMPERATURE) -> dict:
    """Run the research agent loop."""
    if verbose:
        print(f"\n🔍 Researching: {question}\n")
        print("-" * 60)
    
    system_prompt = get_system_prompt()
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    sources: List[Dict[str, Any]] = []
    thinking: List[str] = []
    visited_urls: set = set()
    consecutive_errors = 0
    max_consecutive_errors = 3
    start_time = time.time()
    
    for round_num in range(max_rounds):
        if verbose:
            print(f"\n📝 Round {round_num + 1}/{max_rounds}")
        
        gen_start = time.time()
        content = client.generate(messages, temperature=temperature)
        gen_time = time.time() - gen_start
        
        if content.startswith("[Error]") or content.startswith("[Server Error]"):
            if verbose:
                print(f"   ❌ {content}")
            break
        
        if verbose:
            print(f"   ⏱️  Generated in {gen_time:.1f}s")
        
        messages.append({"role": "assistant", "content": content})
        
        # Extract and display thinking
        if "<think>" in content and "</think>" in content:
            think_content = content.split("<think>")[1].split("</think>")[0].strip()
            thinking.append(think_content)
            if verbose:
                preview = think_content[:200] + "..." if len(think_content) > 200 else think_content
                print(f"   💭 {preview}")
        
        # Check for final answer
        if "<answer>" in content:
            if "</answer>" in content:
                answer = content.split("<answer>")[1].split("</answer>")[0]
            else:
                answer = content.split("<answer>")[1]
            
            elapsed = time.time() - start_time
            
            if verbose:
                print("\n" + "=" * 60)
                print("✅ ANSWER:")
                print("=" * 60)
                print(answer.strip())
                print("=" * 60)
                print(f"\n📊 Stats: {round_num + 1} rounds, {len(sources)} sources, {elapsed:.1f}s")
            
            return {
                "answer": answer.strip(),
                "sources": sources,
                "rounds": round_num + 1,
                "thinking": thinking,
                "elapsed_seconds": elapsed,
            }
        
        # Handle tool calls
        if "<tool_call>" in content and "</tool_call>" in content:
            try:
                tool_json = content.split("<tool_call>")[1].split("</tool_call>")[0]
                tool = json.loads(tool_json.strip())
                name = tool.get("name", "")
                args = tool.get("arguments", {})
                
                if verbose:
                    print(f"   🔧 Tool: {name}")
                
                tool_error = False
                
                if name == "search":
                    queries = args.get("query", [question])
                    if isinstance(queries, str):
                        queries = [queries]
                    if verbose:
                        print(f"      Searching: {queries}")
                    tool_result = duckduckgo_search(queries)
                    if "[Search Error" in tool_result:
                        tool_error = True
                    else:
                        sources.append({"type": "search", "queries": queries})
                
                elif name == "visit":
                    url = args.get("url", "")
                    if isinstance(url, list):
                        url = url[0] if url else ""
                    goal = args.get("goal", "extract information")
                    
                    if url in visited_urls:
                        if verbose:
                            print(f"      ⚠️ Already visited: {url}")
                        tool_result = f"[Already Visited] You already visited {url}. Use the information from your previous visit or try a different source."
                        tool_error = True
                    else:
                        visited_urls.add(url)
                        if verbose:
                            print(f"      Visiting: {url[:60]}...")
                        tool_result = visit_page(url, goal)
                        if "[Visit Error]" in tool_result:
                            tool_error = True
                        else:
                            sources.append({"type": "visit", "url": url})
                
                else:
                    tool_result = f"Unknown tool: {name}. Available tools: search, visit"
                    tool_error = True
                
                # Track consecutive errors for loop detection
                if tool_error:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        if verbose:
                            print(f"\n⚠️  {max_consecutive_errors} consecutive tool errors detected.")
                        messages.append({
                            "role": "user",
                            "content": f"<tool_response>\n{tool_result}\n\n[System Notice] You have encountered {consecutive_errors} consecutive errors. Please provide your best answer now based on information gathered so far, or try a completely different approach.\n</tool_response>"
                        })
                        continue
                else:
                    consecutive_errors = 0  # Reset on success
                
                # Inject tool response
                messages.append({
                    "role": "user", 
                    "content": f"<tool_response>\n{tool_result}\n</tool_response>"
                })
            
            except json.JSONDecodeError as e:
                consecutive_errors += 1
                messages.append({
                    "role": "user", 
                    "content": f"<tool_response>\nError: Invalid JSON in tool call: {e}\n</tool_response>"
                })
            except Exception as e:
                consecutive_errors += 1
                messages.append({
                    "role": "user", 
                    "content": f"<tool_response>\nTool error: {e}\n</tool_response>"
                })
    
    # Force final answer after max rounds
    if verbose:
        print("\n⚠️  Max rounds reached, requesting final answer...")
    
    messages.append({
        "role": "user",
        "content": "You have reached the maximum number of research rounds. Please provide your final answer now based on all the information gathered. Use <answer></answer> tags."
    })
    
    content = client.generate(messages, max_tokens=2048, temperature=temperature)
    
    if "<answer>" in content:
        if "</answer>" in content:
            answer = content.split("<answer>")[1].split("</answer>")[0]
        else:
            answer = content.split("<answer>")[1]
    else:
        answer = content
    
    elapsed = time.time() - start_time
    
    if verbose:
        print("\n" + "=" * 60)
        print("✅ ANSWER:")
        print("=" * 60)
        print(answer.strip())
        print("=" * 60)
        print(f"\n📊 Stats: {max_rounds} rounds (max), {len(sources)} sources, {elapsed:.1f}s")
    
    return {
        "answer": answer.strip(),
        "sources": sources,
        "rounds": max_rounds,
        "thinking": thinking,
        "elapsed_seconds": elapsed,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DeepResearch Interactive CLI - llama.cpp Server Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start the server first (in another terminal):
    ./scripts/start_llama_server.sh
    
    # Run interactive mode:
    python inference/interactive_llamacpp.py
    
    # Single query mode:
    python inference/interactive_llamacpp.py --query "What is quantum entanglement?"
    
    # Connect to a different server:
    python inference/interactive_llamacpp.py --server http://192.168.1.100:8080
"""
    )
    parser.add_argument("--server", type=str, default=LLAMA_SERVER_URL,
                        help="llama.cpp server URL (default: http://127.0.0.1:8080)")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="Single query mode - run one research query and exit")
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS,
                        help=f"Maximum research rounds (default: {MAX_ROUNDS})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Sampling temperature (default: {TEMPERATURE})")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🔬 DeepResearch - Interactive CLI")
    print("   llama.cpp Server Edition (100% Local)")
    print("=" * 60)
    print(f"Server:  {args.server}")
    print(f"Search:  DuckDuckGo (free, no API key)")
    print(f"Reader:  Jina.ai {'✓' if JINA_API_KEY else '(free tier)'}")
    print("=" * 60)
    
    # Initialize client
    client = LlamaCppClient(base_url=args.server)
    
    # Check server connection
    print("\nConnecting to llama.cpp server...", end=" ")
    if not client.check_server():
        print("❌ FAILED")
        print(f"\nError: Cannot connect to llama.cpp server at {args.server}")
        print("\nPlease start the server first:")
        print("    ./scripts/start_llama_server.sh")
        print("\nOr specify a different server URL:")
        print("    python inference/interactive_llamacpp.py --server http://your-server:8080")
        sys.exit(1)
    print("✅ Connected!")
    
    # Single query mode
    if args.query:
        research(client, args.query, max_rounds=args.max_rounds, temperature=args.temperature)
        return
    
    # Interactive mode
    print("\nType your research question (or 'quit' to exit):\n")
    
    while True:
        try:
            question = input("❓ Question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ('quit', 'exit', 'q'):
                print("\n👋 Goodbye!")
                break
            
            research(client, question, max_rounds=args.max_rounds, temperature=args.temperature)
            print("\n" + "-" * 60 + "\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
