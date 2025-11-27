import json
import os
import re
from typing import List, Union, Optional
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from prompt import EXTRACTOR_PROMPT 
from openai import OpenAI
from urllib.parse import urlparse
import time 
import tiktoken

VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))
JINA_API_KEYS = os.getenv("JINA_API_KEYS", "")

# Maximum content length to return when summarization fails
RAW_CONTENT_MAX_CHARS = int(os.getenv("RAW_CONTENT_MAX_CHARS", 8000))


def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    """Truncate text to a maximum number of tokens."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return encoding.decode(tokens[:max_tokens])
    except Exception:
        # Fallback: rough char estimate (4 chars per token)
        max_chars = max_tokens * 4
        return text[:max_chars] if len(text) > max_chars else text


def extract_main_content(html_text: str, max_chars: int = 8000) -> str:
    """
    Extract main content from HTML/markdown text.
    Removes boilerplate like navigation, footers, etc.
    """
    lines = html_text.split('\n')
    content_lines = []
    total_chars = 0
    
    skip_patterns = [
        r'^#{1,2}\s*(navigation|menu|footer|sidebar|cookie|privacy|terms)',
        r'^\s*(\||---)',  # Table separators
        r'^\s*\[.*\]\(.*\)\s*$',  # Standalone links
        r'^(copyright|©|\d{4}\s*[-–]\s*\d{4})',
    ]
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Skip empty lines at start
        if not content_lines and not line.strip():
            continue
            
        # Skip navigation/boilerplate patterns
        skip = False
        for pattern in skip_patterns:
            if re.match(pattern, line_lower):
                skip = True
                break
        
        if skip:
            continue
        
        content_lines.append(line)
        total_chars += len(line) + 1
        
        if total_chars >= max_chars:
            break
    
    return '\n'.join(content_lines)


@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit."
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s)."
            }
        },
        "required": ["url", "goal"]
    }

    def _validate_url(self, url: str) -> bool:
        """Check if URL is valid and has a proper scheme."""
        try:
            parsed = urlparse(url)
            return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
        except Exception:
            return False

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except Exception:
            return "[Visit] Invalid request: need 'url' and 'goal' fields"

        if isinstance(url, str):
            if not self._validate_url(url):
                return f"[Visit] Invalid URL: {url}. URL must start with http:// or https://"
            return self.readpage(url, goal)
        
        # Multiple URLs
        responses = []
        start = time.time()
        for u in url:
            if time.time() - start > 300:  # 5 min timeout for batch
                responses.append(f"[Timeout] Skipped: {u}")
                continue
            if not self._validate_url(u):
                responses.append(f"[Visit] Invalid URL: {u}")
                continue
            try:
                responses.append(self.readpage(u, goal))
            except Exception as e:
                responses.append(f"[Error] {u}: {e}")
        
        return "\n\n---\n\n".join(responses)

    def jina_fetch(self, url: str, timeout: int = 30) -> Optional[str]:
        """Fetch webpage content using Jina Reader API."""
        headers = {}
        if JINA_API_KEYS:
            headers["Authorization"] = f"Bearer {JINA_API_KEYS}"
        
        for attempt in range(3):
            try:
                resp = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=timeout
                )
                if resp.status_code == 200 and len(resp.text) > 100:
                    return resp.text
            except requests.RequestException:
                pass
            time.sleep(0.5)
        
        return None

    def direct_fetch(self, url: str, timeout: int = 20) -> Optional[str]:
        """Fallback: fetch directly with requests."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            
            # Basic HTML to text conversion
            text = resp.text
            # Remove script/style tags
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            # Clean whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text if len(text) > 100 else None
        except Exception:
            return None

    def summarize_content(self, content: str, goal: str) -> Optional[dict]:
        """Use LLM API to summarize content. Returns None if unavailable."""
        api_key = os.environ.get("API_KEY")
        api_base = os.environ.get("API_BASE")
        model = os.environ.get("SUMMARY_MODEL_NAME", "")
        
        if not api_key or not api_base:
            return None
        
        try:
            client = OpenAI(api_key=api_key, base_url=api_base)
            
            # Truncate content for summarization
            content = truncate_to_tokens(content, max_tokens=30000)
            
            prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
            
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            
            result = resp.choices[0].message.content
            if not result:
                return None
            
            # Parse JSON response
            result = result.replace("```json", "").replace("```", "").strip()
            
            # Try to extract JSON
            left = result.find('{')
            right = result.rfind('}')
            if left != -1 and right > left:
                result = result[left:right+1]
            
            return json.loads(result)
        except Exception as e:
            print(f"[visit] Summarization failed: {e}")
            return None

    def readpage(self, url: str, goal: str) -> str:
        """
        Read and process a webpage.
        
        Strategy:
        1. Try Jina Reader first (best for complex pages)
        2. Fallback to direct fetch if Jina fails
        3. Try LLM summarization if API available
        4. Return extracted raw content if summarization unavailable
        """
        # Step 1: Fetch content
        content = self.jina_fetch(url)
        
        if not content:
            content = self.direct_fetch(url)
        
        if not content:
            return self._format_error(url, goal, "Failed to fetch webpage content")
        
        # Step 2: Try summarization
        summary = self.summarize_content(content, goal)
        
        if summary and summary.get("evidence") and summary.get("summary"):
            return self._format_success(url, goal, summary["evidence"], summary["summary"])
        
        # Step 3: Fallback - return extracted raw content
        extracted = extract_main_content(content, RAW_CONTENT_MAX_CHARS)
        
        if len(extracted) < 100:
            return self._format_error(url, goal, "Page content too short or empty")
        
        return self._format_raw(url, goal, extracted)

    def _format_success(self, url: str, goal: str, evidence: str, summary: str) -> str:
        return f"""Content from {url} for goal: {goal}

**Evidence:**
{evidence}

**Summary:**
{summary}"""

    def _format_raw(self, url: str, goal: str, content: str) -> str:
        return f"""Content from {url} for goal: {goal}

**Raw Content (summarization unavailable):**
{content}

Note: Please extract the relevant information for your goal from the content above."""

    def _format_error(self, url: str, goal: str, reason: str) -> str:
        return f"""Could not retrieve content from {url}
Goal: {goal}
Reason: {reason}

Please try a different source or search query."""
