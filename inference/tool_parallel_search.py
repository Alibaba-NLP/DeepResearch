"""Anonymous Parallel Search MCP transport for the inference search tool."""

import asyncio
import json
from urllib.parse import urlsplit

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


MCP_URL = 'https://search.parallel.ai/mcp'
USER_AGENT = 'Tongyi-DeepResearch/ParallelSearch (+https://github.com/Alibaba-NLP/DeepResearch)'
SEARCH_TIMEOUT = 60


def format_search_result(result, query):
    if result.isError:
        return '[Search] Parallel could not complete the search. Please try again later.'
    payload = result.structuredContent
    if payload is None:
        for block in result.content:
            if block.type == 'text':
                try:
                    payload = json.loads(block.text)
                except (ValueError, TypeError):
                    continue
                if isinstance(payload, dict):
                    break
    if not isinstance(payload, dict) or not isinstance(payload.get('results'), list):
        return '[Search] Invalid response from Parallel.'

    snippets = []
    for page in payload['results']:
        if not isinstance(page, dict):
            continue
        url = page.get('url')
        try:
            valid_url = (isinstance(url, str) and urlsplit(url).scheme in ('http', 'https')
                         and bool(urlsplit(url).hostname))
        except ValueError:
            valid_url = False
        if not valid_url:
            continue
        title = page.get('title') or url
        excerpts = page.get('excerpts', [])
        if not isinstance(excerpts, list):
            excerpts = []
        text = '\n'.join(excerpt for excerpt in excerpts if isinstance(excerpt, str))
        date = page.get('publish_date')
        date_text = f'\nDate published: {date}' if date else ''
        snippets.append(f'{len(snippets) + 1}. [{title}]({url}){date_text}\n{text}')
    return (f"A web search for '{query}' found {len(snippets)} results:\n\n## Web Results\n"
            + '\n\n'.join(snippets))


async def parallel_search(query, session_id):
    async def execute():
        async with httpx.AsyncClient(
            headers={'User-Agent': USER_AGENT},
            timeout=SEARCH_TIMEOUT,
        ) as client:
            async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool('web_search', {
                        'objective': query,
                        'search_queries': [query],
                        'session_id': session_id,
                    })
                    return format_search_result(result, query)

    # Bound connection, initialization, tool execution and cleanup together.
    # Cancellation propagates through both MCP context managers.
    return await asyncio.wait_for(execute(), timeout=SEARCH_TIMEOUT)
