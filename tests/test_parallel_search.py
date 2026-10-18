import asyncio
import json
import os
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
import tool_parallel_search as parallel
from tool_search import Search
from search_agent_fixture import load_agent


RESULT = {'results': [{'title': 'Example source', 'url': 'https://example.org/evidence',
                       'publish_date': '2026-01-02', 'excerpts': ['Useful evidence.']}]}


class MCPFixture:
    """Exercise the real MCP SDK over a deterministic HTTP transport."""

    def __init__(self, result=None, error=None, status=200, block_call=False):
        self.requests = []
        self.result = result or {'content': [{'type': 'text', 'text': json.dumps(RESULT)}]}
        self.error = error
        self.status = status
        self.block_call = block_call
        self.call_started = asyncio.Event()
        self.call_closed = asyncio.Event()

    async def handle(self, request):
        message = json.loads(request.content)
        self.requests.append((request, message))
        method = message['method']
        if method == 'notifications/initialized':
            return httpx.Response(202)
        if method == 'initialize':
            result = {'protocolVersion': message['params']['protocolVersion'],
                      'capabilities': {'tools': {}},
                      'serverInfo': {'name': 'search-fixture', 'version': '1'}}
        elif method == 'tools/list':
            result = {'tools': [{'name': 'web_search', 'inputSchema': {'type': 'object'}}]}
        else:
            if self.block_call:
                self.call_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.call_closed.set()
            if self.error:
                raise self.error
            if self.status != 200:
                return httpx.Response(self.status, text='private server diagnostic')
            result = self.result
        return httpx.Response(200, json={'jsonrpc': '2.0', 'id': message['id'], 'result': result})

    @asynccontextmanager
    async def transport(self, *args, **kwargs):
        # Keep the production client headers and timeout; replace only network I/O.
        client = kwargs.pop('http_client')
        async with httpx.AsyncClient(headers=client.headers, timeout=client.timeout,
                                     transport=httpx.MockTransport(self.handle)) as fixture_client, \
                streamable_http_client(*args, **kwargs, http_client=fixture_client) as streams:
            yield streams

    def tool_calls(self):
        return [message['params'] for _, message in self.requests if message['method'] == 'tools/call']


class SearchTests(unittest.TestCase):
    def setUp(self):
        # Clear all ambient provider credentials/configuration for anonymous proof.
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_unconfigured_and_placeholder_default(self):
        self.assertEqual(Search().search_provider(), 'parallel')
        os.environ['SERPER_KEY_ID'] = 'your_key'
        self.assertEqual(Search().search_provider(), 'parallel')

    def test_configured_serper_and_explicit_choices(self):
        os.environ['SERPER_KEY_ID'] = 'configured-key'
        tool = Search()
        with patch.object(tool, 'search_with_serp', return_value='serper result') as serper:
            self.assertEqual(tool.call({'query': ['test query']}), 'serper result')
            serper.assert_called_once_with('test query')
        os.environ['SEARCH_PROVIDER'] = 'parallel'
        self.assertEqual(Search().search_provider(), 'parallel')
        self.assertEqual(Search({'provider': 'serper'}).search_provider(), 'serper')
        os.environ['SEARCH_PROVIDER'] = 'serper'
        self.assertEqual(Search().search_provider(), 'serper')

    def test_disabled_and_invalid_provider_do_not_send_requests(self):
        with patch('tool_search.Search.search_with_parallel') as send:
            os.environ['SEARCH_PROVIDER'] = 'none'
            self.assertIn('disabled', Search().call({'query': 'test'}))
            os.environ['SEARCH_PROVIDER'] = 'typo'
            self.assertIn('Unknown', Search().call({'query': 'test'}))
            send.assert_not_called()

    def test_explicit_serper_without_key_stays_serper(self):
        os.environ['SEARCH_PROVIDER'] = 'serper'
        self.assertIn('requires SERPER_KEY_ID', Search().call({'query': 'test'}))

    def test_invalid_arguments(self):
        for query in ([], [42], [''], ['  '], 123, None, {}):
            self.assertIn('[Search]', Search().call({'query': query}))
        for params in ('invalid json', '{}', None, []):
            self.assertIn('[Search]', Search().call(params))

    def test_anonymous_http_requests_and_batched_result(self):
        fixture = MCPFixture()
        with patch.object(parallel, 'streamable_http_client', fixture.transport):
            output = Search().call(json.dumps({'query': ['first query', 'second query']}),
                                   search_session_id='a' * 32)
        self.assertEqual(output.count('[Example source](https://example.org/evidence)'), 2)
        self.assertIn('Useful evidence.', output)
        self.assertIn('Date published: 2026-01-02', output)
        calls = fixture.tool_calls()
        self.assertEqual([c['arguments']['search_queries'] for c in calls],
                         [['first query'], ['second query']])
        self.assertTrue(all(c['name'] == 'web_search' and c['arguments']['session_id'] == 'a' * 32
                            for c in calls))
        for request, _ in fixture.requests:
            self.assertEqual(str(request.url), 'https://search.parallel.ai/mcp')
            self.assertEqual(request.headers['user-agent'], parallel.USER_AGENT)
            self.assertNotIn('authorization', request.headers)

    def test_failure_and_timeout_do_not_fallback_or_leak(self):
        for fixture in (MCPFixture(result={'isError': True, 'content': [
                {'type': 'text', 'text': 'private server diagnostic'}]}),
                MCPFixture(error=httpx.ReadTimeout('secret transport diagnostic')),
                *(MCPFixture(status=status) for status in (401, 429, 503))):
            with patch.object(parallel, 'streamable_http_client', fixture.transport), \
                    patch.object(Search, 'search_with_serp') as serper:
                output = Search().call({'query': 'test'})
            self.assertIn('[Search]', output)
            self.assertNotIn('diagnostic', output)
            serper.assert_not_called()

    def test_result_formats_urls_and_empty_results(self):
        result = CallToolResult(content=[], structuredContent=RESULT)
        self.assertIn('https://example.org/evidence', parallel.format_search_result(result, 'q'))
        invalid = {'results': [{'url': 'javascript:alert(1)'}, {'url': 'https://['}, 'bad']}
        result = CallToolResult(content=[TextContent(type='text', text=json.dumps(invalid))])
        self.assertIn('found 0 results', parallel.format_search_result(result, 'q'))
        result = CallToolResult(content=[TextContent(type='text', text='bad json')])
        self.assertIn('Invalid response', parallel.format_search_result(result, 'q'))

    def test_scripted_model_loop_consumes_search_and_preserves_permission_boundary(self):
        agent_module = load_agent()
        agent = agent_module.MultiTurnReactAgent(llm={'model': 'scripted', 'generate_cfg': {}})
        fixture = MCPFixture()
        observations = []

        def model(messages, port):
            if len(messages) == 2:
                return '<tool_call>{"name":"search","arguments":{"query":["evidence query"]}}</tool_call>'
            observations.append(messages[-1]['content'])
            return '<answer>Answer using https://example.org/evidence</answer>'

        with patch.object(parallel, 'streamable_http_client', fixture.transport), \
                patch.object(agent, 'call_server', side_effect=model), \
                patch.object(agent, 'count_tokens', return_value=100):
            for _ in range(2):
                result = agent._run({'item': {'question': 'Find evidence', 'answer': ''},
                                     'planning_port': 0}, 'scripted')
                self.assertIn('https://example.org/evidence', result['prediction'])
        self.assertEqual(len(observations), 2)
        self.assertTrue(all('Useful evidence.' in text and '<tool_response>' in text
                            for text in observations))
        ids = [call['arguments']['session_id'] for call in fixture.tool_calls()]
        self.assertEqual(len(set(ids)), 2)
        with patch.dict(agent_module.TOOL_MAP, {}, clear=True), \
                patch.object(Search, 'call') as send:
            self.assertIn('not found', agent.custom_call_tool('search', {'query': ['test']}))
            send.assert_not_called()


class AsyncSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_in_flight_sdk_request(self):
        fixture = MCPFixture(block_call=True)
        with patch.object(parallel, 'streamable_http_client', fixture.transport):
            task = asyncio.create_task(parallel.parallel_search('test', 'c' * 32))
            await asyncio.wait_for(fixture.call_started.wait(), timeout=5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(fixture.call_closed.is_set())

    async def test_deadline_and_cancellation_close_transport(self):
        entered = asyncio.Event()
        closed = asyncio.Event()

        @asynccontextmanager
        async def hanging_transport(*args, **kwargs):
            try:
                entered.set()
                await asyncio.Event().wait()
                yield None
            finally:
                closed.set()

        with patch.object(parallel, 'streamable_http_client', hanging_transport), \
                patch.object(parallel, 'SEARCH_TIMEOUT', 0.02):
            with self.assertRaises(asyncio.TimeoutError):
                await parallel.parallel_search('test', 'b' * 32)
            self.assertTrue(closed.is_set())
        closed.clear()
        entered.clear()
        with patch.object(parallel, 'streamable_http_client', hanging_transport):
            task = asyncio.create_task(parallel.parallel_search('test', 'b' * 32))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(closed.is_set())


if __name__ == '__main__':
    unittest.main()
