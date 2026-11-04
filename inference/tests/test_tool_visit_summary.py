import os
import sys
import unittest
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_visit import Visit


class VisitSummaryHardeningTestCase(unittest.TestCase):
    def setUp(self):
        self.tool = Visit()
        # Avoid the network-backed tiktoken encoding download in tests.
        patcher = mock.patch("tool_visit.truncate_to_tokens", side_effect=lambda text, max_tokens=95000: text)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_call_server_returns_empty_string_when_last_attempt_has_empty_content(self):
        """A summary model that returns empty content on the final attempt must yield '',
        never an implicit None (len(None) crashed readpage_jina's retry loop)."""
        fake_client = mock.MagicMock()
        fake_response = mock.MagicMock()
        fake_response.choices[0].message.content = ""
        fake_client.chat.completions.create.return_value = fake_response
        with mock.patch("tool_visit.OpenAI", return_value=fake_client):
            result = self.tool.call_server([{"role": "user", "content": "goal"}], max_retries=2)
        self.assertEqual(result, "")
        self.assertIsInstance(result, str)

    def test_call_server_returns_content_when_present(self):
        fake_client = mock.MagicMock()
        fake_response = mock.MagicMock()
        fake_response.choices[0].message.content = '{"evidence": "e", "summary": "s"}'
        fake_client.chat.completions.create.return_value = fake_response
        with mock.patch("tool_visit.OpenAI", return_value=fake_client):
            result = self.tool.call_server([{"role": "user", "content": "goal"}], max_retries=2)
        self.assertEqual(result, '{"evidence": "e", "summary": "s"}')

    def test_readpage_jina_survives_valid_json_of_wrong_shape(self):
        """A valid JSON reply that is not the expected object (e.g. a list) must fall
        back to the 'could not be processed' message instead of crashing on raw['evidence']."""
        with mock.patch.object(Visit, "html_readpage_jina", return_value="page content"), \
                mock.patch.object(Visit, "call_server", return_value='```json\n["a", "list"]\n```'):
            result = self.tool.readpage_jina("https://example.com", "goal")
        self.assertIn("could not be processed", result)

    def test_readpage_jina_uses_valid_summary(self):
        summary = {"evidence": "EV", "summary": "SU"}
        with mock.patch.object(Visit, "html_readpage_jina", return_value="page content"), \
                mock.patch.object(Visit, "call_server", return_value='```json\n{"evidence": "EV", "summary": "SU"}\n```'):
            result = self.tool.readpage_jina("https://example.com", "goal")
        self.assertIn("EV", result)
        self.assertIn("SU", result)

    def test_readpage_jina_refetches_strips_fences_on_retry(self):
        """A refetched reply wrapped in ```json fences must still parse."""
        replies = ['```json\nnot json\n```', '```json\n{"evidence": "E2", "summary": "S2"}\n```']
        calls = {"n": 0}

        def fake_call_server(msgs, max_retries=2):
            reply = replies[calls["n"]]
            calls["n"] += 1
            return reply

        with mock.patch.object(Visit, "html_readpage_jina", return_value="page content"), \
                mock.patch.object(Visit, "call_server", side_effect=fake_call_server):
            result = self.tool.readpage_jina("https://example.com", "goal")
        self.assertIn("E2", result)
        self.assertIn("S2", result)


if __name__ == "__main__":
    unittest.main()
