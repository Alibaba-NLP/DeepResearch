"""Regression test for the Serper response handling in ``google_scholar_with_serp``.

``res.read()`` and ``json.loads`` run outside every ``try`` block in
``google_scholar_with_serp``, so any non-JSON response body (an HTML 403/502
error page from a proxy, an empty body, a truncated response, ...) raises
``json.JSONDecodeError`` out of the tool. The method is otherwise designed to
never raise and to report problems as strings, and ``Scholar.call`` relies on
that contract for its batched queries: ``list(executor.map(...))`` discards the
results of every query in the batch when a single query raises.

These tests stub ``http.client.HTTPSConnection`` so no network is used.
"""

import json
import os
import sys
import types
import unittest
from unittest import mock

# The inference package uses sibling-relative imports (e.g. ``from tool_scholar import ...``)
# so we add the inference directory to sys.path for these tests.
HERE = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.dirname(HERE)
if INFERENCE_DIR not in sys.path:
    sys.path.insert(0, INFERENCE_DIR)

import tool_scholar  # noqa: E402
from tool_scholar import Scholar  # noqa: E402


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body


def _fake_http_module(responder):
    """Build a stand-in for the ``http.client`` module used by tool_scholar.

    ``responder`` maps the query (taken from the posted JSON payload) to the
    raw response body that the fake connection should return.
    """

    class _FakeConnection:
        def __init__(self, host):
            self._query = None

        def request(self, method, path, payload, headers):
            self._query = json.loads(payload)["q"]

        def getresponse(self):
            return _FakeResponse(responder(self._query))

    return types.SimpleNamespace(
        client=types.SimpleNamespace(HTTPSConnection=_FakeConnection)
    )


GOOD_BODY = json.dumps(
    {
        "organic": [
            {
                "title": "A Great Paper",
                "year": "2020",
                "publicationInfo": "Some Venue",
                "snippet": "An important result.",
                "pdfUrl": "https://example.com/paper.pdf",
                "citedBy": 42,
            }
        ]
    }
).encode("utf-8")


class SerperResponseParsingTest(unittest.TestCase):
    def test_non_json_body_returns_error_string_instead_of_raising(self):
        scholar = Scholar()
        fake_http = _fake_http_module(lambda q: b"<html>403 Forbidden</html>")
        with mock.patch.object(tool_scholar, "http", fake_http):
            result = scholar.google_scholar_with_serp("attention is all you need")
        self.assertIsInstance(result, str)
        self.assertIn("malformed response", result)
        self.assertNotIn("## Scholar Results", result)

    def test_empty_body_returns_error_string_instead_of_raising(self):
        scholar = Scholar()
        fake_http = _fake_http_module(lambda q: b"")
        with mock.patch.object(tool_scholar, "http", fake_http):
            result = scholar.google_scholar_with_serp("any query")
        self.assertIsInstance(result, str)
        self.assertIn("malformed response", result)

    def test_batch_survives_one_malformed_response(self):
        bodies = {
            "good query": GOOD_BODY,
            "bad query": b"<html>502 Bad Gateway</html>",
        }
        scholar = Scholar()
        fake_http = _fake_http_module(lambda q: bodies.get(q, b""))
        with mock.patch.object(tool_scholar, "http", fake_http):
            result = scholar.call({"query": ["good query", "bad query"]})
        self.assertIsInstance(result, str)
        # The well-formed query's results must survive the malformed sibling.
        self.assertIn("A Great Paper", result)
        self.assertIn("found 1 results", result)
        # The malformed query is reported as a failure string, not a crash.
        self.assertIn("bad query", result)

    def test_valid_json_still_renders_results(self):
        scholar = Scholar()
        fake_http = _fake_http_module(lambda q: GOOD_BODY)
        with mock.patch.object(tool_scholar, "http", fake_http):
            result = scholar.google_scholar_with_serp("good query")
        self.assertIn("## Scholar Results", result)
        self.assertIn("A Great Paper", result)
        self.assertIn("found 1 results", result)


if __name__ == "__main__":
    unittest.main()
