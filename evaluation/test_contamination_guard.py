"""Tests for the contamination guard (evaluation integrity).

Covers: domain checks, query sanitization, content contamination
detection, answer similarity scoring, audit lifecycle and report
serialization.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from contamination_guard import ContaminationGuard, extract_domain, is_subdomain_of


def _make_guard(tmpdir, questions=None, answers=None):
    """Build a ContaminationGuard backed by a small temp benchmark file."""
    if questions is None:
        questions = [
            "What is the revenue of Apple in fiscal year 2023 according to their 10-K filing?",
            "Who won the 2022 FIFA World Cup final?",
        ]
    if answers is None:
        answers = ["383.285 billion USD", "Argentina"]
    path = os.path.join(tmpdir, "bench.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for q, a in zip(questions, answers):
            f.write(json.dumps({"question": q, "answer": a}) + "\n")
    return ContaminationGuard(
        benchmark_name="test", benchmark_data_path=path
    )


class TestDomainHelpers(unittest.TestCase):
    def test_extract_domain_strips_www(self):
        self.assertEqual(extract_domain("https://www.example.com/a"), "example.com")

    def test_extract_domain_invalid(self):
        self.assertEqual(extract_domain("not a url"), "")

    def test_subdomain_of(self):
        self.assertTrue(is_subdomain_of("a.example.com", "example.com"))
        self.assertTrue(is_subdomain_of("example.com", "example.com"))
        self.assertFalse(is_subdomain_of("notexample.com", "example.com"))


class TestQueryCheck(unittest.TestCase):
    def test_benign_query_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            g = _make_guard(d)
            r = g.check_query("python asyncio gather tutorial")
            self.assertTrue(r.allowed)
            self.assertEqual(r.reason, "")

    def test_benchmark_question_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            g = _make_guard(d)
            r = g.check_query(
                "What is the revenue of Apple in fiscal year 2023 "
                "according to their 10-K filing?"
            )
            self.assertFalse(r.allowed)
            self.assertIn("n-grams", r.reason)

    def test_mild_paraphrase_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            g = _make_guard(d)
            r = g.check_query("how large is apple market cap today")
            self.assertTrue(r.allowed)


class TestContentCheck(unittest.TestCase):
    def test_ground_truth_answer_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            g = _make_guard(d)
            r = g.check_content(
                "https://example.com/p",
                "Apple 10-K reports revenue of 383.285 billion USD for FY2023.",
            )
            self.assertTrue(r.contaminated)
            self.assertIn("ground-truth", r.reason)

    def test_clean_content_not_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            g = _make_guard(d)
            r = g.check_content(
                "https://example.com/p",
                "Quarterly results vary across the technology sector.",
            )
            self.assertFalse(r.contaminated)

    def test_short_answers_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            # A 3-char answer must not trigger false positives.
            g = _make_guard(d, questions=["Who directed Citizen Kane?"],
                            answers=["Orson Welles"[:3]])
            r = g.check_content("https://example.com/p", "the cat orson sat")
            self.assertFalse(r.contaminated)


class TestDomainCheck(unittest.TestCase):
    def test_blocked_domain(self):
        with tempfile.TemporaryDirectory() as d:
            g = _make_guard(d)
            allowed, reason = g.check_domain("https://forum.telegram.org/t/1")
            self.assertFalse(allowed)
            self.assertIn("blocked", reason)

    def test_allowed_domain(self):
        with tempfile.TemporaryDirectory() as d:
            g = _make_guard(d)
            allowed, _ = g.check_domain("https://en.wikipedia.org/wiki/Apple")
            self.assertTrue(allowed)

    def test_unknown_domain_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            g = _make_guard(d)
            allowed, _ = g.check_domain("https://some-random-site.example/page")
            self.assertTrue(allowed)


class TestAnswerSimilarity(unittest.TestCase):
    def test_exact_match(self):
        with tempfile.TemporaryDirectory() as d:
            g = _make_guard(d)
            sim, flagged = g.check_answer_similarity("Argentina")
            self.assertGreater(sim, 0.99)
            self.assertTrue(flagged)

    def test_unrelated_answer(self):
        with tempfile.TemporaryDirectory() as d:
            g = _make_guard(d)
            sim, flagged = g.check_answer_similarity(
                "The paper proposes a novel sorting algorithm."
            )
            self.assertLess(sim, g.similarity_threshold)
            self.assertFalse(flagged)


class TestAuditLifecycle(unittest.TestCase):
    def test_full_lifecycle_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            g = _make_guard(d)
            g.start_audit("q1", "What is the revenue of Apple in fiscal year 2023?")
            g.record_tool_call("search", {"query": "apple revenue 2023"}, "ok" * 10)
            blocked = g.check_query(
                "What is the revenue of Apple in fiscal year 2023 according to their 10-K filing?"
            )
            g.record_tool_call("search", {"query": "blocked query"}, "")
            rep = g.end_audit(agent_answer="Argentina won the final")
            self.assertEqual(rep.question_id, "q1")
            self.assertEqual(len(rep.tool_calls), 2)

            out = os.path.join(d, "reports.jsonl")
            g.save_reports([rep], out)
            with open(out, encoding="utf-8") as f:
                loaded = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(loaded[0]["question_id"], "q1")

    def test_summary(self):
        with tempfile.TemporaryDirectory() as d:
            g = _make_guard(d)
            g.start_audit("q1", "question one")
            rep = g.end_audit(agent_answer="nothing")
            s = g.summarize_reports([rep])
            self.assertEqual(s["total_questions"], 1)


if __name__ == "__main__":
    unittest.main()
