"""
Evaluation integrity and contamination prevention for Deep Research Agents.

This module addresses the "Search-Time Data Contamination" issue described in #251:
when a Deep Research Agent processes benchmark questions, its search tools may
unwittingly retrieve benchmark ground-truth data (from HuggingFace mirrors, GitHub
gists, discussion forums, etc.) instead of solving the problem through reasoning.

This module provides:

1. **Domain blacklisting** — blocks known benchmark mirror domains during search
   and URL visitation.
2. **Query sanitization** — detects search queries that directly reference benchmark
   question text or known answer patterns.
3. **Content contamination detection** — scans fetched web content for embedded
   benchmark ground truth.
4. **Audit logging** — records all tool interactions for post-hoc contamination
   analysis, producing a per-question contamination report.
5. **Answer similarity scoring** — flags agent responses whose similarity to known
   ground-truth answers exceeds thresholds without supporting reasoning.

Usage:
    from evaluation.contamination_guard import ContaminationGuard

    guard = ContaminationGuard(benchmark_name="hle", benchmark_data_path="data/hle.jsonl")
    guard.load_benchmark_data()

    # Before executing the agent
    guard.start_audit(question_id="hle_q1", question_text="...")

    # Inside the search tool
    guard.check_query("What is the answer to ...")
    guard.check_domain("https://huggingface.co/datasets/...")

    # After fetching content
    guard.check_content(url, content)

    # After the agent produces its answer
    report = guard.end_audit(agent_answer="...")
    if report.contaminated:
        print(f"Contamination detected: {report.reason}")
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain blacklist
# ---------------------------------------------------------------------------

# Domains known to host benchmark data, ground-truth mirrors, or answer dumps.
DEFAULT_BLOCKED_DOMAINS: FrozenSet[str] = frozenset({
    "huggingface.co",
    "github.com",
    "githubusercontent.com",
    "gitlab.com",
    "pastebin.com",
    "gist.github.com",
    "raw.githubusercontent.com",
    "lastexam.ai",
    "hle-benchmark.github.io",
    "paperswithcode.com",
    "arxiv.org",
    "semanticscholar.org",
    "scholar.google.com",
    "kaggle.com",
    "colab.research.google.com",
    "gist.io",
    "notion.so",
    "notion.site",
    "substack.com",
    "medium.com",
    "twitter.com",
    "x.com",
    "reddit.com",
    "stackexchange.com",
    "stackoverflow.com",
    "quora.com",
    "discord.com",
    "telegram.org",
    "t.me",
})

# Domains that are generally safe and expected for research agents.
DEFAULT_ALLOWED_DOMAINS: FrozenSet[str] = frozenset({
    "wikipedia.org",
    "wikidata.org",
    "britannica.com",
    "nature.com",
    "science.org",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "ieee.org",
    "acm.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "who.int",
    "cdc.gov",
    "nih.gov",
    "nasa.gov",
    "noaa.gov",
    "usgs.gov",
    "census.gov",
    "bls.gov",
    "imf.org",
    "worldbank.org",
    "un.org",
    "oecd.org",
})


def extract_domain(url: str) -> str:
    """Extract the registrable domain from a URL."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        # Strip leading "www."
        if host.startswith("www."):
            host = host[4:]
        return host.lower()
    except Exception:
        return ""


def is_subdomain_of(domain: str, parent: str) -> bool:
    """Check if *domain* is a subdomain of (or equal to) *parent*."""
    return domain == parent or domain.endswith("." + parent)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ContaminationReport:
    """Report for a single audited question."""

    question_id: str
    question_text: str
    agent_answer: str = ""
    ground_truth: str = ""
    contaminated: bool = False
    reason: str = ""
    blocked_queries: List[str] = field(default_factory=list)
    blocked_urls: List[str] = field(default_factory=list)
    contaminated_urls: List[str] = field(default_factory=list)
    answer_similarity: float = 0.0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_text": self.question_text[:200],
            "agent_answer": self.agent_answer[:200],
            "contaminated": self.contaminated,
            "reason": self.reason,
            "blocked_queries": self.blocked_queries,
            "blocked_urls": self.blocked_urls,
            "contaminated_urls": self.contaminated_urls,
            "answer_similarity": round(self.answer_similarity, 4),
            "num_tool_calls": len(self.tool_calls),
        }


@dataclass
class QueryCheckResult:
    """Result of a search-query contamination check."""

    allowed: bool
    reason: str = ""
    matched_benchmark_text: str = ""


@dataclass
class ContentCheckResult:
    """Result of a content contamination check."""

    contaminated: bool
    reason: str = ""
    matched_snippets: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ContaminationGuard
# ---------------------------------------------------------------------------


class ContaminationGuard:
    """Guards against search-time data contamination during Deep Research evaluation.

    The guard maintains a set of known benchmark questions and ground-truth answers.
    During agent execution it:

    1. Blocks search queries that closely match benchmark question text.
    2. Blocks URL visits to known benchmark mirror domains.
    3. Detects benchmark ground-truth text in fetched web content.
    4. Scores the similarity between the agent's final answer and the ground truth.

    Parameters
    ----------
    benchmark_name : str
        Identifier for the benchmark being evaluated (e.g. "hle").
    benchmark_data_path : str, optional
        Path to a JSONL file containing benchmark items with at least
        ``question`` and ``answer`` fields.
    blocked_domains : set of str, optional
        Additional domains to block beyond the defaults.
    allowed_domains : set of str, optional
        Domains to always allow (overrides the default blacklist).
    similarity_threshold : float
        Minimum SequenceMatcher ratio between the agent answer and ground truth
        to flag as potentially contaminated (default 0.85).
    ngram_size : int
        Size of character n-grams used for partial-match detection (default 50).
    min_ngram_matches : int
        Minimum number of shared n-grams to flag content as contaminated (default 3).
    """

    def __init__(
        self,
        benchmark_name: str = "generic",
        benchmark_data_path: Optional[str] = None,
        blocked_domains: Optional[Set[str]] = None,
        allowed_domains: Optional[Set[str]] = None,
        similarity_threshold: float = 0.85,
        ngram_size: int = 50,
        min_ngram_matches: int = 3,
    ):
        self.benchmark_name = benchmark_name
        self.similarity_threshold = similarity_threshold
        self.ngram_size = ngram_size
        self.min_ngram_matches = min_ngram_matches

        # Domain sets
        self.blocked_domains: Set[str] = set(DEFAULT_BLOCKED_DOMAINS)
        if blocked_domains:
            self.blocked_domains |= blocked_domains
        self.allowed_domains: Set[str] = set(DEFAULT_ALLOWED_DOMAINS)
        if allowed_domains:
            self.allowed_domains |= allowed_domains

        # Benchmark data
        self.benchmark_questions: List[str] = []
        self.benchmark_answers: List[str] = []
        self.benchmark_ngrams: List[Set[str]] = []
        if benchmark_data_path:
            self.load_benchmark_data(benchmark_data_path)

        # Audit state
        self._current_report: Optional[ContaminationReport] = None

    # ------------------------------------------------------------------
    # Benchmark data loading
    # ------------------------------------------------------------------

    def load_benchmark_data(self, path: str) -> None:
        """Load benchmark questions and answers from a JSONL file."""
        self.benchmark_questions = []
        self.benchmark_answers = []
        self.benchmark_ngrams = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                q = item.get("question", item.get("problem", ""))
                a = item.get("answer", item.get("ground_truth", ""))
                if q:
                    self.benchmark_questions.append(q)
                    self.benchmark_answers.append(str(a))
                    self.benchmark_ngrams.append(self._char_ngrams(q))

        logger.info(
            "Loaded %d benchmark items from %s",
            len(self.benchmark_questions), path,
        )

    @staticmethod
    def _char_ngrams(text: str) -> Set[str]:
        """Extract normalized character n-grams from text."""
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        n = 8  # fixed char n-gram size for question matching
        return {
            normalized[i:i + n]
            for i in range(0, max(1, len(normalized) - n + 1), 4)
        }

    # ------------------------------------------------------------------
    # Domain checks
    # ------------------------------------------------------------------

    def check_domain(self, url: str) -> Tuple[bool, str]:
        """Check whether a URL's domain is allowed for this evaluation.

        Returns (allowed, reason).
        """
        domain = extract_domain(url)
        if not domain:
            return True, ""

        # Allow-list takes priority
        for allowed in self.allowed_domains:
            if is_subdomain_of(domain, allowed):
                return True, ""

        # Block-list check (including subdomains)
        for blocked in self.blocked_domains:
            if is_subdomain_of(domain, blocked):
                return False, f"Domain {domain} is blocked (matches {blocked})"

        return True, ""

    # ------------------------------------------------------------------
    # Query sanitization
    # ------------------------------------------------------------------

    def check_query(self, query: str) -> QueryCheckResult:
        """Check whether a search query attempts to retrieve benchmark data.

        A query is flagged if it shares a significant number of character
        n-grams with any benchmark question, suggesting the agent is searching
        for the exact problem statement.
        """
        if not self.benchmark_questions:
            return QueryCheckResult(allowed=True)

        query_ngrams = self._char_ngrams(query)

        for idx, bench_ngrams in enumerate(self.benchmark_ngrams):
            if not bench_ngrams:
                continue
            overlap = query_ngrams & bench_ngrams
            if len(overlap) >= self.min_ngram_matches:
                matched = sorted(overlap)[:3]
                return QueryCheckResult(
                    allowed=False,
                    reason=(
                        f"Query shares {len(overlap)} n-grams with benchmark "
                        f"question #{idx}. Matched fragments: {matched}"
                    ),
                    matched_benchmark_text=self.benchmark_questions[idx][:200],
                )

        return QueryCheckResult(allowed=True)

    # ------------------------------------------------------------------
    # Content contamination detection
    # ------------------------------------------------------------------

    def check_content(self, url: str, content: str) -> ContentCheckResult:
        """Check whether fetched web content contains benchmark ground truth.

        Scans the content for:
        1. Exact ground-truth answer strings.
        2. High character n-gram overlap with benchmark questions.
        """
        if not self.benchmark_answers:
            return ContentCheckResult(contaminated=False)

        content_lower = content.lower()
        matched_snippets: List[str] = []

        # Check for exact ground-truth answer strings
        for idx, answer in enumerate(self.benchmark_answers):
            answer_lower = str(answer).lower().strip()
            if len(answer_lower) < 4:
                continue
            if answer_lower in content_lower:
                start = content_lower.index(answer_lower)
                snippet = content[max(0, start - 40):start + len(answer_lower) + 40]
                matched_snippets.append(
                    f"[answer match, benchmark #{idx}] ...{snippet}..."
                )

        if matched_snippets:
            return ContentCheckResult(
                contaminated=True,
                reason=f"Content contains {len(matched_snippets)} ground-truth answer string(s)",
                matched_snippets=matched_snippets[:5],
            )

        # Check for high n-gram overlap with benchmark questions
        content_ngrams = self._char_ngrams(content[:5000])  # limit to first 5k chars
        for idx, bench_ngrams in enumerate(self.benchmark_ngrams):
            if not bench_ngrams:
                continue
            overlap = content_ngrams & bench_ngrams
            if len(overlap) >= self.min_ngram_matches * 2:  # stricter for content
                matched = sorted(overlap)[:3]
                matched_snippets.append(
                    f"[question match, benchmark #{idx}] {matched}"
                )

        if matched_snippets:
            return ContentCheckResult(
                contaminated=True,
                reason=f"Content shares n-grams with {len(matched_snippets)} benchmark question(s)",
                matched_snippets=matched_snippets[:5],
            )

        return ContentCheckResult(contaminated=False)

    # ------------------------------------------------------------------
    # Answer similarity scoring
    # ------------------------------------------------------------------

    def check_answer_similarity(self, agent_answer: str) -> Tuple[float, bool]:
        """Score the similarity between the agent answer and all ground truths.

        Returns (max_similarity, is_flagged).
        """
        if not self.benchmark_answers or not agent_answer:
            return 0.0, False

        max_sim = 0.0
        agent_norm = re.sub(r"\s+", " ", agent_answer.lower()).strip()

        for answer in self.benchmark_answers:
            answer_norm = re.sub(r"\s+", " ", str(answer).lower()).strip()
            sim = SequenceMatcher(None, agent_norm, answer_norm).ratio()
            max_sim = max(max_sim, sim)

        return max_sim, max_sim >= self.similarity_threshold

    # ------------------------------------------------------------------
    # Audit lifecycle
    # ------------------------------------------------------------------

    def start_audit(self, question_id: str, question_text: str,
                    ground_truth: str = "") -> ContaminationReport:
        """Start auditing a new question. Returns the audit report object."""
        self._current_report = ContaminationReport(
            question_id=question_id,
            question_text=question_text,
            ground_truth=ground_truth,
            timestamp_start=time.time(),
        )
        return self._current_report

    def record_tool_call(self, tool_name: str, tool_input: Dict[str, Any],
                         tool_output: str) -> None:
        """Record a tool call in the audit log and perform contamination checks."""
        if self._current_report is None:
            return

        entry: Dict[str, Any] = {
            "tool": tool_name,
            "input": {k: str(v)[:200] for k, v in tool_input.items()},
            "output_length": len(tool_output),
            "timestamp": time.time(),
        }

        # --- Search query checks ---
        if tool_name == "search":
            queries = tool_input.get("query", [])
            if isinstance(queries, str):
                queries = [queries]
            for q in queries:
                result = self.check_query(q)
                if not result.allowed:
                    entry["blocked_query"] = q
                    self._current_report.blocked_queries.append(q)
                self._current_report.tool_calls.append(entry)

        # --- URL visit checks ---
        elif tool_name == "visit":
            urls = tool_input.get("url", [])
            if isinstance(urls, str):
                urls = [urls]
            for u in urls:
                allowed, reason = self.check_domain(u)
                if not allowed:
                    entry["blocked_url"] = u
                    self._current_report.blocked_urls.append(u)
                else:
                    content_result = self.check_content(u, tool_output)
                    if content_result.contaminated:
                        entry["contaminated_url"] = u
                        self._current_report.contaminated_urls.append(u)
                        self._current_report.reason = (
                            f"Contaminated content fetched from {u}: "
                            f"{content_result.reason}"
                        )
                self._current_report.tool_calls.append(entry)
        else:
            self._current_report.tool_calls.append(entry)

    def end_audit(self, agent_answer: str = "") -> ContaminationReport:
        """End the audit for the current question and produce the final report."""
        if self._current_report is None:
            raise RuntimeError("start_audit() must be called before end_audit()")

        self._current_report.agent_answer = agent_answer
        self._current_report.timestamp_end = time.time()

        # Answer similarity check
        if self._current_report.ground_truth:
            sim, flagged = self.check_answer_similarity(agent_answer)
            self._current_report.answer_similarity = sim
            if flagged:
                self._current_report.contaminated = True
                if not self._current_report.reason:
                    self._current_report.reason = (
                        f"Agent answer similarity to ground truth is {sim:.2f} "
                        f"(threshold: {self.similarity_threshold})"
                    )

        # Flag if there were contaminated content fetches
        if self._current_report.contaminated_urls:
            self._current_report.contaminated = True

        report = self._current_report
        self._current_report = None
        return report

    # ------------------------------------------------------------------
    # Report persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_reports(reports: List[ContaminationReport], path: str) -> None:
        """Save a list of contamination reports to a JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for report in reports:
                f.write(json.dumps(report.to_dict(), ensure_ascii=False) + "\n")

    @staticmethod
    def summarize_reports(reports: List[ContaminationReport]) -> Dict[str, Any]:
        """Produce a summary of contamination reports."""
        total = len(reports)
        contaminated = sum(1 for r in reports if r.contaminated)
        blocked_queries = sum(len(r.blocked_queries) for r in reports)
        blocked_urls = sum(len(r.blocked_urls) for r in reports)
        contaminated_urls = sum(len(r.contaminated_urls) for r in reports)

        return {
            "total_questions": total,
            "contaminated_questions": contaminated,
            "contamination_rate": round(contaminated / total, 4) if total else 0.0,
            "total_blocked_queries": blocked_queries,
            "total_blocked_urls": blocked_urls,
            "total_contaminated_urls": contaminated_urls,
        }
