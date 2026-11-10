"""Audit logger for Deep Research Agent evaluation runs.

Records all tool interactions, contamination checks, and final reports
to structured JSONL files for post-hoc analysis.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """Writes structured audit logs for evaluation runs.

    Each evaluation run produces:
    - ``{run_id}.audit.jsonl`` — per-question contamination reports
    - ``{run_id}.summary.json`` — aggregate contamination statistics
    """

    def __init__(self, output_dir: str, run_id: str):
        self.output_dir = output_dir
        self.run_id = run_id
        os.makedirs(output_dir, exist_ok=True)
        self._audit_path = os.path.join(output_dir, f"{run_id}.audit.jsonl")
        self._reports: List[Dict[str, Any]] = []

    def write_report(self, report: Dict[str, Any]) -> None:
        """Append a contamination report to the audit log."""
        self._reports.append(report)
        with open(self._audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")

    def write_summary(self) -> Dict[str, Any]:
        """Write and return the aggregate summary for this run."""
        total = len(self._reports)
        contaminated = sum(1 for r in self._reports if r.get("contaminated", False))
        blocked_queries = sum(len(r.get("blocked_queries", [])) for r in self._reports)
        blocked_urls = sum(len(r.get("blocked_urls", [])) for r in self._reports)
        contaminated_urls = sum(len(r.get("contaminated_urls", [])) for r in self._reports)

        summary = {
            "run_id": self.run_id,
            "total_questions": total,
            "contaminated_questions": contaminated,
            "contamination_rate": round(contaminated / total, 4) if total else 0.0,
            "total_blocked_queries": blocked_queries,
            "total_blocked_urls": blocked_urls,
            "total_contaminated_urls": contaminated_urls,
            "timestamp": time.time(),
        }

        summary_path = os.path.join(self.output_dir, f"{self.run_id}.summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("Audit summary: %s", json.dumps(summary))
        return summary

    @property
    def reports(self) -> List[Dict[str, Any]]:
        return list(self._reports)
