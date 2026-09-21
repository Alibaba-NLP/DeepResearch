"""Tests for ``is_correct_judgement`` in ``evaluation/evaluate_deepsearch_official.py``.

Every headline number this script prints (Pass@1 / Pass@3 / Best Pass@1) is gated by
this function, so a judge reply must only count as correct when the judge actually
said so.
"""

import os
import sys
import unittest

# evaluate_deepsearch_official.py uses sibling-relative imports (``from prompt import *``)
# so we add the evaluation directory to sys.path for these tests.
HERE = os.path.dirname(os.path.abspath(__file__))
EVALUATION_DIR = os.path.dirname(HERE)
if EVALUATION_DIR not in sys.path:
    sys.path.insert(0, EVALUATION_DIR)

from evaluate_deepsearch_official import is_correct_judgement  # noqa: E402


class IsCorrectJudgementTest(unittest.TestCase):
    def test_gaia_style_verdicts(self):
        # JUDGE_PROMPT_GAIA: 'respond with "Correct" ... or "Incorrect"'
        self.assertTrue(is_correct_judgement("Correct"))
        self.assertTrue(is_correct_judgement("correct\n"))
        self.assertFalse(is_correct_judgement("Incorrect"))

    def test_letter_grade_verdicts(self):
        # JUDGE_PROMPT_QA / JUDGE_PROMPT_SEAL0_QA: 'Just return the letters "A", "B", or "C"'
        self.assertTrue(is_correct_judgement("A"))
        self.assertFalse(is_correct_judgement("B"))
        self.assertFalse(is_correct_judgement("C"))

    def test_reduced_verdicts_from_call_llm_judge(self):
        # the xbench / browsecomp branches reduce the judge reply to "Correct" or ""
        self.assertTrue(is_correct_judgement("Correct"))
        self.assertFalse(is_correct_judgement(""))

    def test_incorrect_verdict_with_leading_text_is_not_counted_as_correct(self):
        # a judge that ignores 'Do not include any other text' must not be scored
        # correct just because its reply happens to start with the letter "a"
        for judgement in (
            "Actually, the predicted answer is not equivalent. Incorrect",
            "Answer: Incorrect",
            "Although similar, the predicted answer is incorrect",
            "As the answers differ: Incorrect",
        ):
            with self.subTest(judgement=judgement):
                self.assertFalse(is_correct_judgement(judgement))


if __name__ == "__main__":
    unittest.main()
