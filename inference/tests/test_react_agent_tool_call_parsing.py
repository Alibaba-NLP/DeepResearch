"""Regression test for the tool-call extraction in ``MultiTurnReactAgent._run``.

A model turn contains both the ``<think>`` reasoning and the final
``<tool_call>``. Models routinely restate the required tool-call format inside
their reasoning, so picking the *first* ``<tool_call>`` of the turn silently runs
a quoted example instead of the call the model settled on (issue #148).
"""

import os
import sys
import unittest

# The inference package uses sibling-relative imports (e.g. ``from react_agent import ...``)
# so we add the inference directory to sys.path for these tests.
HERE = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.dirname(HERE)
if INFERENCE_DIR not in sys.path:
    sys.path.insert(0, INFERENCE_DIR)

import react_agent  # noqa: E402
from react_agent import MultiTurnReactAgent  # noqa: E402

FINAL_TURN = "<think>done</think>\n<answer>Jane Doe</answer>"


class _RecordingTool:
    def __init__(self, name, log):
        self.name = name
        self._log = log

    def call(self, args, **kwargs):
        self._log.append((self.name, args.get("query")))
        return f"[{self.name}]"


class ToolCallExtractionTest(unittest.TestCase):
    def _run_turn(self, turn):
        calls = []
        original_tool_map = react_agent.TOOL_MAP
        react_agent.TOOL_MAP = {"search": _RecordingTool("search", calls)}
        try:
            agent = MultiTurnReactAgent(llm={"generate_cfg": {}, "model": "test-model"})
            turns = iter([turn, FINAL_TURN])
            agent.call_server = lambda msgs, port, max_tries=10: next(turns)
            agent.count_tokens = lambda messages: 0
            agent._run(
                {"item": {"question": "q", "answer": "a"}, "planning_port": 0},
                model="test-model",
            )
        finally:
            react_agent.TOOL_MAP = original_tool_map
        return calls

    def test_plain_turn_runs_its_only_tool_call(self):
        turn = (
            '<think>I should search.</think>\n'
            '<tool_call>\n{"name": "search", "arguments": {"query": ["real query"]}}\n</tool_call>'
        )
        self.assertEqual(self._run_turn(turn), [("search", ["real query"])])

    def test_example_quoted_in_think_is_not_executed(self):
        turn = (
            '<think>The required format is '
            '<tool_call>{"name": "search", "arguments": {"query": ["EXAMPLE"]}}</tool_call>'
            ' - now the real one.</think>\n'
            '<tool_call>\n{"name": "search", "arguments": {"query": ["real query"]}}\n</tool_call>'
        )
        self.assertEqual(self._run_turn(turn), [("search", ["real query"])])


if __name__ == "__main__":
    unittest.main()
