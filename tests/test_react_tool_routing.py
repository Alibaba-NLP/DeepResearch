"""Exercise the ReAct loop with offline tools and scripted model responses."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def agent_module(monkeypatch):
    inference = Path(__file__).resolve().parents[1] / "inference"
    monkeypatch.syspath_prepend(str(inference))
    for module_name, class_name, tool_name in [
        ("tool_file", "FileParser", "parse_file"),
        ("tool_scholar", "Scholar", "google_scholar"),
        ("tool_python", "PythonInterpreter", "PythonInterpreter"),
        ("tool_search", "Search", "search"),
        ("tool_visit", "Visit", "visit"),
    ]:
        module = ModuleType(module_name)
        tool = SimpleNamespace(name=tool_name, call=Mock(return_value="observation"))
        setattr(module, class_name, Mock(return_value=tool))
        monkeypatch.setitem(sys.modules, module_name, module)
    spec = importlib.util.spec_from_file_location(
        "routing_test_agent", inference / "react_agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_tool_turn(module, tool_call):
    agent = module.MultiTurnReactAgent(llm={"model": "test-model", "generate_cfg": {}})
    turns = iter(
        [
            f"<think>Use a tool.</think><tool_call>{tool_call}</tool_call>",
            "<think>Done.</think><answer>done</answer>",
        ]
    )
    agent.call_server = lambda *args: next(turns)
    agent.count_tokens = lambda messages: 0
    return agent._run(
        {"item": {"question": "test", "answer": "done"}, "planning_port": 0},
        "test-model",
    )


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("search", {"query": ["Python release history"]}),
        (
            "visit",
            {"url": ["https://docs.python.org/"], "goal": "Read the documentation"},
        ),
        ("google_scholar", {"query": ["Python program synthesis"]}),
        ("search", {"query": ["ordinary query"]}),
    ],
)
def test_arguments_do_not_change_tool_routing(agent_module, name, arguments):
    result = run_tool_turn(
        agent_module, json.dumps({"name": name, "arguments": arguments})
    )
    agent_module.TOOL_MAP[name].call.assert_called_once()
    agent_module.TOOL_MAP["PythonInterpreter"].call.assert_not_called()
    assert result["prediction"] == "done"


def test_python_header_followed_by_code_still_executes(agent_module):
    run_tool_turn(
        agent_module,
        '{"name": "PythonInterpreter", "arguments": {}}\n<code>print(1)</code>',
    )
    agent_module.TOOL_MAP["PythonInterpreter"].call.assert_called_once_with("print(1)")


def test_unknown_python_named_tool_is_not_an_interpreter_alias(agent_module):
    result = run_tool_turn(agent_module, '{"name": "python_docs", "arguments": {}}')
    agent_module.TOOL_MAP["PythonInterpreter"].call.assert_not_called()
    assert "Tool python_docs not found" in result["messages"][3]["content"]


def test_python_without_code_returns_formatting_error(agent_module):
    result = run_tool_turn(
        agent_module, '{"name": "PythonInterpreter", "arguments": {}}'
    )
    agent_module.TOOL_MAP["PythonInterpreter"].call.assert_not_called()
    assert "Formatting error" in result["messages"][3]["content"]
