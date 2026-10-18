"""Load the real ReAct loop without unrelated file, sandbox or page tools.

The search registry, dispatcher and observation handling use production code.
Only model generation/token counting and unused tools are replaced by tests.
"""

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))


def load_agent():
    modules = {}
    for module_name, class_name, tool_name in (
        ('tool_file', 'FileParser', 'parse_file'),
        ('tool_scholar', 'Scholar', 'google_scholar'),
        ('tool_python', 'PythonInterpreter', 'PythonInterpreter'),
        ('tool_visit', 'Visit', 'visit'),
    ):
        module = types.ModuleType(module_name)
        setattr(module, class_name, type(class_name, (), {'name': tool_name}))
        modules[module_name] = module
    with patch.dict(sys.modules, modules):
        return importlib.import_module('react_agent')
