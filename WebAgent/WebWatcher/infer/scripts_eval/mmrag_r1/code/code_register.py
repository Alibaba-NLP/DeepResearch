from qwen_agent.tools import BaseTool
from .sandbox_module import PythonCodeExecutor

class CodeInterpreterTool(BaseTool):
    name = "code_interpreter"
    description = "Call this tool to execute Python code for calculation, data analysis, or content extraction tasks."
    file_access = False

    def __init__(self, timeout=50):
        self.executor = PythonCodeExecutor()

    def call(self, code: str, goal: str = "") -> dict:
        """
        Qwen-agent会自动传入code（和goal）。
        """
        result, raw_resp = self.executor.execute_code(code)
        return {"result": result}
