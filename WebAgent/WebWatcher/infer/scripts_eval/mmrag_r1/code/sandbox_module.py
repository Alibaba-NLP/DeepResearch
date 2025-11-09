import os
from typing import Optional

def run_code_in_sandbox(code, timeout=50):
    """
    使用 HTTP API 调用沙箱服务编译运行 Python 代码
    需要跟你的 PythonInterpreter 工具后端兼容
    """
    SANDBOX_FUSION_ENDPOINT = os.environ.get('SANDBOX_FUSION_ENDPOINT', f'{CODE_SERVER_IP}:8080')
    if not (SANDBOX_FUSION_ENDPOINT.startswith("http://") or SANDBOX_FUSION_ENDPOINT.startswith("https://")):
        SANDBOX_FUSION_ENDPOINT = "http://" + SANDBOX_FUSION_ENDPOINT

    payload = {
        "code": code,
        "language": "python"
    }
    try:
        resp = requests.post(
            f"{SANDBOX_FUSION_ENDPOINT}/run_code",
            json=payload,
            timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
        # data 格式： {run_result: {stdout:...,stderr:...}, status:...}
    except Exception as e:
        return f"[Code Execution Error]: {e}", None

    stdout = data.get("run_result", {}).get("stdout", "")
    stderr = data.get("run_result", {}).get("stderr", "")
    result = ""
    if stdout.strip():
        result += f"stdout:\n{stdout}"
    if stderr.strip():
        result += f"\nstderr:\n{stderr}"
    return result if result.strip() else "Finished execution.", data

def extract_code_from_response(resp: str) -> Optional[str]:
    code_match = re.search(r"<code>([\s\S]+?)</code>", resp)
    if code_match:
        return code_match.group(1)

    code_block_match = re.search(r'```[^\n]*\n(.+?)```', resp, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1)
    return None

class PythonCodeExecutor:
    def __init__(self, timeout=50):
        self.timeout = timeout

    def execute_code(self, code):
        return run_code_in_sandbox(code, timeout=self.timeout)