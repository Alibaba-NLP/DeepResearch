Run the search integration tests from the repository root:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -c requirements.txt mcp qwen-agent transformers openai python-dateutil
.venv/bin/python -m unittest discover -s tests -v
```

This focused installation uses the project's pinned dependencies without installing
the GPU inference stack. The tests exercise provider selection, the MCP SDK over
an in-memory HTTP transport, and the real ReAct tool dispatcher and observations.
Model replies and token counting are scripted; unrelated file, sandbox, Scholar
and page-reading tools are replaced by fixtures. No model weights, API keys or
live network requests are needed.
