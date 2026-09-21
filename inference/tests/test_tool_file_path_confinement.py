"""Tests that ``parse_file`` keeps model-supplied file names inside ``file_root_path``.

``react_agent.custom_call_tool`` calls ``FileParser.call`` with a fixed
``file_root_path`` ("./eval_data/file_corpus") and forwards the ``files`` entries
straight from the model's tool call, so an absolute path or a ``..`` segment must
not be able to address a file outside that directory.
"""

import asyncio
import os
import sys
import tempfile
import unittest

# The inference package uses sibling-relative imports (e.g. ``from tool_file import ...``)
# so we add the inference directory to sys.path for these tests.
HERE = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.dirname(HERE)
if INFERENCE_DIR not in sys.path:
    sys.path.insert(0, INFERENCE_DIR)

from tool_file import FileParser  # noqa: E402

SECRET = "PARSE-FILE-MUST-NOT-READ-THIS"


class ParseFilePathConfinementTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self._tmp.name, "eval_data", "file_corpus")
        os.makedirs(self.root)
        with open(os.path.join(self.root, "uploaded.txt"), "w") as f:
            f.write("uploaded corpus document\n")
        self.outside = os.path.join(self._tmp.name, "credentials.txt")
        with open(self.outside, "w") as f:
            f.write(SECRET + "\n")

    def tearDown(self):
        self._tmp.cleanup()

    def _call(self, files):
        return "\n".join(
            str(part)
            for part in asyncio.run(
                FileParser().call({"files": files}, file_root_path=self.root)
            )
        )

    def test_file_inside_root_is_parsed(self):
        self.assertIn("uploaded corpus document", self._call(["uploaded.txt"]))

    def test_absolute_path_cannot_escape_root(self):
        # os.path.join() silently discards file_root_path for an absolute second arg.
        self.assertNotIn(SECRET, self._call([self.outside]))

    def test_parent_traversal_cannot_escape_root(self):
        self.assertNotIn(SECRET, self._call([os.path.join("..", "..", "credentials.txt")]))


if __name__ == "__main__":
    unittest.main()
