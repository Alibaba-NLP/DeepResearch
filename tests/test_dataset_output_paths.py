"""Exercise the real runner while replacing only the external model agent."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

RUNNER = Path(__file__).resolve().parents[1] / "inference/run_multi_react.py"


class DatasetOutputPathTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(prefix="deepresearch-paths-")
        self.directory = Path(self.fixture.name).resolve()
        self.assertIn(Path(tempfile.gettempdir()).resolve(), self.directory.parents)
        self.output = self.directory / "results"
        self.agent = Mock()
        self.agent._run.side_effect = lambda task, model: {
            "question": task["item"]["question"],
            "prediction": "offline fixture result",
        }
        self.agent_module = types.ModuleType("react_agent")
        self.agent_module.MultiTurnReactAgent = Mock(return_value=self.agent)

    def tearDown(self):
        self.assertIn(Path(tempfile.gettempdir()).resolve(), self.directory.parents)
        self.fixture.cleanup()

    def run_cli(self, dataset, cwd=None):
        args = [
            str(RUNNER),
            "--model",
            "test-model",
            "--dataset",
            str(dataset),
            "--output",
            str(self.output),
            "--roll_out_count",
            "1",
            "--max_workers",
            "1",
        ]
        previous = Path.cwd()
        try:
            os.chdir(cwd or self.directory)
            with patch.object(sys, "argv", args), patch.dict(
                sys.modules, {"react_agent": self.agent_module}
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                runpy.run_path(str(RUNNER), run_name="__main__")
        finally:
            os.chdir(previous)

    def expected_directory(self, dataset):
        return self.output / "test-model_sglang" / Path(dataset).name

    def test_absolute_json_dataset(self):
        dataset = self.directory / "questions.json"
        dataset.write_text("[]", encoding="utf-8")
        self.run_cli(dataset)
        self.assertTrue(self.expected_directory(dataset).is_dir())
        self.assertEqual(dataset.read_text(encoding="utf-8"), "[]")
        self.agent_module.MultiTurnReactAgent.assert_not_called()

    def test_absolute_jsonl_dataset(self):
        dataset = self.directory / "questions.jsonl"
        dataset.write_text("", encoding="utf-8")
        self.run_cli(dataset)
        self.assertTrue(self.expected_directory(dataset).is_dir())

    def test_nested_relative_dataset_keeps_existing_output_layout(self):
        dataset = self.directory / "inputs" / "questions.json"
        dataset.parent.mkdir()
        dataset.write_text("[]", encoding="utf-8")
        self.run_cli(Path("inputs") / dataset.name)
        self.assertTrue(
            (self.output / "test-model_sglang" / "inputs" / dataset.name).is_dir()
        )

    def test_parent_relative_dataset_keeps_existing_output_layout(self):
        dataset = self.directory / "questions.json"
        dataset.write_text("[]", encoding="utf-8")
        worker = self.directory / "worker"
        worker.mkdir()
        self.run_cli(Path("..") / dataset.name, cwd=worker)
        self.assertTrue((self.output / dataset.name).is_dir())

    def test_bare_relative_dataset_keeps_existing_output_name(self):
        dataset = self.directory / "questions.json"
        dataset.write_text("[]", encoding="utf-8")
        self.run_cli(dataset.name)
        self.assertTrue(self.expected_directory(dataset).is_dir())

    def test_absolute_dataset_writes_results_and_resumes(self):
        dataset = self.directory / "questions.json"
        original = json.dumps([{"question": "offline question"}])
        dataset.write_text(original, encoding="utf-8")
        self.run_cli(dataset)
        self.run_cli(dataset)
        output_file = self.expected_directory(dataset) / "iter1.jsonl"
        records = [
            json.loads(line)
            for line in output_file.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            records,
            [{"question": "offline question", "prediction": "offline fixture result"}],
        )
        self.agent._run.assert_called_once()
        self.assertEqual(dataset.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
