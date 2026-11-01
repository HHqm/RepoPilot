import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from repopilot import cli


class CliDemoTests(unittest.TestCase):
    def test_offline_demo_completes_under_default_safety_settings(self):
        """The demo must work out of the box: workspace roots and host test
        execution stay locked down for everything except its own temp copy."""

        with tempfile.TemporaryDirectory() as directory:
            demo_parent = Path(directory) / "demo"
            demo_parent.mkdir()
            environment = {
                "REPOPILOT_LLM_MODE": "mock",
                "REPOPILOT_RAG_EMBEDDING_PROVIDER": "deterministic",
                "REPOPILOT_RAG_RERANKER_PROVIDER": "deterministic",
                "REPOPILOT_ALLOW_HOST_TEST_EXECUTION": "0",
                "REPOPILOT_WORKSPACE_ROOTS": str(Path(directory) / "elsewhere"),
                "REPOPILOT_QDRANT_PATH": "",
                "REPOPILOT_QDRANT_URL": "",
            }
            output = io.StringIO()
            real_mkdtemp = tempfile.mkdtemp

            def mkdtemp(*args, **kwargs):
                # Only redirect the demo workspace; the test runner's own
                # temporary HOME must still be created normally.
                if kwargs.get("prefix") == "repopilot-demo-":
                    return str(demo_parent)
                return real_mkdtemp(*args, **kwargs)

            with (
                patch.dict(os.environ, environment),
                patch("repopilot.config.load_dotenv", lambda **_: None),
                patch.object(tempfile, "mkdtemp", mkdtemp),
                contextlib.redirect_stdout(output),
            ):
                exit_code = cli.command_demo(None)
        self.assertEqual(exit_code, 0, output.getvalue()[-2000:])
        self.assertIn('"status": "completed"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
