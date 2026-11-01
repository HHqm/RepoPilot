import tempfile
import unittest
from pathlib import Path

from repopilot.repo_tools import TestRunner
from repopilot.security import resolve_safe_path, scan_content


class SecurityTests(unittest.TestCase):
    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ok.py").write_text("value = 1\n", encoding="utf-8")
            with self.assertRaises(PermissionError):
                resolve_safe_path(root, "../outside.py")

    def test_rejects_sensitive_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("SECRET=x", encoding="utf-8")
            with self.assertRaises(PermissionError):
                resolve_safe_path(root, ".env")

    def test_rejects_shell_operators(self):
        with self.assertRaises(PermissionError):
            TestRunner.validate_command("python -m unittest && echo unsafe")

    def test_rejects_dangerous_pytest_plugin_options(self):
        with self.assertRaises(PermissionError):
            TestRunner.validate_command("python -m pytest -p malicious_plugin")

    def test_host_test_execution_is_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PermissionError):
                TestRunner().run(
                    directory,
                    "python -m unittest discover -s tests",
                )

    def test_static_scan_detects_dangerous_code(self):
        findings = scan_content("app.py", "result = eval(user_input)\n")
        self.assertTrue(findings)
        self.assertEqual(findings[0]["severity"], "critical")

    def test_documented_dangerous_api_is_not_treated_as_executed_code(self):
        findings = scan_content(
            "docs/SECURITY.md",
            "Never call `os.system()` or disable TLS verification.",
        )
        self.assertEqual(findings, [])

    def test_documentation_still_blocks_embedded_secrets(self):
        findings = scan_content(
            "README.md",
            "Accidentally pasted token sk-abcdefghijklmnopqrstuvwxyz123456",
        )
        self.assertTrue(findings)
        self.assertEqual(findings[0]["message"], "Possible API key")


if __name__ == "__main__":
    unittest.main()
