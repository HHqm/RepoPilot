import os
import tempfile
import unittest
from unittest.mock import patch

from repopilot.config import Settings


class SettingsFromEnvTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        # Never read the developer's real .env during unit tests.
        dotenv = patch("repopilot.config.load_dotenv", lambda **_: None)
        dotenv.start()
        self.addCleanup(dotenv.stop)

    def tearDown(self):
        self.temporary.cleanup()

    def load(self, **env: str) -> Settings:
        with patch.dict(os.environ, env):
            return Settings.from_env(self.temporary.name)

    def test_blank_values_fall_back_to_defaults(self):
        settings = self.load(
            REPOPILOT_MAX_FILES="",
            REPOPILOT_RAG_SCORE_THRESHOLD="  ",
            REPOPILOT_ALLOW_HOST_TEST_EXECUTION="",
        )
        self.assertEqual(settings.max_files, 400)
        self.assertEqual(settings.rag_score_threshold, 0.05)
        self.assertFalse(settings.allow_host_test_execution)

    def test_lower_bounds_are_enforced(self):
        settings = self.load(
            REPOPILOT_MAX_WORKERS="0",
            REPOPILOT_RAG_CHUNK_TOKENS="8",
            REPOPILOT_CONTEXT_MAX_INPUT_TOKENS="10",
        )
        self.assertEqual(settings.max_workers, 1)
        self.assertEqual(settings.rag_chunk_tokens, 64)
        self.assertEqual(settings.context_max_input_tokens, 2_048)

    def test_malformed_number_names_the_variable(self):
        with self.assertRaisesRegex(ValueError, "REPOPILOT_MAX_WORKERS"):
            self.load(REPOPILOT_MAX_WORKERS="four")
        with self.assertRaisesRegex(
            ValueError, "REPOPILOT_RAG_SCORE_THRESHOLD"
        ):
            self.load(REPOPILOT_RAG_SCORE_THRESHOLD="high")

    def test_boolean_flags_accept_common_spellings(self):
        for raw in ("1", "true", "YES", " on "):
            settings = self.load(REPOPILOT_ALLOW_PRIVATE_MODEL_ENDPOINTS=raw)
            self.assertTrue(settings.allow_private_model_endpoints, raw)
        settings = self.load(REPOPILOT_ALLOW_PRIVATE_MODEL_ENDPOINTS="no")
        self.assertFalse(settings.allow_private_model_endpoints)

    def test_unknown_llm_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "REPOPILOT_LLM_MODE"):
            self.load(REPOPILOT_LLM_MODE="gpt")


if __name__ == "__main__":
    unittest.main()
