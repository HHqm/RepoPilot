import tempfile
import unittest
from pathlib import Path

from repopilot.agents import AgentSuite
from repopilot.config import Settings
from repopilot.llm import LLMConfigurationError, build_llm


class LlmConfigurationTests(unittest.TestCase):
    def test_deepseek_mode_requires_environment_key(self):
        settings = Settings(
            deepseek_api_key=None,
            deepseek_base_url="https://api.deepseek.com",
            deepseek_model="deepseek-v4-pro",
            llm_mode="deepseek",
            max_files=10,
            max_file_bytes=10_000,
            rag_top_k=3,
            test_timeout_seconds=10,
            runtime_dir=Path(".repopilot-test"),
        )
        with self.assertRaises(LLMConfigurationError):
            build_llm("deepseek", settings)

    def test_invalid_structured_output_gets_one_bounded_repair(self):
        class RepairingLLM:
            def __init__(self):
                self.calls = 0
                self.last_usage = {}

            def complete_json(self, **_):
                self.calls += 1
                if self.calls == 1:
                    return {"approved": "not-a-boolean"}
                return {
                    "approved": True,
                    "summary": "Repaired response",
                    "findings": [],
                }

        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                deepseek_api_key=None,
                deepseek_base_url="https://api.deepseek.com",
                deepseek_model="deepseek-v4-pro",
                llm_mode="mock",
                max_files=10,
                max_file_bytes=10_000,
                rag_top_k=3,
                test_timeout_seconds=10,
                runtime_dir=Path(directory) / "runtime",
                rag_embedding_provider="deterministic",
                rag_reranker_provider="deterministic",
            )
            suite = AgentSuite(settings)
            fake = RepairingLLM()
            suite._llms["mock"] = fake
            try:
                result, context = suite._complete_json(
                    {"llm_mode": "mock"},
                    system_prompt="Return a review decision.",
                    schema_name="review_decision",
                    payload={
                        "issue": "review",
                        "constraints": [],
                        "plan": {},
                        "evidence": [],
                        "diffs": "",
                    },
                )
            finally:
                suite.close()
        self.assertTrue(result["approved"])
        self.assertEqual(fake.calls, 2)
        self.assertEqual(context["repair_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
