import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from repopilot.agents import AgentSuite
from repopilot.config import Settings
from repopilot.llm import (
    LLMConfigurationError,
    OpenAICompatibleLLM,
    build_llm,
)


class _ProviderError(Exception):
    def __init__(self, status_code: int | None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _completion(content: str, usage: dict | None = None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=(
            SimpleNamespace(model_dump=lambda: dict(usage)) if usage else None
        ),
    )


class _ScriptedCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _routed_llm(outcomes, *, supports_json_object=True):
    settings = Settings(
        deepseek_api_key=None,
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-v4-pro",
        llm_mode="mock",
        max_files=10,
        max_file_bytes=10_000,
        rag_top_k=3,
        test_timeout_seconds=10,
        runtime_dir=Path(".repopilot-test"),
    )
    llm = OpenAICompatibleLLM(
        {
            "api_key": "sk-test",
            "provider_id": "custom",
            "provider_name": "Custom",
            "credential_id": "cred-1",
            "model": "test-model",
            "agent_id": "qa",
            "base_url": "https://example.invalid/v1",
            "supports_json_object": supports_json_object,
        },
        settings,
    )
    completions = _ScriptedCompletions(outcomes)
    llm.client.close()
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        close=lambda: None,
    )
    return llm, completions


@patch("repopilot.llm.time.sleep", lambda _seconds: None)
class OpenAICompatibleAdapterTests(unittest.TestCase):
    def complete(self, llm):
        return llm.complete_json(
            system_prompt="Return JSON.",
            payload={"issue": "x"},
            schema_name="qa_report",
        )

    def test_rate_limit_is_retried_and_usage_is_recorded(self):
        llm, completions = _routed_llm(
            [
                _ProviderError(429),
                _completion(
                    '{"ok": true}',
                    {"prompt_tokens": 11, "completion_tokens": 3, "noise": 9},
                ),
            ]
        )
        self.assertEqual(self.complete(llm), {"ok": True})
        self.assertEqual(len(completions.requests), 2)
        self.assertEqual(
            llm.last_usage,
            {"prompt_tokens": 11, "completion_tokens": 3},
        )

    def test_client_errors_fail_fast_without_retry(self):
        llm, completions = _routed_llm([_ProviderError(401)])
        with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
            self.complete(llm)
        self.assertEqual(len(completions.requests), 1)

    def test_retries_are_bounded(self):
        llm, completions = _routed_llm([_ProviderError(503)] * 5)
        with self.assertRaisesRegex(RuntimeError, "failed after retries"):
            self.complete(llm)
        self.assertEqual(len(completions.requests), 3)

    def test_fenced_json_is_accepted_without_json_mode(self):
        llm, completions = _routed_llm(
            [_completion('```json\n{"ok": 1}\n```')],
            supports_json_object=False,
        )
        self.assertEqual(self.complete(llm), {"ok": 1})
        self.assertNotIn("response_format", completions.requests[0])
        self.assertIn(
            "Return exactly one valid JSON object",
            completions.requests[0]["messages"][0]["content"],
        )


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
