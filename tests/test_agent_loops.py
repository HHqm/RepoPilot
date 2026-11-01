import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from repopilot.config import Settings
from repopilot.llm import MockLLM
from repopilot.models import ApprovalDecision, IssueRequest
from repopilot.rag.multi_query import (
    extract_code_symbols,
    failure_query,
    fuse_rankings,
    plan_queries,
)
from repopilot.run_metrics import summarize_run_metrics
from repopilot.service import RepoPilotService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = PROJECT_ROOT / "examples" / "buggy_calculator"


class ScriptedLLM(MockLLM):
    """MockLLM with per-schema overrides and a record of every payload."""

    def __init__(self, overrides):
        self.overrides = {key: list(value) for key, value in overrides.items()}
        self.payloads: dict[str, list[dict]] = {}

    def complete_json(self, *, system_prompt, payload, schema_name):
        self.payloads.setdefault(schema_name, []).append(payload)
        scripted = self.overrides.get(schema_name)
        if scripted:
            return scripted.pop(0)
        return super().complete_json(
            system_prompt=system_prompt,
            payload=payload,
            schema_name=schema_name,
        )


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        shutil.copytree(EXAMPLE, self.repo)
        self.settings = Settings(
            deepseek_api_key=None,
            deepseek_base_url="https://api.deepseek.com",
            deepseek_model="deepseek-v4-pro",
            llm_mode="mock",
            max_files=100,
            max_file_bytes=200_000,
            rag_top_k=6,
            test_timeout_seconds=30,
            runtime_dir=self.root / "runtime",
            allow_host_test_execution=True,
        )
        self.service = None

    def tearDown(self):
        if self.service is not None:
            self.service.close()
        self.temporary.cleanup()

    def start(self, llm: ScriptedLLM, settings: Settings | None = None):
        self.service = RepoPilotService(settings or self.settings)
        self.service.suite._llms["mock"] = llm
        accepted = self.service.start(
            IssueRequest(
                repo_path=str(self.repo),
                issue="Fix safe_divide so a zero divisor raises ValueError.",
                test_command="python -m unittest discover -s tests",
                llm_mode="mock",
            )
        )
        return self.service.wait(accepted.run_id, timeout_seconds=60)

    def test_rejected_review_is_revised_by_the_coder_then_approved(self):
        llm = ScriptedLLM(
            {
                "review_decision": [
                    {
                        "approved": False,
                        "summary": "Error message is not specific enough.",
                        "findings": [
                            {
                                "severity": "medium",
                                "message": "Mention the divisor in the error.",
                                "path": "src/calculator.py",
                            }
                        ],
                    }
                ]
            }
        )
        run = self.start(llm)
        self.assertEqual(run.status, "awaiting_approval")
        self.assertEqual(run.state["review_revision"], 1)
        self.assertEqual(len(llm.payloads["code_edits"]), 2)
        self.assertNotIn("review_feedback", llm.payloads["code_edits"][0])
        feedback = llm.payloads["code_edits"][1]["review_feedback"]
        self.assertEqual(
            feedback["findings"][0]["message"], "Mention the divisor in the error."
        )
        self.assertIn("rejected_diffs", feedback)
        coder_traces = [
            trace for trace in run.state["traces"] if trace["agent"] == "CoderAgent"
        ]
        self.assertEqual(
            [trace["metadata"]["addresses_review"] for trace in coder_traces],
            [False, True],
        )

    def test_review_budget_exhausted_ends_the_run(self):
        rejection = {
            "approved": False,
            "summary": "Still wrong.",
            "findings": [{"severity": "high", "message": "Wrong fix."}],
        }
        llm = ScriptedLLM({"review_decision": [rejection, rejection]})
        run = self.start(llm)
        self.assertEqual(run.status, "review_rejected")
        self.assertEqual(len(llm.payloads["code_edits"]), 2)

    def test_reflection_can_decline_a_retry(self):
        (self.repo / "tests" / "test_env.py").write_text(
            "import unittest\n\n"
            "class Env(unittest.TestCase):\n"
            "    def test_env(self):\n"
            "        self.fail('missing service dependency')\n",
            encoding="utf-8",
        )
        llm = ScriptedLLM(
            {
                "reflection": [
                    {
                        "root_cause": "The test environment lacks a service.",
                        "next_action": "Provision the dependency in CI.",
                        "retry_recommended": False,
                    }
                ]
            }
        )
        run = self.start(llm)
        self.service.decide(
            run.run_id,
            ApprovalDecision(
                approved=True,
                artifact_digest=run.interrupt["artifact_digest"],
            ),
        )
        done = self.service.wait(run.run_id, timeout_seconds=60)
        self.assertEqual(done.status, "failed_rolled_back")
        self.assertEqual(done.state.get("repair_attempt", 0), 0)
        self.assertEqual(len(llm.payloads["change_plan"]), 1)

    def test_repair_attempt_replans_with_failure_history(self):
        (self.repo / "tests" / "test_flaky.py").write_text(
            "import unittest\n\n"
            "class Strict(unittest.TestCase):\n"
            "    def test_strict(self):\n"
            "        self.fail('first patch is not enough')\n",
            encoding="utf-8",
        )
        settings = replace(self.settings, max_repair_attempts=1)
        llm = ScriptedLLM({})
        run = self.start(llm, settings)
        self.service.decide(
            run.run_id,
            ApprovalDecision(
                approved=True, artifact_digest=run.interrupt["artifact_digest"]
            ),
        )
        retry = self.service.wait(run.run_id, timeout_seconds=60)
        self.assertEqual(retry.status, "awaiting_approval")
        self.assertEqual(retry.state["repair_attempt"], 1)
        replan_payload = llm.payloads["change_plan"][1]
        self.assertEqual(len(replan_payload["previous_attempts"]), 1)
        self.assertIn(
            "first patch is not enough",
            replan_payload["previous_attempts"][0]["test_failure"],
        )
        self.assertNotIn("previous_attempts", llm.payloads["change_plan"][0])


class QueryPlanningTests(unittest.TestCase):
    def test_symbols_are_extracted_without_duplicating_paths(self):
        symbols = extract_code_symbols(
            "Fix safe_divide in src/calculator.py; see os.path.join, "
            "parseConfig and HTTPClient but not HTTP"
        )
        self.assertEqual(
            symbols,
            [
                "src/calculator.py",
                "os.path.join",
                "safe_divide",
                "parseConfig",
                "HTTPClient",
            ],
        )

    def test_failure_query_keeps_project_frames_and_error(self):
        output = (
            "FAIL: test_zero (tests.test_calc.T.test_zero)\n"
            'File "/usr/lib/python3.11/unittest/case.py", line 1, in run\n'
            'File "/repo/src/calc.py", line 5, in safe_divide\n'
            "ZeroDivisionError: division by zero\n"
        )
        query = failure_query(output, "guard missing")
        self.assertIn("calc.py", query)
        self.assertIn("safe_divide", query)
        self.assertIn("ZeroDivisionError: division by zero", query)
        self.assertNotIn("case.py", query)
        self.assertTrue(query.endswith("guard missing"))

    def test_plan_adds_failure_query_only_after_a_failed_attempt(self):
        first = plan_queries("Fix safe_divide", [])
        self.assertEqual([item.purpose for item in first], ["issue", "symbols"])
        retry = plan_queries(
            "Fix safe_divide",
            [],
            last_failure={"test_failure": "ValueError: boom", "root_cause": "x"},
        )
        self.assertEqual(
            [item.purpose for item in retry], ["issue", "symbols", "failure"]
        )

    def test_rrf_fusion_promotes_chunks_found_by_several_queries(self):
        from repopilot.models import Evidence

        def evidence(chunk: str) -> Evidence:
            return Evidence(
                evidence_id="E?",
                chunk_id=chunk,
                document_id="d",
                path=f"{chunk}.py",
                text=chunk,
                source_type="code",
                language="python",
                fusion_score=0.0,
                rerank_score=0.0,
                score=0.0,
            )

        fused = fuse_rankings(
            [
                [evidence("a"), evidence("shared")],
                [evidence("shared"), evidence("b")],
            ],
            top_k=3,
        )
        self.assertEqual([item.chunk_id for item in fused], ["shared", "a", "b"])
        self.assertEqual([item.evidence_id for item in fused], ["E1", "E2", "E3"])


class RunMetricsTests(unittest.TestCase):
    def test_counts_nested_repair_calls_and_sums_usage(self):
        call = {"schema": "code_edits", "estimated_tokens_before": 10,
                "estimated_tokens_after": 8, "api_usage": {"total_tokens": 30}}
        repair = {"schema": "schema_repair", "estimated_tokens_before": 4,
                  "estimated_tokens_after": 4, "api_usage": {"total_tokens": 5}}
        metrics = summarize_run_metrics(
            [
                {"agent": "CoderAgent", "latency_ms": 12.5,
                 "metadata": {"context": {**call, "repair_context": repair}}},
                {"agent": "TestAgent", "latency_ms": 100, "metadata": {}},
            ]
        )
        self.assertEqual(metrics["llm_calls"], 2)
        self.assertEqual(metrics["token_usage"], {"total_tokens": 35})
        self.assertEqual(metrics["estimated_input_tokens"], 12)
        self.assertEqual(list(metrics["agent_latency_ms"]), ["TestAgent", "CoderAgent"])


if __name__ == "__main__":
    unittest.main()
