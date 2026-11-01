import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from repopilot.config import Settings
from repopilot.models import ApprovalDecision, IssueRequest
from repopilot.service import RepoPilotService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = PROJECT_ROOT / "examples" / "buggy_calculator"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WorkflowTests(unittest.TestCase):
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
        self.services = []

    def tearDown(self):
        for service in self.services:
            service.close()
        self.temporary.cleanup()

    def make_service(self):
        service = RepoPilotService(self.settings)
        self.services.append(service)
        return service

    def test_pauses_before_write_then_completes_after_approval(self):
        target = self.repo / "src" / "calculator.py"
        before = file_hash(target)
        service = self.make_service()
        accepted = service.start(
            IssueRequest(
                repo_path=str(self.repo),
                issue=(
                    "Fix safe_divide so a zero divisor raises ValueError "
                    "instead of ZeroDivisionError."
                ),
                constraints=["Keep the public function signature unchanged."],
                test_command="python -m unittest discover -s tests",
                llm_mode="mock",
            )
        )
        self.assertIn(accepted.status, {"queued", "running"})
        run = service.wait(accepted.run_id, timeout_seconds=30)
        self.assertEqual(run.status, "awaiting_approval")
        self.assertIsNotNone(run.interrupt)
        self.assertEqual(file_hash(target), before, "unapproved run modified a file")
        self.assertTrue(run.state["review"]["approved"])
        model_traces = {
            trace["agent"]: trace
            for trace in run.state["traces"]
            if trace["agent"] in {"PlannerAgent", "CoderAgent", "ReviewerAgent"}
        }
        self.assertEqual(
            set(model_traces),
            {"PlannerAgent", "CoderAgent", "ReviewerAgent"},
        )
        for trace in model_traces.values():
            context = trace["metadata"]["context"]
            self.assertEqual(context["scope"], "dynamic_payload_estimate")
            self.assertLessEqual(
                context["estimated_tokens_after"],
                context["budget_tokens"],
            )
        with self.assertRaises(RuntimeError):
            service.decide(
                run.run_id,
                ApprovalDecision(
                    approved=True,
                    comment="stale approval",
                    artifact_digest="0" * 64,
                ),
            )

        resumed = service.decide(
            run.run_id,
            ApprovalDecision(
                approved=True,
                comment="Test approval",
                artifact_digest=run.interrupt["artifact_digest"],
            ),
        )
        self.assertEqual(resumed.status, "resuming")
        completed = service.wait(
            run.run_id,
            statuses={"completed", "failed", "failed_rolled_back"},
            timeout_seconds=30,
        )
        self.assertEqual(completed.status, "completed")
        self.assertNotEqual(file_hash(target), before)
        self.assertTrue(completed.state["test_result"]["success"])
        self.assertTrue(completed.state["verification"]["success"])
        self.assertIn("ValueError", target.read_text(encoding="utf-8"))

    def test_rejection_keeps_repository_unchanged(self):
        target = self.repo / "src" / "calculator.py"
        before = file_hash(target)
        service = self.make_service()
        accepted = service.start(
            IssueRequest(
                repo_path=str(self.repo),
                issue="Fix safe_divide zero divisor behavior.",
                test_command="python -m unittest discover -s tests",
                llm_mode="mock",
            )
        )
        run = service.wait(accepted.run_id, timeout_seconds=30)
        service.decide(
            run.run_id,
            ApprovalDecision(
                approved=False,
                comment="Needs more analysis",
                artifact_digest=run.interrupt["artifact_digest"],
            ),
        )
        rejected = service.wait(
            run.run_id,
            statuses={"cancelled", "failed"},
            timeout_seconds=30,
        )
        self.assertEqual(rejected.status, "cancelled")
        self.assertEqual(file_hash(target), before)
        self.assertNotIn("apply_result", rejected.state)

    def test_failed_verification_rolls_back_every_change(self):
        target = self.repo / "src" / "calculator.py"
        before = file_hash(target)
        (self.repo / "tests" / "test_forced_failure.py").write_text(
            "import unittest\n\n"
            "class ForcedFailure(unittest.TestCase):\n"
            "    def test_failure(self):\n"
            "        self.fail('forced verifier failure')\n",
            encoding="utf-8",
        )
        service = self.make_service()
        accepted = service.start(
            IssueRequest(
                repo_path=str(self.repo),
                issue="Fix safe_divide zero divisor behavior.",
                test_command="python -m unittest discover -s tests",
                llm_mode="mock",
            )
        )
        run = service.wait(accepted.run_id, timeout_seconds=30)
        approvals = 0
        # Every repair attempt is a new patch and must be approved again.
        while run.status == "awaiting_approval":
            approvals += 1
            self.assertEqual(file_hash(target), before, "approval gate leaked")
            service.decide(
                run.run_id,
                ApprovalDecision(
                    approved=True,
                    comment="Exercise rollback path",
                    artifact_digest=run.interrupt["artifact_digest"],
                ),
            )
            run = service.wait(run.run_id, timeout_seconds=60)
        completed = run
        self.assertEqual(completed.status, "failed_rolled_back")
        budget = self.settings.max_repair_attempts
        self.assertEqual(approvals, budget + 1)
        self.assertEqual(completed.state["repair_attempt"], budget)
        self.assertEqual(len(completed.state["attempt_history"]), budget + 1)
        self.assertIn(
            "forced verifier failure",
            completed.state["attempt_history"][0]["test_failure"],
        )
        self.assertFalse(completed.state["test_result"]["success"])
        self.assertEqual(
            completed.state["verification"]["verdict"], "failed_and_rolled_back"
        )
        self.assertEqual(file_hash(target), before)
        self.assertTrue(completed.state["reflection"]["retry_recommended"])
        # Repair attempts re-retrieve with a query built from the failure.
        purposes = [
            item["purpose"] for item in completed.state["retrieval"]["queries"]
        ]
        self.assertIn("failure", purposes)
        self.assertIn("Self-correction history", completed.state["report"])
        self.assertGreater(completed.state["run_metrics"]["llm_calls"], 0)


if __name__ == "__main__":
    unittest.main()
