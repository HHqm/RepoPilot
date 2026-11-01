import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from repopilot.config import Settings
from repopilot.models import IssueRequest
from repopilot.repo_tools import content_sha256
from repopilot.service import RepoPilotService, RunRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = PROJECT_ROOT / "examples" / "buggy_calculator"


class StreamingRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
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
            max_cached_indices=2,
            max_workers=2,
        )
        self.service = RepoPilotService(self.settings)

    def tearDown(self):
        self.service.close()
        self.temporary.cleanup()

    def copy_repo(self, name: str) -> Path:
        target = self.root / name
        shutil.copytree(EXAMPLE, target)
        return target

    @staticmethod
    def request(repo: Path) -> IssueRequest:
        return IssueRequest(
            repo_path=str(repo),
            issue="Fix safe_divide zero divisor behavior.",
            test_command="python -m unittest discover -s tests",
            llm_mode="mock",
        )

    def test_trace_is_published_before_graph_reaches_interrupt(self):
        repo = self.copy_repo("streaming-repo")
        release_mapper = threading.Event()
        mapper_entered = threading.Event()
        original_mapper = self.service.suite.repo_mapper_agent

        def blocked_mapper(state):
            mapper_entered.set()
            if not release_mapper.wait(timeout=5):
                raise TimeoutError("test did not release mapper")
            return original_mapper(state)

        self.service.suite.repo_mapper_agent = blocked_mapper
        accepted = self.service.start(self.request(repo))
        self.assertTrue(mapper_entered.wait(timeout=5))

        runtime = self.service._get_runtime(accepted.run_id)
        intake_events = [
            event
            for event in runtime.events
            if event["type"] == "agent_trace"
            and event["data"].get("agent") == "IntakeAgent"
        ]
        self.assertEqual(len(intake_events), 1)
        self.assertNotEqual(self.service.get(accepted.run_id).status, "awaiting_approval")

        release_mapper.set()
        paused = self.service.wait(accepted.run_id, timeout_seconds=30)
        self.assertEqual(paused.status, "awaiting_approval")

    def test_node_exception_becomes_structured_failed_run(self):
        missing = self.root / "missing-repository"
        accepted = self.service.start(self.request(missing))
        failed = self.service.wait(
            accepted.run_id,
            statuses={"failed"},
            timeout_seconds=30,
        )
        self.assertEqual(failed.state["error"]["agent"], "IntakeAgent")
        self.assertIn("report", failed.state)
        self.assertEqual(
            failed.state["traces"][-1]["agent"], "FailureReporterAgent"
        )

    def test_two_runs_use_concurrent_safe_sqlite_sessions(self):
        first = self.service.start(self.request(self.copy_repo("repo-one")))
        second = self.service.start(self.request(self.copy_repo("repo-two")))
        first_result = self.service.wait(first.run_id, timeout_seconds=30)
        second_result = self.service.wait(second.run_id, timeout_seconds=30)
        self.assertEqual(first_result.status, "awaiting_approval")
        self.assertEqual(second_result.status, "awaiting_approval")
        self.assertLessEqual(
            len(self.service.suite._indices),
            self.settings.max_cached_indices,
        )

    def test_event_sequence_survives_service_restart(self):
        accepted = self.service.start(
            self.request(self.root / "missing-for-event-replay")
        )
        self.service.wait(
            accepted.run_id,
            statuses={"failed"},
            timeout_seconds=30,
        )
        before = self.service._load_durable_events(accepted.run_id)
        self.assertTrue(before)
        self.service.close()
        self.service = RepoPilotService(self.settings)
        replayed = [
            event
            for event in self.service.iter_events(
                accepted.run_id,
                heartbeat_seconds=0.01,
            )
            if event is not None
        ]
        self.assertEqual(
            [event["sequence"] for event in replayed],
            [event["sequence"] for event in before],
        )
        self.assertEqual(
            [event["occurred_at"] for event in replayed],
            [event["occurred_at"] for event in before],
        )

    def test_reconnecting_client_only_receives_unseen_events(self):
        accepted = self.service.start(
            self.request(self.root / "missing-for-resume-cursor")
        )
        self.service.wait(
            accepted.run_id,
            statuses={"failed"},
            timeout_seconds=30,
        )
        everything = [
            event
            for event in self.service.iter_events(
                accepted.run_id,
                heartbeat_seconds=0.01,
            )
            if event is not None
        ]
        self.assertGreaterEqual(len(everything), 3)
        cursor = everything[1]["sequence"]
        tail = [
            event
            for event in self.service.iter_events(
                accepted.run_id,
                after_sequence=cursor,
                heartbeat_seconds=0.01,
            )
            if event is not None
        ]
        self.assertEqual(tail, everything[2:])

    def test_compiled_graph_is_reused_across_sessions(self):
        with self.service._graph_session("maintenance") as first:
            pass
        with self.service._graph_session("maintenance") as second:
            pass
        with self.service._graph_session("project_factory") as studio:
            pass
        self.assertIsNot(first, second)
        self.assertIsNot(first.checkpointer, second.checkpointer)
        self.assertEqual(
            sorted(self.service._graph_templates),
            ["maintenance", "project_factory"],
        )
        self.assertIn("client_intake", studio.nodes)
        self.assertIn("repo_mapper", first.nodes)

    def test_framework_failure_rolls_back_unverified_write(self):
        repo = self.copy_repo("framework-rollback")
        target = repo / "src" / "calculator.py"
        original = target.read_text(encoding="utf-8")
        replacement = original.replace("return a / b", "return float(a) / b")
        run_id = "frameworkrollback"
        apply_result = self.service.suite.applier.apply(
            repo,
            run_id,
            [
                {
                    "path": "src/calculator.py",
                    "original_sha256": content_sha256(original),
                    "new_content": replacement,
                }
            ],
        )
        runtime = RunRuntime(
            run_id=run_id,
            status="running",
            state={
                "run_id": run_id,
                "repo_path": str(repo),
                "workflow_kind": "maintenance",
                "apply_result": apply_result,
                "status": "running",
                "traces": [],
            },
        )
        self.service._handle_framework_failure(
            runtime,
            RuntimeError("forced framework failure"),
        )
        self.assertEqual(target.read_text(encoding="utf-8"), original)
        self.assertTrue(runtime.state["error"]["rollback"]["success"])


if __name__ == "__main__":
    unittest.main()
