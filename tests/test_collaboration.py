import tempfile
import unittest
import zipfile
from pathlib import Path

from repopilot.config import Settings
from repopilot.models import (
    ApprovalDecision,
    ProductUnderstanding,
    ProjectChatRequest,
    ProjectSessionCreate,
)
from repopilot.service import RepoPilotService


class ProjectCollaborationTests(unittest.TestCase):
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
            rag_top_k=8,
            test_timeout_seconds=30,
            runtime_dir=self.root / "runtime",
            rag_document_parser="text-only",
        )
        self.service = RepoPilotService(self.settings)

    def tearDown(self):
        self.service.close()
        self.temporary.cleanup()

    def test_chat_upload_workshop_qa_and_zip_delivery(self):
        session = self.service.collaboration.create(
            ProjectSessionCreate(
                project_name="客户协作平台",
                target_users="业务人员和运维人员",
                budget_cny=180_000,
                delivery_deadline="8 周",
                preferred_stack="Go",
                deployment_target="docker",
                llm_mode="mock",
            )
        )
        self.assertEqual(session.status, "discovering")
        self.assertEqual(session.messages[0].role, "product_manager")

        uploaded = self.service.collaboration.add_document(
            session.session_id,
            filename="requirements.md",
            media_type="text/markdown",
            content=(
                b"# Booking workflow\nUsers create and query bookings.\n"
                b"api_key=sk-1234567890abcdefghijklmnop\n"
            ),
        )
        self.assertEqual(len(uploaded.assets), 1)
        self.assertEqual(uploaded.assets[0].chunk_count, 1)
        self.assertEqual(uploaded.assets[0].redaction_count, 1)

        discussed = self.service.collaboration.chat(
            session.session_id,
            ProjectChatRequest(
                content=(
                    "需要实现预约创建、查询和健康检查；首期不做支付，"
                    "部署到客户内网 Docker，由客户运维团队负责上线。"
                )
            ),
        )
        self.assertTrue(discussed.understanding.ready_for_workshop)

        run, launched = self.service.launch_collaboration(session.session_id)
        self.assertEqual(launched.run_id, run.run_id)
        requirements = self.service.wait(
            run.run_id,
            statuses={"awaiting_requirements_approval", "failed"},
            timeout_seconds=30,
        )
        self.assertEqual(requirements.status, "awaiting_requirements_approval")
        self.assertTrue(requirements.state["requirements_workshop"]["risks"])
        self.assertTrue(
            requirements.state["requirements_workshop"]["decomposition"]
        )
        memory_counts = [
            trace.get("metadata", {})
            .get("context", {})
            .get("project_memory_items", 0)
            for trace in requirements.state["traces"]
        ]
        self.assertTrue(any(count > 0 for count in memory_counts))

        self.service.decide(
            run.run_id,
            ApprovalDecision(
                approved=True,
                comment="PRD 和会议纪要确认",
                artifact_digest=requirements.interrupt["artifact_digest"],
            ),
        )
        release = self.service.wait(
            run.run_id,
            statuses={"awaiting_release_approval", "failed"},
            timeout_seconds=30,
        )
        self.assertEqual(release.status, "awaiting_release_approval")
        self.assertTrue(release.state["qa_report"]["passed"])
        self.assertTrue(release.state["qa_report"]["test_cases"])
        self.assertTrue(
            release.state["qa_report"]["runtime_validation_required"]
        )

        self.service.decide(
            run.run_id,
            ApprovalDecision(
                approved=True,
                comment="批准生成 ZIP 交付包",
                artifact_digest=release.interrupt["artifact_digest"],
            ),
        )
        completed = self.service.wait(
            run.run_id,
            statuses={"completed", "delivery_failed", "failed"},
            timeout_seconds=30,
        )
        self.assertEqual(completed.status, "completed")
        archive = Path(completed.state["delivery_archive_path"])
        self.assertTrue(archive.is_file())
        with zipfile.ZipFile(archive) as bundle:
            names = set(bundle.namelist())
        self.assertIn("DEPLOYMENT.md", names)
        self.assertIn("MANIFEST.json", names)
        self.assertIn("docs/PRD.md", names)
        self.assertIn("docs/REQUIREMENTS_WORKSHOP.md", names)
        self.assertIn("docs/PROJECT_PLAN.md", names)
        self.assertIn("docs/ARCHITECTURE.md", names)
        self.assertIn("docs/TEST_REPORT.md", names)

    def test_launch_reservation_blocks_duplicate_submission(self):
        session = self.service.collaboration.create(
            ProjectSessionCreate(
                project_name="Idempotent Launch",
                target_users="operators",
                llm_mode="mock",
            )
        )
        self.service.collaboration.chat(
            session.session_id,
            ProjectChatRequest(content="需要一个资产登记和查询系统，首期不做支付。"),
        )
        self.service.collaboration.chat(
            session.session_id,
            ProjectChatRequest(content="用户是运维人员，部署在本地 Docker。"),
        )
        _, token = self.service.collaboration.reserve_launch(session.session_id)
        with self.assertRaises(RuntimeError):
            self.service.collaboration.reserve_launch(session.session_id)
        self.service.collaboration.abort_launch(session.session_id, token)
        restored = self.service.collaboration.get(session.session_id)
        self.assertEqual(restored.status, "ready_for_workshop")

    def test_discovery_gate_repairs_contradictions_and_bounds_questions(self):
        session = {
            "target_users": "internal operators",
            "delivery_deadline": "six weeks",
            "deployment_target": "docker",
            "messages": [
                {
                    "role": "client",
                    "content": (
                        "Build an incident API for internal operators with create, "
                        "read and state update workflows. Keep data in the intranet, "
                        "deliver in six weeks, include tests and deployment guidance."
                    ),
                },
                {
                    "role": "client",
                    "content": (
                        "The MVP excludes login, notifications and attachments. "
                        "Docker deployment is owned by client operations."
                    ),
                },
            ],
        }
        inconsistent = ProductUnderstanding(
            summary="A" * 100,
            confirmed_goals=["Deliver the incident workflow"],
            constraints=["Intranet-only data"],
            open_questions=[],
            assumptions=[],
            ready_for_workshop=False,
        )
        repaired, note = self.service.collaboration._normalize_discovery_gate(
            session,
            inconsistent,
        )
        self.assertTrue(repaired.ready_for_workshop)
        self.assertTrue(note)

        session["messages"].append(
            {"role": "client", "content": "All material boundaries are confirmed."}
        )
        endless = inconsistent.model_copy(
            update={
                "open_questions": ["Should endpoint naming be decided now?"],
                "ready_for_workshop": False,
            }
        )
        bounded, note = self.service.collaboration._normalize_discovery_gate(
            session,
            endless,
        )
        self.assertTrue(bounded.ready_for_workshop)
        self.assertEqual(bounded.open_questions, [])
        self.assertIn("Should endpoint naming", bounded.assumptions[-1])
        self.assertTrue(note)


if __name__ == "__main__":
    unittest.main()
