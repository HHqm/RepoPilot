import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from repopilot.config import Settings
from repopilot.llm import MockLLM
from repopilot.models import ApprovalDecision, ProjectRequest
from repopilot.service import RepoPilotService
from repopilot.studio_agents import _generated_project_contract_findings


class SoftwareStudioWorkflowTests(unittest.TestCase):
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
        )
        self.service = RepoPilotService(self.settings)

    def tearDown(self):
        self.service.close()
        self.temporary.cleanup()

    @staticmethod
    def request(preferred_stack: str = "Go") -> ProjectRequest:
        return ProjectRequest(
            project_name="社区公共服务预约平台",
            client_brief=(
                "为社区居民提供公共服务预约、预约结果查询和运营健康检查，"
                "首期只交付可运行的后端 MVP 与部署使用说明。"
            ),
            target_users="社区居民、社区服务人员和系统运维人员",
            budget_cny=120_000,
            delivery_deadline="6 周内完成 MVP",
            policy_constraints=[
                "不得在代码和日志中保存居民身份证等敏感信息",
                "界面和文档使用简体中文",
            ],
            business_constraints=["首期不采购昂贵商业中间件"],
            preferred_stack=preferred_stack,
            deployment_target="docker",
            deployment_conditions="部署在客户 Linux 服务器，2 核 4GB，必须提供回滚步骤",
            llm_mode="mock",
        )

    def test_two_human_gates_then_delivers_polyglot_project(self):
        accepted = self.service.start_project(self.request("Go"))
        requirements = self.service.wait(
            accepted.run_id,
            statuses={"awaiting_requirements_approval", "failed"},
            timeout_seconds=30,
        )
        self.assertEqual(requirements.status, "awaiting_requirements_approval")
        self.assertEqual(requirements.interrupt["checkpoint"], "requirements")
        self.assertTrue(requirements.state["requirements_workshop"]["risks"])
        self.assertTrue(
            requirements.state["requirements_workshop"]["decomposition"]
        )
        requirements_root = Path(requirements.state["repo_path"])
        self.assertTrue(requirements_root.is_dir())
        self.assertFalse(any(requirements_root.iterdir()))
        self.assertTrue(
            requirements.state["product_requirements"][
                "functional_requirements"
            ]
        )

        self.service.decide(
            requirements.run_id,
            ApprovalDecision(
                approved=False,
                comment="补充运维人员健康检查，并明确首期不做在线支付。",
                artifact_digest=requirements.interrupt["artifact_digest"],
            ),
        )
        revised = self.service.wait(
            requirements.run_id,
            statuses={"awaiting_requirements_approval", "failed"},
            timeout_seconds=30,
        )
        self.assertEqual(revised.status, "awaiting_requirements_approval")
        self.assertEqual(revised.state["requirements_revision"], 1)
        self.assertIn(
            "Revision incorporates client feedback",
            " ".join(revised.state["product_requirements"]["assumptions"]),
        )

        self.service.decide(
            revised.run_id,
            ApprovalDecision(
                approved=True,
                comment="PRD 已确认，可以进入研发。",
                artifact_digest=revised.interrupt["artifact_digest"],
            ),
        )
        release = self.service.wait(
            revised.run_id,
            statuses={"awaiting_release_approval", "failed", "release_rejected"},
            timeout_seconds=30,
        )
        self.assertEqual(release.status, "awaiting_release_approval")
        self.assertEqual(release.interrupt["checkpoint"], "release")
        self.assertEqual(release.state["architecture"]["primary_language"], "Go 1.22")
        self.assertTrue(release.state["project_plan"]["phases"])
        self.assertTrue(release.state["deployment_guide"]["release_conditions"])
        self.assertTrue(release.state["qa_report"]["passed"])
        self.assertTrue(release.state["qa_report"]["test_cases"])
        output = Path(release.state["project_output_path"])
        self.assertTrue(output.is_dir())
        self.assertFalse(any(output.iterdir()), "unapproved project files were written")

        self.service.decide(
            release.run_id,
            ApprovalDecision(
                approved=True,
                comment="已检查 PRD、架构、代码清单和部署条件，同意生成交付包。",
                artifact_digest=release.interrupt["artifact_digest"],
            ),
        )
        completed = self.service.wait(
            release.run_id,
            statuses={"completed", "delivery_failed", "failed"},
            timeout_seconds=30,
        )
        self.assertEqual(completed.status, "completed")
        self.assertTrue(completed.state["quality_gate"]["success"])
        self.assertEqual(
            completed.state["quality_gate"]["runtime_execution"],
            "not_executed_untrusted_code",
        )
        output = Path(completed.state["project_output_path"])
        self.assertTrue((output / "README.md").is_file())
        self.assertTrue((output / "DEPLOYMENT.md").is_file())
        self.assertTrue((output / "MANIFEST.json").is_file())
        self.assertTrue((output / "docs" / "TEST_REPORT.md").is_file())
        self.assertTrue((output / "go.mod").is_file())
        self.assertIn(
            "module example.com/repopilot-project",
            (output / "go.mod").read_text(encoding="utf-8"),
        )
        self.assertTrue((output / "cmd" / "server" / "main.go").is_file())
        self.assertTrue(
            Path(completed.state["delivery_archive_path"]).is_file()
        )

    def test_generated_identifiers_and_literals_are_safe_across_languages(self):
        project_name = '社区 "Alpha"\n../../x'

        generated = {
            language: {
                item["path"]: item["content"]
                for item in MockLLM._generated_files(language, project_name)
            }
            for language in (
                "Java 21",
                "Go 1.22",
                "C++20",
                "TypeScript / Node.js 20",
                "Python 3.11",
            )
        }

        self.assertIn("<artifactId>alpha-x</artifactId>", generated["Java 21"]["pom.xml"])
        self.assertIn("module example.com/alpha-x", generated["Go 1.22"]["go.mod"])
        self.assertIn("project(alpha_x LANGUAGES CXX)", generated["C++20"]["CMakeLists.txt"])
        self.assertEqual(
            "alpha-x",
            __import__("json").loads(
                generated["TypeScript / Node.js 20"]["package.json"]
            )["name"],
        )
        compile(
            generated["Python 3.11"]["src/product/service.py"],
            "generated_service.py",
            "exec",
        )

    def test_qa_revision_changes_only_targeted_files(self):
        state = {
            "repo_path": str(self.root),
            "project_name": "Targeted Revision",
            "product_requirements": {"functional_requirements": []},
            "requirements_workshop": {"decisions": []},
            "architecture": {"primary_language": "Python 3.11"},
            "policy_constraints": [],
            "business_constraints": [],
            "development_revision": 0,
            "deployment_guide": {"rollback_steps": ["old"]},
            "generated_project": {
                "summary": "initial",
                "files": [
                    {
                        "path": "app.py",
                        "purpose": "application",
                        "content": "VALUE = 1\n",
                    },
                    {
                        "path": "README.md",
                        "purpose": "documentation",
                        "content": "# Keep me\n",
                    },
                ],
            },
            "qa_report": {
                "passed": False,
                "summary": "app value is wrong",
                "test_cases": [
                    {
                        "case_id": "TC-001",
                        "requirement_id": "FR-001",
                        "name": "value",
                        "method": "static",
                        "status": "failed",
                        "evidence": "VALUE must be 2",
                    }
                ],
                "defects": ["app.py must set VALUE to 2"],
                "acceptance_coverage": ["FR-001 -> TC-001"],
                "recommended_commands": [],
                "runtime_validation_required": True,
                "validation_level": "static_preflight",
                "runtime_verified": False,
                "environment_limitations": [],
            },
        }
        response = {
            "summary": "Changed only app.py.",
            "changed_files": [
                {
                    "path": "app.py",
                    "purpose": "application",
                    "content": "VALUE = 2\n",
                }
            ],
            "delete_paths": [],
            "addressed_defects": ["app.py now sets VALUE to 2"],
        }
        with patch.object(
            self.service.studio_suite,
            "_complete",
            return_value=(response, {"schema": "project_revision"}),
        ):
            result = self.service.studio_suite.developer_revision_agent(state)

        files = {
            item["path"]: item["content"]
            for item in result["generated_project"]["files"]
        }
        self.assertEqual(files["app.py"], "VALUE = 2\n")
        self.assertEqual(files["README.md"], "# Keep me\n")
        self.assertEqual(result["development_revision"], 1)
        self.assertEqual(
            result["traces"][0]["metadata"]["changed_paths"],
            ["app.py"],
        )

    def test_python_framework_contract_defects_are_deterministic(self):
        files = [
            {
                "path": "requirements.txt",
                "purpose": "dependencies",
                "content": "pydantic==2.5.2\npytest-asyncio==0.23.2\n",
            },
            {
                "path": "tests/test_api.py",
                "purpose": "tests",
                "content": (
                    "import pytest\n"
                    "@pytest.fixture\n"
                    "async def client():\n"
                    "    yield object()\n"
                ),
            },
            {
                "path": "app/models.py",
                "purpose": "models",
                "content": (
                    "from datetime import datetime, timezone\n"
                    "from sqlalchemy import Column, DateTime\n"
                    "created_at = Column(DateTime(), "
                    "default=datetime.now(timezone.utc))\n"
                ),
            },
            {
                "path": "app/schemas.py",
                "purpose": "schemas",
                "content": (
                    "class Config:\n"
                    "    orm_mode = True\n"
                ),
            },
        ]
        findings = _generated_project_contract_findings(
            files,
            {"primary_language": "Python 3.11"},
        )

        self.assertEqual(len(findings), 3)
        self.assertTrue(any("async pytest fixture" in item for item in findings))
        self.assertTrue(any("module import" in item for item in findings))
        self.assertTrue(any("Pydantic v2" in item for item in findings))

    def test_python_framework_contract_defect_overrides_llm_qa_pass(self):
        state = {
            "generated_project": {
                "files": [
                    {
                        "path": "requirements.txt",
                        "purpose": "dependencies",
                        "content": "pytest-asyncio==0.23.2\n",
                    },
                    {
                        "path": "tests/test_api.py",
                        "purpose": "tests",
                        "content": (
                            "import pytest\n"
                            "@pytest.fixture\n"
                            "async def client():\n"
                            "    yield object()\n"
                        ),
                    },
                ]
            },
            "product_requirements": {"functional_requirements": []},
            "requirements_workshop": {"decisions": []},
            "architecture": {"primary_language": "Python 3.11"},
            "deployment_guide": {},
            "development_revision": 0,
        }
        llm_pass = {
            "passed": True,
            "summary": "The model approved the project.",
            "test_cases": [
                {
                    "case_id": "TC-001",
                    "requirement_id": "FR-001",
                    "name": "model review",
                    "method": "static",
                    "status": "passed",
                    "evidence": "Model inspection",
                }
            ],
            "defects": [],
            "acceptance_coverage": [],
            "recommended_commands": [],
            "runtime_validation_required": True,
            "environment_limitations": [],
        }
        with patch.object(
            self.service.studio_suite,
            "_complete",
            return_value=(llm_pass, {"schema": "qa_report"}),
        ):
            result = self.service.studio_suite.qa_agent(state)

        self.assertFalse(result["qa_report"]["passed"])
        self.assertEqual(result["status"], "qa_failed")
        self.assertIn(
            "async pytest fixture",
            result["qa_report"]["defects"][0],
        )

    def test_release_rejection_writes_nothing(self):
        accepted = self.service.start_project(self.request("C++"))
        requirements = self.service.wait(
            accepted.run_id,
            statuses={"awaiting_requirements_approval", "failed"},
            timeout_seconds=30,
        )
        self.service.decide(
            requirements.run_id,
            ApprovalDecision(
                approved=True,
                comment="确认需求",
                artifact_digest=requirements.interrupt["artifact_digest"],
            ),
        )
        release = self.service.wait(
            requirements.run_id,
            statuses={"awaiting_release_approval", "failed"},
            timeout_seconds=30,
        )
        self.assertEqual(
            release.state["architecture"]["primary_language"],
            "C++20",
        )
        self.service.decide(
            release.run_id,
            ApprovalDecision(
                approved=False,
                comment="部署资源条件还没有确认，暂不生成交付包。",
                artifact_digest=release.interrupt["artifact_digest"],
            ),
        )
        rejected = self.service.wait(
            release.run_id,
            statuses={"release_rejected", "failed"},
            timeout_seconds=30,
        )
        self.assertEqual(rejected.status, "release_rejected")
        rejected_root = Path(rejected.state["project_output_path"])
        self.assertTrue(rejected_root.is_dir())
        self.assertFalse(any(rejected_root.iterdir()))
        self.assertNotIn("apply_result", rejected.state)


if __name__ == "__main__":
    unittest.main()
