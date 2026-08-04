import os
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

# API tests must never attach to the developer/server Qdrant Local directory.
# Keep the module-level FastAPI service alive for this test process, but build it
# against an isolated deterministic runtime before restoring the caller's env.
_API_RUNTIME = tempfile.TemporaryDirectory(prefix="repopilot-api-tests-")
_TEST_ENV = {
    "REPOPILOT_RUNTIME_DIR": _API_RUNTIME.name,
    "REPOPILOT_QDRANT_PATH": str(Path(_API_RUNTIME.name) / "qdrant"),
    "REPOPILOT_LLM_MODE": "mock",
    "REPOPILOT_RAG_EMBEDDING_PROVIDER": "deterministic",
    "REPOPILOT_RAG_RERANKER_PROVIDER": "deterministic",
    "REPOPILOT_API_TOKEN": "",
}
_PREVIOUS_ENV = {key: os.environ.get(key) for key in _TEST_ENV}
os.environ.update(_TEST_ENV)
try:
    from repopilot.api import app
finally:
    for _key, _value in _PREVIOUS_ENV.items():
        if _value is None:
            os.environ.pop(_key, None)
        else:
            os.environ[_key] = _value


class ApiTests(unittest.TestCase):
    def test_public_project_contract_rejects_internal_storage_fields(self):
        response = TestClient(app).post(
            "/projects",
            json={
                "project_name": "Boundary Test",
                "client_brief": "Create a small internal scheduling service.",
                "target_users": "Operations staff",
                "project_knowledge_path": "D:/attacker-controlled",
                "uploaded_documents": [{"stored_name": "forged.pdf"}],
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_health_does_not_expose_api_key(self):
        response = TestClient(app).get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["execution"], "background")
        self.assertEqual(body["events"], "live")
        self.assertEqual(
            body["workflow"],
            "client_chat_files_workshop_to_zip",
        )
        self.assertEqual(body["rag"], "qdrant_dense_sparse_rrf_rerank")
        self.assertEqual(body["vector_store"], "qdrant-local")
        self.assertEqual(body["model"], "deepseek-v4-pro")
        self.assertEqual(
            body["context_compression"],
            "deterministic_schema_aware",
        )
        self.assertEqual(body["context_max_input_tokens"], 64_000)
        self.assertIn("embedding_provider", body)
        self.assertIn("sparse_model", body)
        self.assertNotIn("api_key", body)

    def test_project_session_accepts_secret_free_agent_model_routing(self):
        client = TestClient(app)
        response = client.post(
            "/project-sessions",
            json={
                "project_name": "Routed project",
                "target_users": "Internal users",
                "llm_mode": "routed",
                "model_routing": {
                    "preset": "single",
                    "default_route": {
                        "credential_id": "builtin:mock",
                        "model": "mock-deterministic",
                    },
                    "agent_routes": {
                        "DevelopmentTeamAgent": {
                            "credential_id": "builtin:mock",
                            "model": "mock-deterministic",
                        }
                    },
                },
            },
        )
        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["llm_mode"], "routed")
        self.assertEqual(
            body["model_routing"]["agent_routes"][
                "DevelopmentTeamAgent"
            ]["model"],
            "mock-deterministic",
        )
        self.assertNotIn("api_key", response.text)

    def test_collaboration_workbench_and_assets_are_served(self):
        client = TestClient(app)
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.headers["cache-control"], "no-store, max-age=0")
        self.assertEqual(page.headers["x-repopilot-frontend"], "0.6.0")
        self.assertIn("客户协作软件公司", page.text)
        self.assertIn("直接告诉产品经理", page.text)
        self.assertIn("UI 0.6.0 · MULTI MODEL", page.text)
        self.assertIn("AUTO-CREATE ON FIRST MESSAGE", page.text)
        self.assertIn("YOUR PRODUCT MANAGER", page.text)
        self.assertIn("项目资料库", page.text)
        self.assertIn("交付总监需求澄清与避坑会议", page.text)
        self.assertIn("软件研发与测试流水线", page.text)
        self.assertIn("下载 ZIP 交付包", page.text)
        self.assertNotIn("验收命令", page.text)
        self.assertIn('id="approvalCard"', page.text)
        self.assertIn('id="fileInput"', page.text)
        self.assertIn('id="chatMessages"', page.text)
        self.assertIn('id="modelCenter"', page.text)
        self.assertIn('id="agentRoutingList"', page.text)
        self.assertLess(
            page.text.index('id="chatMessages"'),
            page.text.index('id="promptChips"'),
        )
        self.assertLess(
            page.text.index('id="promptChips"'),
            page.text.index('id="chatForm"'),
        )
        self.assertNotIn('id="chatInput" disabled', page.text)
        self.assertNotIn('id="sendMessageButton" class="primary-button" type="submit" disabled', page.text)
        self.assertIn(
            'id="approveButton" class="primary-button" type="button" disabled',
            page.text,
        )

        css = client.get("/assets/styles.css")
        self.assertEqual(css.status_code, 200)
        self.assertEqual(css.headers["cache-control"], "no-store, max-age=0")
        self.assertIn('grid-template-areas:\n    "chat knowledge"', css.text)
        self.assertIn(".chat-kickoff", css.text)
        self.assertIn(".meeting-room", css.text)
        self.assertIn(
            '.approval-section[data-state="awaiting"]',
            css.text,
        )
        self.assertIn(
            "@media (prefers-reduced-motion: reduce)",
            css.text,
        )

        script = client.get("/assets/app.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn('fetch("/project-sessions"', script.text)
        self.assertIn("/files`", script.text)
        self.assertIn("/messages`", script.text)
        self.assertIn("/launch`", script.text)
        self.assertIn("/download`", script.text)
        self.assertIn("async function ensureProjectSession", script.text)
        self.assertIn("await ensureProjectSession(content)", script.text)
        self.assertNotIn("if (!projectSession || !content) return;", script.text)
        self.assertNotIn('elements.llmMode.value = "mock";', script.text)
        self.assertIn('elements.toast.textContent = "";', script.text)
        self.assertIn("function renderApproval", script.text)
        self.assertIn("function renderMeeting", script.text)

    def test_ui_config_and_openapi_expose_collaboration_contract(self):
        client = TestClient(app)
        response = client.get("/ui-config")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertGreaterEqual(len(body["agents"]), 15)
        self.assertTrue(all("id" in agent for agent in body["agents"]))
        self.assertEqual(
            body["workflow"],
            "collaborative_project_factory",
        )
        self.assertEqual(body["frontend_version"], "0.6.0")
        self.assertEqual(len(body["model_agents"]), 8)
        self.assertTrue(
            any(
                provider["preset_id"] == "zzshu"
                and provider["base_url"] == "https://zzshu.cc/v1"
                for provider in body["model_providers"]
            )
        )
        self.assertIn("Go", body["supported_languages"])
        self.assertIn("locations", body)
        self.assertIn("phases", body)
        health = client.get("/health").json()
        self.assertEqual(body["default_llm_mode"], health["llm_mode"])
        self.assertEqual(body["model"], "deepseek-v4-pro")
        self.assertEqual(
            body["context"]["compression"],
            "deterministic_schema_aware",
        )
        self.assertNotIn("api_key", response.text.lower())

        paths = client.get("/openapi.json").json()["paths"]
        self.assertIn("/project-sessions", paths)
        self.assertIn(
            "/project-sessions/{session_id}/messages",
            paths,
        )
        self.assertIn(
            "/project-sessions/{session_id}/files",
            paths,
        )
        self.assertIn(
            "/project-sessions/{session_id}/launch",
            paths,
        )
        self.assertIn("/runs/{run_id}/download", paths)

    def test_demo_workspace_is_an_isolated_copy(self):
        response = TestClient(app).post("/demo-workspace")
        self.assertEqual(response.status_code, 200)
        path = response.json()["repo_path"]
        self.assertIn("demos", path)
        target = Path(path)
        try:
            self.assertTrue((target / "src" / "calculator.py").is_file())
        finally:
            shutil.rmtree(target)

    def test_failed_run_is_streamed_as_structured_sse(self):
        client = TestClient(app)
        response = client.post(
            "/runs",
            json={
                "repo_path": str(
                    Path(__file__).resolve().parents[1]
                    / "definitely-missing-repopilot-repository"
                ),
                "issue": "Fail this missing repository through the safe boundary.",
                "test_command": "python -m unittest discover -s tests",
                "llm_mode": "mock",
            },
        )
        self.assertEqual(response.status_code, 202)
        run_id = response.json()["run_id"]
        with client.stream("GET", f"/runs/{run_id}/events") as stream:
            self.assertEqual(stream.status_code, 200)
            body = "\n".join(stream.iter_lines())
        self.assertIn("event: agent_trace", body)
        self.assertIn("FailureReporterAgent", body)
        self.assertIn('"status": "failed"', body)


if __name__ == "__main__":
    unittest.main()
