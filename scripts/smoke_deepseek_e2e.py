from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TERMINAL_FAILURES = {
    "failed",
    "failed_rolled_back",
    "delivery_failed",
    "requirements_rejected",
    "release_rejected",
}
REQUIRED_STATE_ARTIFACTS = {
    "product_requirements",
    "requirements_workshop",
    "project_plan",
    "architecture",
    "generated_project",
    "deployment_guide",
    "qa_report",
    "release_review",
}
REQUIRED_ZIP_FILES = {
    "DEPLOYMENT.md",
    "MANIFEST.json",
    "docs/PRD.md",
    "docs/REQUIREMENTS_WORKSHOP.md",
    "docs/PROJECT_PLAN.md",
    "docs/ARCHITECTURE.md",
    "docs/TEST_REPORT.md",
}


class SmokeFailure(RuntimeError):
    pass


class RepoPilotClient:
    def __init__(self, base_url: str, timeout_seconds: int):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.api_token = os.getenv("REPOPILOT_API_TOKEN", "").strip()

    def request(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        data: bytes | None = None,
        content_type: str | None = None,
        expect_json: bool = True,
    ):
        headers = {"Accept": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        if content_type:
            headers["Content-Type"] = content_type
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:4_000]
            raise SmokeFailure(
                f"{method} {path} returned HTTP {exc.code}: {detail}"
            ) from exc
        except (TimeoutError, URLError) as exc:
            raise SmokeFailure(f"{method} {path} failed: {exc}") from exc
        if not expect_json:
            return body
        return json.loads(body.decode("utf-8"))

    def upload_markdown(
        self,
        session_id: str,
        *,
        filename: str,
        content: str,
    ) -> dict:
        boundary = f"----RepoPilotSmoke{uuid.uuid4().hex}"
        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                (
                    'Content-Disposition: form-data; name="file"; '
                    f'filename="{filename}"\r\n'
                ).encode(),
                b"Content-Type: text/markdown; charset=utf-8\r\n\r\n",
                content.encode("utf-8"),
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        return self.request(
            "POST",
            f"/project-sessions/{session_id}/files",
            data=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )


def emit(stage: str, **data) -> None:
    print(
        json.dumps(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                **data,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def wait_for_run(
    client: RepoPilotClient,
    run_id: str,
    *,
    deadline: float,
    approved_gates: set[str],
) -> dict:
    last_status = ""
    while time.monotonic() < deadline:
        run = client.request("GET", f"/runs/{run_id}")
        status = str(run["status"])
        if status != last_status:
            emit(
                "run_status",
                run_id=run_id,
                status=status,
                trace_count=len(run.get("state", {}).get("traces", [])),
            )
            last_status = status
        if status in TERMINAL_FAILURES:
            state = run.get("state", {})
            raise SmokeFailure(
                json.dumps(
                    {
                        "status": status,
                        "error": state.get("error"),
                        "qa_report": state.get("qa_report"),
                        "release_review": state.get("release_review"),
                    },
                    ensure_ascii=False,
                )
            )
        if status == "completed":
            return run
        if status in {
            "awaiting_requirements_approval",
            "awaiting_release_approval",
        }:
            gate = (
                "requirements"
                if status == "awaiting_requirements_approval"
                else "release"
            )
            if gate not in approved_gates:
                interrupt = run.get("interrupt") or {}
                digest = interrupt.get("artifact_digest")
                if not digest:
                    raise SmokeFailure(f"{gate} approval has no artifact digest")
                client.request(
                    "POST",
                    f"/runs/{run_id}/decision",
                    {
                        "approved": True,
                        "comment": f"DeepSeek E2E smoke approved {gate} artifacts.",
                        "actor_id": "deepseek-e2e-smoke",
                        "artifact_digest": digest,
                    },
                )
                approved_gates.add(gate)
                emit("human_gate_approved", gate=gate, digest_prefix=digest[:12])
        time.sleep(2)
    raise SmokeFailure("Timed out waiting for the multi-agent workflow")


def parse_sse(raw: bytes) -> list[dict]:
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
    events: list[dict] = []
    for block in text.split("\n\n"):
        event_match = re.search(r"^event:\s*(.+)$", block, re.MULTILINE)
        data_match = re.search(r"^data:\s*(.+)$", block, re.MULTILINE)
        if not event_match or not data_match:
            continue
        events.append(
            {
                "type": event_match.group(1).strip(),
                "data": json.loads(data_match.group(1)),
            }
        )
    return events


def aggregate_usage(traces: list[dict]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for trace in traces:
        usage = (
            trace.get("metadata", {})
            .get("context", {})
            .get("api_usage", {})
        )
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += value
    return dict(totals)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a real DeepSeek RepoPilot workflow through the HTTP API."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=int, default=1_200)
    args = parser.parse_args()

    started = time.monotonic()
    client = RepoPilotClient(args.base_url, timeout_seconds=240)
    report: dict = {
        "base_url": args.base_url,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }
    try:
        health = client.request("GET", "/health")
        if health.get("llm_mode") != "deepseek":
            raise SmokeFailure(f"Server LLM mode is {health.get('llm_mode')!r}")
        report["health"] = {
            key: health.get(key)
            for key in (
                "status",
                "llm_mode",
                "model",
                "rag",
                "vector_store",
                "execution",
                "events",
            )
        }
        emit("health_ok", **report["health"])

        suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
        session = client.request(
            "POST",
            "/project-sessions",
            {
                "project_name": f"DeepSeek E2E Incident API {suffix}",
                "target_users": "30 名企业内网运维人员和 3 名值班主管",
                "budget_cny": 80_000,
                "delivery_deadline": "6 周内交付可部署 MVP",
                "policy_constraints": [
                    "业务数据不得离开企业内网",
                    "日志不得记录令牌、密码和个人敏感信息",
                ],
                "business_constraints": [
                    "首期不接入短信、支付和外部 SaaS",
                    "必须提供健康检查、审计字段和回滚指南",
                ],
                "preferred_stack": "Python 3.11 + FastAPI",
                "deployment_target": "docker",
                "deployment_conditions": (
                    "Linux x86_64，Docker Compose；生产数据库由客户提供，"
                    "本次只交付可部署包，不执行生产上线。"
                ),
                "llm_mode": "deepseek",
            },
        )
        session_id = session["session_id"]
        report["session_id"] = session_id
        emit("session_created", session_id=session_id)

        marker = f"RAG-E2E-{uuid.uuid4().hex[:12]}"
        uploaded = client.upload_markdown(
            session_id,
            filename="e2e-requirements.md",
            content=(
                "# 事故工单 API 需求\n\n"
                f"追踪标识：{marker}\n\n"
                "必须支持创建、查询、更新状态和追加处理记录；状态仅允许 "
                "open、investigating、resolved。所有时间使用 UTC ISO 8601。"
                "提供 `/health`；首期不实现登录、通知、附件和生产部署。"
            ),
        )
        assets = uploaded.get("assets", [])
        if len(assets) != 1 or assets[0].get("chunk_count", 0) < 1:
            raise SmokeFailure(f"Uploaded document was not chunked: {assets!r}")
        report["checks"]["upload_and_chunking"] = True
        report["asset"] = {
            key: assets[0].get(key)
            for key in ("original_name", "parser", "chunk_count", "redaction_count")
        }
        emit("document_indexed", **report["asset"])

        chat_content = (
            "请基于上传文件设计事故工单 REST API。验收标准：能创建工单，按 ID "
            "查询，把状态从 open 更新为 investigating/resolved，并追加处理记录；"
            "非法状态返回 422，不存在返回 404，提供健康检查。用户、预算、期限、"
            "部署与合规约束均已在项目边界中确认。MVP 不做认证、通知、附件和真实"
            "生产上线；测试需覆盖正常流程、非法状态和不存在资源。"
        )
        for turn in range(1, 5):
            discussed = client.request(
                "POST",
                f"/project-sessions/{session_id}/messages",
                {"content": chat_content},
            )
            understanding = discussed.get("understanding", {})
            emit(
                "product_manager_turn",
                turn=turn,
                ready=understanding.get("ready_for_workshop"),
                open_question_count=len(understanding.get("open_questions", [])),
            )
            if understanding.get("ready_for_workshop"):
                break
            open_questions = understanding.get("open_questions", [])
            chat_content = (
                "对尚未确认项统一答复：数据保存在客户提供的 PostgreSQL；30 名内网"
                "用户；单实例 MVP；SLA 不作承诺；由客户运维执行部署；验收以 API "
                "测试、静态安全检查和详细部署手册为准。首期范围不再增加。待确认问题："
                + "；".join(str(item) for item in open_questions)
            )
        else:
            raise SmokeFailure("Product manager did not reach workshop readiness")
        report["checks"]["real_product_manager_chat"] = True
        report["product_chat_turns"] = turn

        launched = client.request(
            "POST",
            f"/project-sessions/{session_id}/launch",
        )
        run_id = launched["run_id"]
        report["run_id"] = run_id
        emit("workflow_launched", run_id=run_id)

        completed = wait_for_run(
            client,
            run_id,
            deadline=time.monotonic() + args.timeout_seconds,
            approved_gates=set(),
        )
        state = completed.get("state", {})
        missing_artifacts = sorted(REQUIRED_STATE_ARTIFACTS - state.keys())
        if missing_artifacts:
            raise SmokeFailure(f"Missing state artifacts: {missing_artifacts}")
        report["checks"]["all_state_artifacts"] = True

        traces = state.get("traces", [])
        memory_counts = [
            trace.get("metadata", {})
            .get("context", {})
            .get("project_memory_items", 0)
            for trace in traces
        ]
        if not any(value > 0 for value in memory_counts):
            raise SmokeFailure("No downstream agent consumed project RAG memory")
        report["checks"]["project_rag_consumed"] = True

        sse_raw = client.request(
            "GET",
            f"/runs/{run_id}/events",
            expect_json=False,
        )
        events = parse_sse(sse_raw)
        sequences = [
            int(event["data"]["sequence"])
            for event in events
            if "sequence" in event["data"]
        ]
        if not sequences or sequences != sorted(set(sequences)):
            raise SmokeFailure("SSE event sequence is empty, duplicated, or unordered")
        event_types = Counter(event["type"] for event in events)
        if event_types["agent_trace"] < 1 or event_types["run_status"] < 1:
            raise SmokeFailure(f"Expected live events are missing: {event_types}")
        report["checks"]["durable_sse_sequence"] = True

        archive_bytes = client.request(
            "GET",
            f"/runs/{run_id}/download",
            expect_json=False,
        )
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            zip_names = set(archive.namelist())
            bad_entry = next(
                (
                    name
                    for name in zip_names
                    if name.startswith(("/", "\\"))
                    or ".." in name.replace("\\", "/").split("/")
                ),
                None,
            )
            if bad_entry:
                raise SmokeFailure(f"Unsafe ZIP entry: {bad_entry}")
            missing_zip = sorted(REQUIRED_ZIP_FILES - zip_names)
            if missing_zip:
                raise SmokeFailure(f"Missing ZIP files: {missing_zip}")
            manifest = json.loads(archive.read("MANIFEST.json").decode("utf-8"))
        if not manifest:
            raise SmokeFailure("ZIP manifest is empty")
        report["checks"]["safe_complete_zip"] = True

        qa = state["qa_report"]
        report.update(
            {
                "status": completed["status"],
                "duration_seconds": round(time.monotonic() - started, 2),
                "approved_human_gates": 2,
                "trace_count": len(traces),
                "agent_names": sorted(
                    {
                        str(trace.get("agent", ""))
                        for trace in traces
                        if trace.get("agent")
                    }
                ),
                "api_usage": aggregate_usage(traces),
                "project_memory_max_items": max(memory_counts, default=0),
                "event_counts": dict(event_types),
                "event_sequence_range": [min(sequences), max(sequences)],
                "qa": {
                    "passed": qa.get("passed"),
                    "validation_level": qa.get("validation_level"),
                    "runtime_verified": qa.get("runtime_verified"),
                    "runtime_validation_required": qa.get(
                        "runtime_validation_required"
                    ),
                    "test_case_count": len(qa.get("test_cases", [])),
                    "defect_count": len(qa.get("defects", [])),
                },
                "release_approved": state["release_review"].get("approved"),
                "zip_size_bytes": len(archive_bytes),
                "zip_file_count": len(zip_names),
                "zip_files": sorted(zip_names),
            }
        )
        report["checks"]["two_human_gates"] = True
        report["checks"]["deepseek_usage_recorded"] = bool(report["api_usage"])
        if not report["checks"]["deepseek_usage_recorded"]:
            raise SmokeFailure("DeepSeek usage metadata was not recorded")
        emit("completed", run_id=run_id, duration_seconds=report["duration_seconds"])
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["duration_seconds"] = round(time.monotonic() - started, 2)
        report["error"] = str(exc)[:8_000]
        print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
