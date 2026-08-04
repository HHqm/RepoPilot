from __future__ import annotations

import ast
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import tomllib
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from smoke_deepseek_e2e import (
    REQUIRED_STATE_ARTIFACTS,
    REQUIRED_ZIP_FILES,
    RepoPilotClient,
    SmokeFailure,
    aggregate_usage,
    emit,
    parse_sse,
    wait_for_run,
)


ALLOWED_IMPORT_ROOTS = {
    "calculator",
    "decimal",
    "math",
    "os",
    "sys",
    "typing",
    "unittest",
}
FORBIDDEN_CALL_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "open",
}
FORBIDDEN_ATTRIBUTE_ROOTS = {
    "http",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "urllib",
}
ALLOWED_OS_CALLS = {
    "os.path.abspath",
    "os.path.dirname",
    "os.path.join",
}
ALLOWED_SYS_CALLS = {
    "sys.path.insert",
}


def _attribute_path(node: ast.Attribute) -> str:
    parts = [node.attr]
    current: ast.expr = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _validate_tiny_python_project(root: Path, generated_paths: set[str]) -> None:
    if len(generated_paths) > 8:
        raise SmokeFailure(
            f"Tiny project expanded beyond 8 generated files: {len(generated_paths)}"
        )
    if not any(path.startswith("tests/") and path.endswith(".py") for path in generated_paths):
        raise SmokeFailure("Tiny project has no Python unittest file")

    requirements = [
        line.strip()
        for path in generated_paths
        if Path(path).name.lower().startswith("requirements")
        for line in (root / path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if requirements:
        raise SmokeFailure(f"Tiny project declared third-party requirements: {requirements}")

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        parsed = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        dependencies = parsed.get("project", {}).get("dependencies", [])
        if dependencies:
            raise SmokeFailure(
                f"Tiny project declared pyproject dependencies: {dependencies}"
            )

    for relative in sorted(generated_paths):
        if not relative.endswith(".py"):
            continue
        source = (root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [(node.module or "").split(".", 1)[0]]
            else:
                imports = []
            unexpected = sorted(
                name for name in imports if name and name not in ALLOWED_IMPORT_ROOTS
            )
            if unexpected:
                raise SmokeFailure(
                    f"{relative} imports outside the tiny allowlist: {unexpected}"
                )
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id in FORBIDDEN_CALL_NAMES
                ):
                    raise SmokeFailure(
                        f"{relative} calls forbidden function {node.func.id}"
                    )
                if isinstance(node.func, ast.Attribute):
                    current = node.func.value
                    while isinstance(current, ast.Attribute):
                        current = current.value
                    attribute_path = _attribute_path(node.func)
                    if (
                        isinstance(current, ast.Name)
                        and current.id == "os"
                        and attribute_path not in ALLOWED_OS_CALLS
                    ):
                        raise SmokeFailure(
                            f"{relative} calls non-path os API {attribute_path}"
                        )
                    if (
                        isinstance(current, ast.Name)
                        and current.id == "sys"
                        and attribute_path not in ALLOWED_SYS_CALLS
                    ):
                        raise SmokeFailure(
                            f"{relative} calls unexpected sys API {attribute_path}"
                        )
                    if (
                        isinstance(current, ast.Name)
                        and current.id in FORBIDDEN_ATTRIBUTE_ROOTS
                    ):
                        raise SmokeFailure(
                            f"{relative} calls forbidden module {current.id}"
                        )


def _run_tiny_unittests(root: Path) -> dict:
    runner = (
        "import sys,unittest;"
        "sys.path.insert(0,'.');"
        "suite=unittest.defaultTestLoader.discover('tests');"
        "result=unittest.TextTestRunner(verbosity=2).run(suite);"
        "raise SystemExit(0 if result.wasSuccessful() else 1)"
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        in {
            "COMSPEC",
            "PATH",
            "PATHEXT",
            "SYSTEMDRIVE",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "WINDIR",
        }
    }
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-I", "-c", runner],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
        shell=False,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    if completed.returncode != 0:
        raise SmokeFailure(
            "Generated tiny-project tests failed:\n" + output[-8_000:]
        )
    return {
        "return_code": completed.returncode,
        "output": output[-4_000:],
    }


def main() -> int:
    started = time.monotonic()
    client = RepoPilotClient("http://127.0.0.1:8000", timeout_seconds=240)
    report: dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }
    try:
        health = client.request("GET", "/health")
        if health.get("llm_mode") != "deepseek":
            raise SmokeFailure(f"Server LLM mode is {health.get('llm_mode')!r}")
        emit("health_ok", model=health.get("model"))

        suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
        session = client.request(
            "POST",
            "/project-sessions",
            {
                "project_name": f"Tiny DeepSeek Calculator {suffix}",
                "target_users": "学习 Python 单元测试的初学者",
                "budget_cny": 1_000,
                "delivery_deadline": "今天完成",
                "policy_constraints": [
                    "代码必须离线运行，不访问网络、文件系统、环境变量或子进程",
                    "不得使用任何第三方依赖",
                ],
                "business_constraints": [
                    "只做一个纯函数和 unittest，最多 8 个文本文件",
                    "不做 Web、数据库、登录、Docker、云部署和图形界面",
                ],
                "preferred_stack": "Python 3.11 标准库 + unittest",
                "deployment_target": "local",
                "deployment_conditions": (
                    "Windows 或 Linux 已安装 Python 3.11；"
                    "运行 python -m unittest discover -s tests -v"
                ),
                "llm_mode": "deepseek",
            },
        )
        session_id = session["session_id"]
        report["session_id"] = session_id
        emit("session_created", session_id=session_id)

        specification = """
# 极小折扣计算器

只实现 `calculator.py` 中的纯函数：

`calculate_total(unit_price_cents: int, quantity: int, discount_percent: int) -> int`

- 三个参数都必须是 int（bool 不算 int）；
- unit_price_cents >= 0，quantity >= 1，discount_percent 在 0..100；
- 非法输入抛 ValueError；
- 返回折扣后的整数分，公式为 `(单价 * 数量 * (100 - 折扣) + 50) // 100`；
- 使用 Python 标准库 unittest 覆盖正常值、四舍五入、零价、100% 折扣和非法输入；
- 不读写文件、不联网、无第三方依赖。
""".strip()
        uploaded = client.upload_markdown(
            session_id,
            filename="tiny-calculator-spec.md",
            content=specification,
        )
        assets = uploaded.get("assets", [])
        if len(assets) != 1 or assets[0].get("chunk_count", 0) < 1:
            raise SmokeFailure("Tiny specification was not indexed")
        report["checks"]["upload_and_rag_index"] = True
        emit("document_indexed", chunk_count=assets[0]["chunk_count"])

        message = (
            "请严格按已上传规格交付极小折扣计算器。目标用户、预算、期限、"
            "技术栈、运行环境、验收公式、异常规则和排除范围均已确认；"
            "没有其他功能，没有待确认问题，可以进入需求评审会。"
        )
        for turn in range(1, 4):
            response = client.request(
                "POST",
                f"/project-sessions/{session_id}/messages",
                {"content": message},
            )
            understanding = response["understanding"]
            emit(
                "product_manager_turn",
                turn=turn,
                ready=understanding.get("ready_for_workshop"),
                open_question_count=len(understanding.get("open_questions", [])),
            )
            if understanding.get("ready_for_workshop"):
                break
            message = (
                "对所有待确认问题统一回答：以已上传规格为唯一 MVP 范围；"
                "只在本地 Python 3.11 运行；无第三方依赖、无外部集成、"
                "无数据持久化；验收命令为 python -m unittest discover -s tests -v。"
            )
        else:
            raise SmokeFailure("Product manager did not finish tiny discovery")
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
            deadline=time.monotonic() + 1_200,
            approved_gates=set(),
        )
        state = completed["state"]
        missing_state = sorted(REQUIRED_STATE_ARTIFACTS - state.keys())
        if missing_state:
            raise SmokeFailure(f"Missing workflow artifacts: {missing_state}")
        if not state["qa_report"].get("passed"):
            raise SmokeFailure(f"QA failed: {state['qa_report'].get('defects')}")
        if not state["release_review"].get("approved"):
            raise SmokeFailure("Release reviewer did not approve tiny project")
        report["checks"]["workflow_artifacts_and_gates"] = True

        traces = state.get("traces", [])
        memory_counts = [
            trace.get("metadata", {})
            .get("context", {})
            .get("project_memory_items", 0)
            for trace in traces
        ]
        if not any(count > 0 for count in memory_counts):
            raise SmokeFailure("No downstream agent consumed project memory")
        usage = aggregate_usage(traces)
        if not usage:
            raise SmokeFailure("DeepSeek usage metadata is missing")
        report["checks"]["rag_consumed_and_usage_recorded"] = True

        event_bytes = client.request(
            "GET",
            f"/runs/{run_id}/events",
            expect_json=False,
        )
        events = parse_sse(event_bytes)
        event_types = Counter(event["type"] for event in events)
        sequences = [
            int(event["data"]["sequence"])
            for event in events
            if "sequence" in event["data"]
        ]
        if (
            event_types["human_decision"] != 2
            or event_types["agent_trace"] < 1
            or event_types["run_status"] < 1
            or not sequences
            or sequences != sorted(set(sequences))
        ):
            raise SmokeFailure(f"Invalid durable event stream: {event_types}")
        report["checks"]["durable_events_and_two_human_gates"] = True

        archive_bytes = client.request(
            "GET",
            f"/runs/{run_id}/download",
            expect_json=False,
        )
        with tempfile.TemporaryDirectory(prefix="repopilot-tiny-runtime-") as temp:
            runtime_root = Path(temp)
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                names = set(archive.namelist())
                unsafe = [
                    name
                    for name in names
                    if name.startswith(("/", "\\"))
                    or ".." in name.replace("\\", "/").split("/")
                ]
                if unsafe:
                    raise SmokeFailure(f"Unsafe ZIP entries: {unsafe}")
                missing_zip = sorted(REQUIRED_ZIP_FILES - names)
                if missing_zip:
                    raise SmokeFailure(f"Missing delivery files: {missing_zip}")
                archive.extractall(runtime_root)

            generated_paths = {
                str(item["path"]).replace("\\", "/")
                for item in state["generated_project"]["files"]
            }
            _validate_tiny_python_project(runtime_root, generated_paths)
            runtime_result = _run_tiny_unittests(runtime_root)

        report.update(
            {
                "status": completed["status"],
                "duration_seconds": round(time.monotonic() - started, 2),
                "trace_count": len(traces),
                "generated_file_count": len(state["generated_project"]["files"]),
                "generated_paths": sorted(generated_paths),
                "project_memory_max_items": max(memory_counts),
                "api_usage": usage,
                "event_counts": dict(event_types),
                "runtime_test": runtime_result,
                "zip_size_bytes": len(archive_bytes),
            }
        )
        report["checks"]["tiny_generated_code_runtime"] = True
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
