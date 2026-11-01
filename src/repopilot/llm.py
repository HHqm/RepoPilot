from __future__ import annotations

import json
import random
import re
import time
from abc import ABC, abstractmethod
from threading import local
from typing import Any

from openai import OpenAI

from .config import Settings


class LLMConfigurationError(RuntimeError):
    pass


class StructuredLLM(ABC):
    @abstractmethod
    def complete_json(
        self, *, system_prompt: str, payload: dict[str, Any], schema_name: str
    ) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def route_metadata(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Release provider transports when the implementation owns one."""


_USAGE_KEYS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    }
)
_JSON_OBJECT_RULE = (
    "\n\nReturn exactly one valid JSON object. Do not use Markdown "
    "fences, comments, or prose outside the JSON object."
)


def _is_retryable(exc: Exception) -> bool:
    """Retry transport failures, malformed output, rate limits and 5xx."""

    status_code = getattr(exc, "status_code", None)
    return status_code is None or status_code == 429 or (
        isinstance(status_code, int) and status_code >= 500
    )


class _OpenAIChatLLM(StructuredLLM):
    """Shared transport, bounded retry and usage accounting for chat APIs.

    Usage is stored thread-locally so concurrent runs sharing one adapter can
    never read each other's token counts.
    """

    provider_label = "OpenAI-compatible"
    max_attempts = 3

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_output_tokens: int,
    ) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=90.0,
            max_retries=0,
        )
        self._request_local = local()

    @property
    def last_usage(self) -> dict[str, int]:
        """Usage belongs to the current worker thread, never another run."""

        return dict(getattr(self._request_local, "usage", {}))

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _messages(
        system_prompt: str, payload: dict[str, Any], schema_name: str
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Produce JSON for schema '{schema_name}'. Input:\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            },
        ]

    def _request_json(self, request: dict[str, Any]) -> dict[str, Any]:
        self._request_local.usage = {}
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                response = self.client.chat.completions.create(**request)
                content = response.choices[0].message.content
                if not content:
                    raise ValueError(
                        f"{self.provider_label} returned empty JSON content"
                    )
                if response.usage:
                    self._request_local.usage = {
                        key: int(value)
                        for key, value in response.usage.model_dump().items()
                        if key in _USAGE_KEYS and value is not None
                    }
                return self._parse_json_object(content)
            except Exception as exc:  # SDK exposes multiple transport errors
                last_error = exc
                if attempt + 1 < self.max_attempts and _is_retryable(exc):
                    time.sleep(
                        1.5 * (attempt + 1) + random.uniform(0.0, 0.35)
                    )
                    continue
                break
        status_code = getattr(last_error, "status_code", None)
        suffix = f" HTTP {status_code}" if status_code else ""
        raise RuntimeError(
            f"{self.provider_label} structured request failed after retries: "
            f"{type(last_error).__name__}{suffix}"
        ) from last_error

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
            stripped = re.sub(r"\s*```$", "", stripped)
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start < 0 or end <= start:
                raise
            value = json.loads(stripped[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("Structured response must be a JSON object")
        return value


class DeepSeekLLM(_OpenAIChatLLM):
    provider_label = "DeepSeek"

    def __init__(self, settings: Settings) -> None:
        if not settings.deepseek_api_key:
            raise LLMConfigurationError(
                "DEEPSEEK_API_KEY is required when llm_mode=deepseek"
            )
        super().__init__(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            max_output_tokens=settings.llm_max_output_tokens,
        )

    @property
    def route_metadata(self) -> dict[str, Any]:
        return {
            "provider": "deepseek",
            "model": self.model,
            "credential_source": "environment",
        }

    def complete_json(
        self, *, system_prompt: str, payload: dict[str, Any], schema_name: str
    ) -> dict[str, Any]:
        return self._request_json(
            {
                "model": self.model,
                "messages": self._messages(
                    system_prompt, payload, schema_name
                ),
                "response_format": {"type": "json_object"},
                "max_tokens": self.max_output_tokens,
                "extra_body": {"thinking": {"type": "disabled"}},
            }
        )


class OpenAICompatibleLLM(_OpenAIChatLLM):
    """Provider-neutral adapter for user-supplied OpenAI-compatible APIs."""

    def __init__(self, route: dict[str, Any], settings: Settings) -> None:
        api_key = str(route.get("api_key") or "")
        if not api_key:
            raise LLMConfigurationError("The selected model credential is unavailable")
        self.provider_id = str(route["provider_id"])
        self.provider_name = str(route["provider_name"])
        self.credential_id = str(route["credential_id"])
        self.agent_id = str(route["agent_id"])
        self.supports_json_object = bool(route.get("supports_json_object", True))
        super().__init__(
            api_key=api_key,
            base_url=str(route["base_url"]),
            model=str(route["model"]),
            max_output_tokens=settings.llm_max_output_tokens,
        )

    @property
    def route_metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "provider_id": self.provider_id,
            "model": self.model,
            "agent_id": self.agent_id,
            "credential_id": self.credential_id,
        }

    def complete_json(
        self, *, system_prompt: str, payload: dict[str, Any], schema_name: str
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(
                system_prompt + _JSON_OBJECT_RULE, payload, schema_name
            ),
            "max_tokens": self.max_output_tokens,
        }
        if self.supports_json_object:
            request["response_format"] = {"type": "json_object"}
        return self._request_json(request)


class MockLLM(StructuredLLM):
    """Deterministic test double used by CI and the bundled demo."""

    @staticmethod
    def _language(payload: dict[str, Any]) -> str:
        preference = str(
            payload.get("preferred_stack")
            or payload.get("architecture", {}).get("primary_language")
            or ""
        ).lower()
        if "java" in preference:
            return "Java 21"
        if "c++" in preference or "cpp" in preference:
            return "C++20"
        if "go" in preference or "golang" in preference:
            return "Go 1.22"
        if any(value in preference for value in ("typescript", "javascript", "node")):
            return "TypeScript / Node.js 20"
        return "Python 3.11"

    @staticmethod
    def _generated_files(language: str, project_name: str) -> list[dict[str, str]]:
        safe_name = (
            re.sub(r"[^a-z0-9]+", "-", project_name.casefold()).strip("-")
            or "repopilot-project"
        )
        project_literal = json.dumps(project_name, ensure_ascii=False)
        project_response_literal = json.dumps(
            json.dumps({"project": project_name}, ensure_ascii=False),
            ensure_ascii=False,
        )
        ready_message_literal = json.dumps(
            f"{project_name} is ready. Use --health for a health check.\n",
            ensure_ascii=False,
        )
        if language.startswith("Java"):
            return [
                {
                    "path": "pom.xml",
                    "purpose": "Maven build and Java version declaration.",
                    "content": (
                        '<project xmlns="http://maven.apache.org/POM/4.0.0">\n'
                        "  <modelVersion>4.0.0</modelVersion>\n"
                        f"  <groupId>com.{safe_name.replace('-', '')}</groupId>\n"
                        f"  <artifactId>{safe_name}</artifactId>\n"
                        "  <version>0.1.0</version>\n"
                        "  <properties><maven.compiler.release>21</maven.compiler.release>"
                        "<project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>"
                        "</properties>\n"
                        "</project>\n"
                    ),
                },
                {
                    "path": "src/main/java/com/example/App.java",
                    "purpose": "Runnable HTTP service with health and product endpoints.",
                    "content": (
                        "package com.example;\n\n"
                        "import com.sun.net.httpserver.HttpServer;\n"
                        "import java.net.InetSocketAddress;\n"
                        "import java.nio.charset.StandardCharsets;\n\n"
                        "public final class App {\n"
                        "  public static void main(String[] args) throws Exception {\n"
                        "    int port = Integer.parseInt(System.getenv().getOrDefault"
                        '(\"PORT\", \"8080\"));\n'
                        "    HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);\n"
                        '    server.createContext("/health", exchange -> respond(exchange, '
                        '200, "{\\"status\\":\\"ok\\"}"));\n'
                        '    server.createContext("/api/summary", exchange -> respond(exchange, '
                        f"200, {project_response_literal}));\n"
                        "    server.start();\n"
                        "  }\n"
                        "  private static void respond(com.sun.net.httpserver.HttpExchange exchange,"
                        " int status, String body) throws java.io.IOException {\n"
                        "    byte[] bytes = body.getBytes(StandardCharsets.UTF_8);\n"
                        '    exchange.getResponseHeaders().set("Content-Type", "application/json");\n'
                        "    exchange.sendResponseHeaders(status, bytes.length);\n"
                        "    exchange.getResponseBody().write(bytes);\n"
                        "    exchange.close();\n"
                        "  }\n"
                        "}\n"
                    ),
                },
                {
                    "path": "Dockerfile",
                    "purpose": "Reproducible container build.",
                    "content": (
                        "FROM maven:3.9-eclipse-temurin-21 AS build\n"
                        "WORKDIR /app\nCOPY . .\nRUN mvn -q -DskipTests package\n"
                        "FROM eclipse-temurin:21-jre\nWORKDIR /app\n"
                        "COPY --from=build /app/target/*.jar app.jar\n"
                        'EXPOSE 8080\nCMD ["java", "-cp", "app.jar", "com.example.App"]\n'
                    ),
                },
            ]
        if language.startswith("Go"):
            module = safe_name
            return [
                {
                    "path": "go.mod",
                    "purpose": "Go module declaration.",
                    "content": f"module example.com/{module}\n\ngo 1.22\n",
                },
                {
                    "path": "cmd/server/main.go",
                    "purpose": "Runnable Go HTTP service.",
                    "content": (
                        "package main\n\n"
                        'import ("encoding/json"; "log"; "net/http"; "os")\n\n'
                        "func main() {\n"
                        '  http.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) '
                        '{ json.NewEncoder(w).Encode(map[string]string{"status":"ok"}) })\n'
                        '  http.HandleFunc("/api/summary", func(w http.ResponseWriter, _ *http.Request) '
                        f'{{ json.NewEncoder(w).Encode(map[string]string{{"project":{project_literal}}}) }})\n'
                        '  port := os.Getenv("PORT"); if port == "" { port = "8080" }\n'
                        '  log.Fatal(http.ListenAndServe(":"+port, nil))\n'
                        "}\n"
                    ),
                },
                {
                    "path": "Dockerfile",
                    "purpose": "Reproducible multi-stage Go container.",
                    "content": (
                        "FROM golang:1.22 AS build\nWORKDIR /src\nCOPY . .\n"
                        "RUN CGO_ENABLED=0 go build -o /server ./cmd/server\n"
                        "FROM gcr.io/distroless/static-debian12\nCOPY --from=build /server /server\n"
                        'EXPOSE 8080\nENTRYPOINT ["/server"]\n'
                    ),
                },
            ]
        if language.startswith("C++"):
            return [
                {
                    "path": "CMakeLists.txt",
                    "purpose": "Portable C++ build definition.",
                    "content": (
                        "cmake_minimum_required(VERSION 3.20)\n"
                        f"project({safe_name.replace('-', '_')} LANGUAGES CXX)\n"
                        "set(CMAKE_CXX_STANDARD 20)\n"
                        "add_executable(product src/main.cpp)\n"
                    ),
                },
                {
                    "path": "src/main.cpp",
                    "purpose": "Runnable CLI MVP with a machine-readable health command.",
                    "content": (
                        "#include <iostream>\n#include <string>\n\n"
                        "int main(int argc, char** argv) {\n"
                        '  if (argc > 1 && std::string(argv[1]) == "--health") {\n'
                        '    std::cout << "{\\"status\\":\\"ok\\"}\\n"; return 0;\n'
                        "  }\n"
                        f"  std::cout << {ready_message_literal};\n"
                        "  return 0;\n}\n"
                    ),
                },
                {
                    "path": "Dockerfile",
                    "purpose": "Reproducible C++ container build.",
                    "content": (
                        "FROM gcc:14 AS build\nWORKDIR /src\nCOPY . .\n"
                        "RUN g++ -std=c++20 -O2 src/main.cpp -o /product\n"
                        "FROM debian:bookworm-slim\nCOPY --from=build /product /product\n"
                        'ENTRYPOINT ["/product"]\n'
                    ),
                },
            ]
        if language.startswith("TypeScript"):
            return [
                {
                    "path": "package.json",
                    "purpose": "Node.js scripts and runtime declaration.",
                    "content": json.dumps(
                        {
                            "name": safe_name,
                            "version": "0.1.0",
                            "private": True,
                            "type": "module",
                            "scripts": {
                                "start": "node src/server.mjs",
                                "test": "node --test",
                            },
                            "engines": {"node": ">=20"},
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                },
                {
                    "path": "src/server.mjs",
                    "purpose": "Dependency-free Node.js HTTP MVP.",
                    "content": (
                        'import http from "node:http";\n'
                        'const port = Number(process.env.PORT || "8080");\n'
                        "const server = http.createServer((request, response) => {\n"
                        '  response.setHeader("content-type", "application/json; charset=utf-8");\n'
                        '  if (request.url === "/health") return response.end(JSON.stringify({status:"ok"}));\n'
                        f'  if (request.url === "/api/summary") return response.end(JSON.stringify({{project:{project_literal}}}));\n'
                        '  response.statusCode = 404; response.end(JSON.stringify({error:"not_found"}));\n'
                        "});\nserver.listen(port);\n"
                    ),
                },
                {
                    "path": "Dockerfile",
                    "purpose": "Minimal Node.js runtime image.",
                    "content": (
                        "FROM node:20-alpine\nWORKDIR /app\nCOPY package.json ./\n"
                        "COPY src ./src\nEXPOSE 8080\nCMD [\"npm\", \"start\"]\n"
                    ),
                },
            ]
        return [
            {
                "path": "pyproject.toml",
                "purpose": "Python package and interpreter declaration.",
                "content": (
                    "[project]\n"
                    f'name = "{safe_name}"\nversion = "0.1.0"\n'
                    'requires-python = ">=3.11"\n\n'
                    "[tool.unittest]\nstart-directory = \"tests\"\n"
                ),
            },
            {
                "path": "src/product/__init__.py",
                "purpose": "Application package marker.",
                "content": '"""Generated product package."""\n',
            },
            {
                "path": "src/product/service.py",
                "purpose": "Testable product-domain service.",
                "content": (
                    "from __future__ import annotations\n\n"
                    "def project_summary() -> dict[str, str]:\n"
                    f'    return {{"project": {project_name!r}, "status": "ready"}}\n'
                ),
            },
            {
                "path": "src/product/main.py",
                "purpose": "Dependency-free HTTP entry point.",
                "content": (
                    "from __future__ import annotations\n\n"
                    "import json\nimport os\nfrom http.server import BaseHTTPRequestHandler, HTTPServer\n\n"
                    "from product.service import project_summary\n\n"
                    "class Handler(BaseHTTPRequestHandler):\n"
                    "    def do_GET(self) -> None:\n"
                    '        payload = {"status": "ok"} if self.path == "/health" else project_summary()\n'
                    "        body = json.dumps(payload, ensure_ascii=False).encode()\n"
                    "        self.send_response(200)\n"
                    '        self.send_header("Content-Type", "application/json; charset=utf-8")\n'
                    "        self.send_header(\"Content-Length\", str(len(body)))\n"
                    "        self.end_headers()\n        self.wfile.write(body)\n\n"
                    "if __name__ == \"__main__\":\n"
                    '    HTTPServer(("0.0.0.0", int(os.getenv("PORT", "8080"))), Handler).serve_forever()\n'
                ),
            },
            {
                "path": "tests/test_service.py",
                "purpose": "Deterministic unit test for the generated domain service.",
                "content": (
                    "import sys\nimport unittest\nfrom pathlib import Path\n\n"
                    "sys.path.insert(0, str(Path(__file__).parents[1] / \"src\"))\n"
                    "from product.service import project_summary\n\n"
                    "class ServiceTests(unittest.TestCase):\n"
                    "    def test_summary_is_ready(self):\n"
                    '        self.assertEqual(project_summary()["status"], "ready")\n\n'
                    'if __name__ == "__main__":\n    unittest.main()\n'
                ),
            },
            {
                "path": "Dockerfile",
                "purpose": "Minimal Python runtime image.",
                "content": (
                    "FROM python:3.11-slim\nWORKDIR /app\nCOPY src ./src\n"
                    "ENV PYTHONPATH=/app/src PORT=8080\nEXPOSE 8080\n"
                    'CMD ["python", "-m", "product.main"]\n'
                ),
            },
        ]

    def complete_json(
        self, *, system_prompt: str, payload: dict[str, Any], schema_name: str
    ) -> dict[str, Any]:
        if schema_name == "product_chat":
            memory = payload.get("conversation_memory") or {}
            messages = memory.get("recent_messages") or payload.get("messages") or []
            client_messages = [
                str(item.get("content") or "")
                for item in messages
                if item.get("role") == "client"
            ]
            assets = payload.get("assets") or []
            latest = client_messages[-1] if client_messages else ""
            ready = len(client_messages) >= 2 or bool(assets)
            questions = [] if ready else [
                "请补充最核心的用户操作流程，以及首期明确不做的范围。",
                "期望部署到什么环境，谁负责上线后的日常运维？",
            ]
            return {
                "reply": (
                    "我已经把这轮信息并入需求底稿。"
                    + (
                        "目前还需要确认："
                        + "；".join(item.rstrip("。？?") for item in questions)
                        + "。"
                        if questions
                        else "核心目标、约束和交付预期已经足够清楚，可以提交跨职能需求评审会。"
                    )
                ),
                "understanding": {
                    "summary": "；".join(client_messages) or latest,
                    "confirmed_goals": [
                        "交付可运行、可测试、可部署的软件项目",
                        "提供源代码与面向客户的详细使用指南",
                    ],
                    "constraints": [
                        str(item)
                        for item in (
                            payload.get("policy_constraints", [])
                            + payload.get("business_constraints", [])
                        )
                    ],
                    "open_questions": questions,
                    "assumptions": [
                        f"已上传 {len(assets)} 份资料并按项目隔离入库。",
                        "真实生产上线需客户提供目标环境与明确授权。",
                    ],
                    "ready_for_workshop": ready,
                },
            }
        if schema_name == "product_requirements":
            name = str(payload.get("project_name") or "New Product")
            feedback = str(payload.get("requirements_feedback") or "").strip()
            assumptions = [
                "The first delivery is an MVP and excludes unapproved scope.",
                "External production deployment requires a separate human release decision.",
            ]
            if feedback:
                assumptions.append(f"Revision incorporates client feedback: {feedback}")
            return {
                "product_name": name,
                "vision": f"Deliver a usable and operable MVP for {name}.",
                "problem_statement": str(payload.get("client_brief") or ""),
                "target_users": [str(payload.get("target_users") or "业务用户")],
                "user_journeys": [
                    "A user opens the product and completes the primary business task.",
                    "An operator checks service health and diagnoses a failed request.",
                ],
                "functional_requirements": [
                    {
                        "requirement_id": "FR-001",
                        "title": "Primary business workflow",
                        "description": str(payload.get("client_brief") or ""),
                        "priority": "must",
                        "acceptance_criteria": [
                            "The primary workflow has a documented input and output.",
                            "Invalid input returns an actionable error without exposing secrets.",
                        ],
                    },
                    {
                        "requirement_id": "FR-002",
                        "title": "Operational health",
                        "description": "Expose a deterministic health signal for deployment checks.",
                        "priority": "must",
                        "acceptance_criteria": [
                            "The documented health check reports an explicit healthy status."
                        ],
                    },
                ],
                "non_functional_requirements": [
                    "No credentials are stored in source control.",
                    "The MVP includes reproducible local run instructions.",
                    "Critical actions and failures are observable without exposing private data.",
                ],
                "in_scope": [
                    "Runnable MVP source code",
                    "Build and deployment configuration",
                    "Operator and end-user documentation",
                ],
                "out_of_scope": [
                    "Purchasing cloud resources",
                    "Registering domains or external accounts",
                    "Production deployment without a human release decision",
                ],
                "assumptions": assumptions,
                "clarification_questions": [
                    "Which business metric determines whether the MVP is successful?",
                    "Who owns production operations and incident response after handover?",
                ],
                "success_metrics": [
                    "The generated project starts using the documented steps.",
                    "The health check passes in the target environment.",
                    "All must-have acceptance criteria are traceable to implementation.",
                ],
            }
        if schema_name == "requirements_workshop":
            requirements = payload.get("product_requirements") or {}
            functional = requirements.get("functional_requirements") or []
            decomposition = []
            for index, item in enumerate(functional, start=1):
                requirement_id = str(item.get("requirement_id") or f"FR-{index:03d}")
                decomposition.extend(
                    [
                        {
                            "item_id": f"W-{index:03d}-DEV",
                            "requirement_id": requirement_id,
                            "workstream": "development",
                            "deliverable": f"Implement {item.get('title', requirement_id)}",
                            "dependencies": [],
                            "done_definition": list(
                                item.get("acceptance_criteria") or ["Peer review passed"]
                            ),
                        },
                        {
                            "item_id": f"W-{index:03d}-QA",
                            "requirement_id": requirement_id,
                            "workstream": "qa",
                            "deliverable": f"Verify {item.get('title', requirement_id)}",
                            "dependencies": [f"W-{index:03d}-DEV"],
                            "done_definition": [
                                f"Acceptance criteria for {requirement_id} are traceable to tests"
                            ],
                        },
                    ]
                )
            return {
                "meeting_title": "客户需求澄清与交付避坑评审会",
                "participants": [
                    "交付总监",
                    "产品经理",
                    "首席架构师",
                    "开发负责人",
                    "测试负责人",
                    "安全审查官",
                    "DevOps 负责人",
                ],
                "client_intent_summary": str(
                    requirements.get("problem_statement") or ""
                ),
                "decisions": [
                    "首期只交付经客户确认的 MVP 范围。",
                    "需求、代码、测试和部署指南必须可追溯。",
                    "未取得目标环境授权前不声称完成生产上线。",
                ],
                "unresolved_questions": list(
                    requirements.get("clarification_questions") or []
                ),
                "risks": [
                    {
                        "risk_id": "R-001",
                        "category": "scope",
                        "scenario": "新增需求在开发中途进入但未调整预算和工期。",
                        "impact": "返工、延期和验收争议。",
                        "avoidance_strategy": "冻结 MVP，新增范围走变更单并重新估算。",
                        "early_warning": "出现未关联 FR 编号的开发任务。",
                        "owner": "产品经理",
                    },
                    {
                        "risk_id": "R-002",
                        "category": "operations",
                        "scenario": "目标运行环境、证书、网络或回滚负责人未确认。",
                        "impact": "代码完成但无法上线。",
                        "avoidance_strategy": "在发布门禁前完成环境清单和回滚演练责任确认。",
                        "early_warning": "发布前仍没有目标环境接入信息。",
                        "owner": "DevOps 负责人",
                    },
                    {
                        "risk_id": "R-003",
                        "category": "security",
                        "scenario": "客户资料中的敏感信息进入日志、代码或模型上下文。",
                        "impact": "隐私与合规风险。",
                        "avoidance_strategy": "项目级隔离、秘密脱敏、最小化检索和禁止源码硬编码。",
                        "early_warning": "静态扫描出现凭据或个人敏感字段。",
                        "owner": "安全审查官",
                    },
                ],
                "decomposition": decomposition
                or [
                    {
                        "item_id": "W-001",
                        "requirement_id": "FR-001",
                        "workstream": "development",
                        "deliverable": "实现核心用户路径",
                        "dependencies": [],
                        "done_definition": ["核心路径有明确输入、输出和错误行为"],
                    }
                ],
                "testing_strategy": [
                    "每个 must-have FR 至少对应一个测试用例。",
                    "静态、契约、安全和文档测试在发布前完成。",
                    "需要真实运行环境的测试明确标为 runtime_required。",
                ],
                "delivery_boundaries": [
                    "交付 ZIP 包、源代码、需求与架构文档、测试报告和部署指南。",
                    "不代购云资源，不创建外部账号，不在未授权环境执行生产部署。",
                ],
                "readiness_recommendation": "风险已显式分配，可提交客户确认后进入研发。",
            }
        if schema_name == "project_plan":
            budget = payload.get("budget_cny")
            per_phase = int(budget / 5) if isinstance(budget, int) and budget else None
            phase_names = [
                ("PH-01", "需求与体验确认", "Freeze the MVP scope and user journey."),
                ("PH-02", "架构与风险设计", "Select the stack and close high-risk decisions."),
                ("PH-03", "开发与持续审查", "Implement the vertical MVP slice."),
                ("PH-04", "质量与安全验证", "Verify functionality, security, and operability."),
                ("PH-05", "发布与移交", "Prepare release conditions and operating guidance."),
            ]
            return {
                "methodology": "Human-gated iterative delivery",
                "executive_summary": "Deliver one traceable MVP through explicit quality and release gates.",
                "phases": [
                    {
                        "phase_id": phase_id,
                        "name": name,
                        "objective": objective,
                        "deliverables": [f"{name}交付物", "阶段决策记录"],
                        "estimated_days": 3 if index < 2 else 5,
                        "budget_cny": per_phase,
                        "dependencies": [] if index == 0 else [phase_names[index - 1][0]],
                        "exit_criteria": ["阶段交付物完整", "阻断风险有责任人和处置计划"],
                    }
                    for index, (phase_id, name, objective) in enumerate(phase_names)
                ],
                "team_roles": [
                    "产品经理",
                    "项目经理",
                    "架构师",
                    "开发工程师",
                    "安全审查员",
                    "测试工程师",
                    "DevOps/运维工程师",
                ],
                "budget_summary": (
                    f"客户预算上限为 {budget} CNY，按五个阶段设置预算门禁。"
                    if budget
                    else "客户尚未给出固定预算；每个阶段估算后再由人类确认。"
                ),
                "schedule_summary": str(
                    payload.get("delivery_deadline") or "待确认交付期限"
                ),
                "risks": [
                    "需求歧义导致返工",
                    "目标环境与所选技术栈不兼容",
                    "合规或政治约束在开发后期才被发现",
                    "预算不足以覆盖生产级高可用要求",
                ],
                "compliance_actions": [
                    str(item) for item in payload.get("policy_constraints", [])
                ]
                or ["在发布前完成适用法规、数据与内容政策检查"],
                "communication_plan": [
                    "需求确认点由产品经理与客户签字确认",
                    "每个阶段结束发布风险、预算与范围状态",
                    "上线前由客户负责人做最终 Release Gate 决策",
                ],
            }
        if schema_name == "architecture_blueprint":
            language = self._language(payload)
            stack_map = {
                "Python 3.11": ["Python 3.11", "stdlib HTTP", "unittest", "Docker"],
                "Java 21": ["Java 21", "Maven", "JDK HttpServer", "Docker"],
                "Go 1.22": ["Go 1.22", "net/http", "go test", "Docker"],
                "C++20": ["C++20", "CMake", "standard library", "Docker"],
                "TypeScript / Node.js 20": [
                    "Node.js 20",
                    "ES modules",
                    "node:test",
                    "Docker",
                ],
            }
            return {
                "architecture_style": "Modular monolith with explicit operational boundary",
                "primary_language": language,
                "recommended_stack": stack_map[language],
                "rationale": [
                    f"The client preference and delivery constraints support {language}.",
                    "A modular monolith minimizes cost and operational complexity for an MVP.",
                    "Container packaging keeps local and production execution consistent.",
                ],
                "components": [
                    {
                        "name": "Product API",
                        "responsibility": "Expose the primary user workflow and health signal.",
                        "technology": language,
                        "interfaces": ["/health", "/api/summary"],
                    },
                    {
                        "name": "Operations boundary",
                        "responsibility": "Configuration, logging, deployment, and rollback.",
                        "technology": "Environment variables and Docker",
                        "interfaces": ["PORT", "container health check"],
                    },
                ],
                "data_design": [
                    "The MVP keeps state minimal; introduce a managed database only when persistence is confirmed."
                ],
                "api_design": [
                    "GET /health returns machine-readable service health.",
                    "GET /api/summary demonstrates the product integration boundary.",
                ],
                "security_design": [
                    "No credentials in source or images.",
                    "Validate external input at the API boundary.",
                    "Run containers as a non-privileged identity in production hardening.",
                ],
                "observability": [
                    "Health endpoint",
                    "Structured request and error logs",
                    "Release version recorded with deployment",
                ],
                "deployment_topology": [
                    str(payload.get("deployment_target") or "undecided"),
                    "One stateless application instance for MVP; scale only after load evidence.",
                ],
            }
        if schema_name == "generated_project":
            architecture = payload.get("architecture") or {}
            language = self._language(
                {
                    "preferred_stack": architecture.get("primary_language")
                    or payload.get("preferred_stack")
                }
            )
            project_name = str(payload.get("project_name") or "software-product")
            files = self._generated_files(language, project_name)
            files.append(
                {
                    "path": "README.md",
                    "purpose": "Project overview and pointer to the generated deployment guide.",
                    "content": (
                        f"# {project_name}\n\n"
                        "This MVP was generated from a human-confirmed PRD and reviewed before write.\n\n"
                        f"- Primary language: {language}\n"
                        "- Read `DEPLOYMENT.md` before running or releasing the product.\n"
                        "- Never commit real credentials; use the environment variables documented there.\n"
                    ),
                }
            )
            return {
                "summary": f"Generated a compact runnable {language} MVP.",
                "files": files,
            }
        if schema_name == "project_revision":
            current = payload.get("current_project") or {}
            files = list(current.get("files") or [])
            if not files:
                raise ValueError("Mock project revision requires current files")
            defects = [str(item) for item in payload.get("qa_defects") or []]
            return {
                "summary": "Applied a bounded mock revision for QA feedback.",
                "changed_files": [files[0]],
                "delete_paths": [],
                "addressed_defects": defects[:1]
                or ["Preserved the deterministic fixture without scope expansion."],
            }
        if schema_name == "studio_review":
            files = payload.get("files") or []
            content = "\n".join(str(item.get("content") or "") for item in files)
            dangerous = any(
                marker in content
                for marker in ("eval(", "exec(", "shell=True", "sk-", "AKIA")
            )
            return {
                "approved": not dangerous,
                "summary": (
                    "The generated MVP is traceable to the PRD and ready for human release review."
                    if not dangerous
                    else "The generated project contains a blocking security pattern."
                ),
                "findings": (
                    []
                    if not dangerous
                    else [
                        {
                            "severity": "critical",
                            "message": "Dangerous code or credential pattern detected.",
                            "path": None,
                        }
                    ]
                ),
            }
        if schema_name == "qa_report":
            requirements = (
                payload.get("product_requirements", {}).get(
                    "functional_requirements", []
                )
            )
            cases = []
            for index, requirement in enumerate(requirements, start=1):
                requirement_id = str(
                    requirement.get("requirement_id") or f"FR-{index:03d}"
                )
                cases.append(
                    {
                        "case_id": f"TC-{index:03d}",
                        "requirement_id": requirement_id,
                        "name": f"{requirement_id} acceptance traceability",
                        "method": "contract",
                        "status": "passed",
                        "evidence": "Requirement is mapped to generated source and documented behavior.",
                    }
                )
            cases.extend(
                [
                    {
                        "case_id": "TC-SEC-001",
                        "requirement_id": "NFR-SEC",
                        "name": "Secret and dangerous-pattern scan",
                        "method": "security",
                        "status": "passed",
                        "evidence": "Deterministic scan found no blocking pattern.",
                    },
                    {
                        "case_id": "TC-RUN-001",
                        "requirement_id": "NFR-OPS",
                        "name": "Target-environment startup and health check",
                        "method": "runtime",
                        "status": "runtime_required",
                        "evidence": "Must run in isolated client CI or target environment.",
                    },
                ]
            )
            language = str(
                payload.get("architecture", {}).get("primary_language")
                or "selected runtime"
            )
            commands = {
                "Python 3.11": ["python -m unittest discover -s tests"],
                "Java 21": ["mvn test", "mvn package"],
                "Go 1.22": ["go test ./...", "go build ./..."],
                "C++20": [
                    "cmake -S . -B build",
                    "cmake --build build",
                    "ctest --test-dir build",
                ],
                "TypeScript / Node.js 20": ["npm test", "npm start"],
            }.get(language, ["Use the generated build manifest in isolated CI."])
            return {
                "passed": True,
                "summary": (
                    "Static, contract, security and documentation checks passed; "
                    "target-environment runtime checks remain explicitly assigned."
                ),
                "test_cases": cases,
                "defects": [],
                "acceptance_coverage": [
                    f"{case['requirement_id']} -> {case['case_id']}"
                    for case in cases
                ],
                "recommended_commands": commands,
                "runtime_validation_required": True,
                "environment_limitations": [
                    "Generated code is not executed directly on the RepoPilot host.",
                    "Container, network, database and external integration checks require isolated CI or the client target environment.",
                ],
            }
        if schema_name == "deployment_guide":
            architecture = payload.get("architecture") or {}
            language = str(architecture.get("primary_language") or "selected runtime")
            target = str(payload.get("deployment_target") or "undecided")
            commands = {
                "Python 3.11": [
                    "Set PYTHONPATH=src and run: python -m product.main",
                    "Run unit checks: python -m unittest discover -s tests",
                ],
                "Java 21": [
                    "Build: mvn clean package",
                    "Run the generated main class using Java 21.",
                ],
                "Go 1.22": [
                    "Run checks: go test ./...",
                    "Start: go run ./cmd/server",
                ],
                "C++20": [
                    "Configure: cmake -S . -B build",
                    "Build: cmake --build build",
                    "Run: ./build/product",
                ],
                "TypeScript / Node.js 20": [
                    "Run checks: npm test",
                    "Start: npm start",
                ],
            }.get(language, ["Follow the generated build manifest for the selected runtime."])
            return {
                "target": target,
                "prerequisites": [
                    language,
                    "Git",
                    "Docker when using the container release path",
                    "A deployment account owned by the client",
                ],
                "environment_variables": [
                    "PORT — listening port; defaults to 8080",
                    "Never place API keys or passwords in source control",
                ],
                "local_run_steps": commands,
                "production_steps": [
                    "Review the PRD, architecture, generated source, and quality report.",
                    "Build the container image from the generated Dockerfile.",
                    "Scan the image and dependencies in the client CI environment.",
                    "Deploy first to a non-production environment.",
                    "Run the documented health and smoke checks.",
                    "Promote the exact reviewed image digest after human release approval.",
                ],
                "health_checks": [
                    "Call /health and require a successful response with status=ok.",
                    "Exercise the primary user journey using non-sensitive test data.",
                    "Confirm logs contain no credentials or private request payloads.",
                ],
                "release_conditions": [
                    "Human-confirmed PRD and release decision are recorded.",
                    "All deterministic quality checks pass.",
                    "Target environment satisfies runtime, network, storage, and policy constraints.",
                    "Rollback owner and previous stable version are identified.",
                ],
                "rollback_steps": [
                    "Stop promotion and preserve failing logs.",
                    "Route traffic back to the previous stable image or binary.",
                    "Verify the previous version health check.",
                    "Open a corrective project run before another release attempt.",
                ],
                "usage_guide": [
                    "Start the service with the local or container command.",
                    "Check /health before sending business traffic.",
                    "Use /api/summary as the initial integration example.",
                    "Replace the demonstration workflow only through a reviewed change request.",
                ],
                "environment_specification": [
                    f"Runtime: {language}",
                    "Operating system: 64-bit Linux recommended for the documented container path",
                    "CPU and memory: confirm against workload; MVP baseline is 2 CPU and 4 GB RAM",
                    "Network: reserve the documented service port and configure TLS at the ingress or reverse proxy",
                    "Storage: persistent storage is required only when the architecture introduces state",
                ],
                "build_steps": [
                    *commands[:1],
                    "Build from a clean checkout using the exact reviewed source revision.",
                    "Record the binary or image digest in the release record.",
                ],
                "test_steps": [
                    *(commands[1:] or commands[:1]),
                    "Run static, contract and security checks in isolated CI.",
                    "Run smoke and health checks with non-sensitive test data before production traffic.",
                ],
                "boundary_conditions": [
                    "The ZIP proves generation and review, not successful deployment to an external platform.",
                    "Cloud accounts, domains, certificates, databases and production credentials are client-owned inputs.",
                    "Environment-specific integrations remain blocked until access, schemas and owners are confirmed.",
                ],
                "troubleshooting": [
                    "Service does not start -> verify runtime version, environment variables, port conflicts and logs.",
                    "Health check fails -> compare the deployed digest, configuration and dependency reachability.",
                    "Permission denied -> verify file ownership and the non-privileged runtime identity.",
                    "Rollback fails -> stop traffic promotion and restore the previous recorded digest.",
                ],
                "backup_restore": [
                    "Back up stateful data before schema or release changes.",
                    "Test restore in a non-production environment and record recovery time.",
                ],
                "observability": [
                    "Collect structured startup, request and error logs without request secrets.",
                    "Monitor health, latency, error rate and resource saturation.",
                    "Alert ownership and escalation contacts must be assigned before production release.",
                ],
                "security_checklist": [
                    "No credentials in source, image layers or logs.",
                    "Dependencies and container image scanned in client CI.",
                    "TLS, least privilege, network exposure and backup access reviewed.",
                    "Uploaded client documents remain project-scoped and are not included in the public source tree.",
                ],
                "handover_checklist": [
                    "Client has the ZIP, checksum and source manifest.",
                    "Client reviewed PRD, workshop minutes, architecture and QA limitations.",
                    "Operators can start, stop, verify, back up and roll back the service.",
                    "Production owner, incident contact and change process are recorded.",
                ],
            }
        if schema_name == "change_plan":
            repo_map = payload.get("repo_map", [])
            target = next(
                (
                    item["path"]
                    for item in repo_map
                    if "calculator" in item["path"].lower()
                ),
                repo_map[0]["path"] if repo_map else "src/calculator.py",
            )
            evidence = payload.get("evidence", [])
            evidence_ids = [item["evidence_id"] for item in evidence[:2]]
            return {
                "summary": "Add an explicit zero-divisor guard and verify behavior.",
                "assumptions": ["The public function signature must remain unchanged."],
                "steps": [
                    {
                        "step_id": "P1",
                        "description": "Add a ValueError guard before division.",
                        "target_files": [target],
                        "evidence_ids": evidence_ids,
                        "risk": "low",
                        "verification": "Run the repository unit test suite.",
                    }
                ],
            }
        if schema_name == "code_edits":
            files = payload.get("files", [])
            for item in files:
                content = "\n".join(item.get("excerpts", []))
                if "return a / b" in content:
                    return {
                        "edits": [
                            {
                                "path": item["path"],
                                "reasoning": (
                                    "Convert the implicit ZeroDivisionError into the "
                                    "explicit ValueError required by the issue."
                                ),
                                "search": "    return a / b",
                                "replace": (
                                    '    if b == 0:\n'
                                    '        raise ValueError("divisor must not be zero")\n'
                                    "    return a / b"
                                ),
                            }
                        ]
                    }
            raise ValueError("MockLLM demo pattern was not found in target files")
        if schema_name == "review_decision":
            diffs = payload.get("diffs", "")
            dangerous = any(
                marker in diffs
                for marker in ("eval(", "exec(", "shell=True", "sk-", "AKIA")
            )
            return {
                "approved": not dangerous,
                "summary": (
                    "The patch is minimal and matches the requested behavior."
                    if not dangerous
                    else "The patch contains a dangerous pattern."
                ),
                "findings": (
                    []
                    if not dangerous
                    else [
                        {
                            "severity": "critical",
                            "message": "Dangerous code or credential pattern detected.",
                            "path": None,
                        }
                    ]
                ),
            }
        if schema_name == "reflection":
            return {
                "root_cause": "The applied change did not satisfy the allowlisted tests.",
                "next_action": (
                    "Inspect the failing assertion, retrieve additional context, "
                    "and create a new reviewed proposal."
                ),
                "retry_recommended": True,
            }
        raise ValueError(f"Unsupported mock schema: {schema_name}")


def build_llm(mode: str, settings: Settings) -> StructuredLLM:
    if mode == "deepseek":
        return DeepSeekLLM(settings)
    if mode == "mock":
        return MockLLM()
    raise ValueError(f"Unsupported LLM mode: {mode}")


def build_routed_llm(
    route: dict[str, Any], settings: Settings
) -> StructuredLLM:
    if route["credential_id"] == "builtin:mock":
        return MockLLM()
    return OpenAICompatibleLLM(route, settings)
