from __future__ import annotations

import ipaddress
import socket
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from openai import OpenAI

from .config import Settings
from .models import (
    AgentModelRoute,
    ModelRoutingConfig,
    ModelRoutingValidation,
    ProviderCredentialCreate,
    ProviderCredentialView,
    ProviderModelView,
)


BUILTIN_PROVIDER_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "preset_id": "deepseek",
        "name": "DeepSeek",
        "protocol": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "supports_model_discovery": True,
        "supports_json_object": True,
        "description": "DeepSeek 官方 API；可使用服务端环境变量或用户凭证。",
    },
    {
        "preset_id": "openai",
        "name": "OpenAI",
        "protocol": "openai_compatible",
        "base_url": "https://api.openai.com/v1",
        "supports_model_discovery": True,
        "supports_json_object": True,
        "description": "OpenAI API，模型列表在连接后动态获取。",
    },
    {
        "preset_id": "qwen",
        "name": "通义千问 DashScope",
        "protocol": "openai_compatible",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "supports_model_discovery": True,
        "supports_json_object": True,
        "description": "阿里云百炼 OpenAI-compatible 接口。",
    },
    {
        "preset_id": "doubao",
        "name": "豆包 Ark",
        "protocol": "openai_compatible",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "supports_model_discovery": True,
        "supports_json_object": True,
        "description": "火山方舟兼容接口；模型 ID 可能是用户创建的推理接入点。",
    },
    {
        "preset_id": "zhipu",
        "name": "智谱 GLM",
        "protocol": "openai_compatible",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "supports_model_discovery": True,
        "supports_json_object": True,
        "description": "智谱开放平台 OpenAI-compatible 接口。",
    },
    {
        "preset_id": "moonshot",
        "name": "Moonshot / Kimi",
        "protocol": "openai_compatible",
        "base_url": "https://api.moonshot.cn/v1",
        "supports_model_discovery": True,
        "supports_json_object": True,
        "description": "Moonshot OpenAI-compatible 接口。",
    },
    {
        "preset_id": "gemini",
        "name": "Google Gemini",
        "protocol": "openai_compatible",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "supports_model_discovery": True,
        "supports_json_object": True,
        "description": "Gemini 的 OpenAI compatibility 接口。",
    },
    {
        "preset_id": "zzshu",
        "name": "ZZSHU 第三方网关",
        "protocol": "openai_compatible",
        "base_url": "https://zzshu.cc/v1",
        "supports_model_discovery": True,
        "supports_json_object": True,
        "description": "第三方 OpenAI-compatible 网关。",
    },
    {
        "preset_id": "ollama",
        "name": "Ollama / 本地模型",
        "protocol": "openai_compatible",
        "base_url": "http://127.0.0.1:11434/v1",
        "supports_model_discovery": True,
        "supports_json_object": False,
        "description": "仅在管理员允许私网模型端点时可连接。",
    },
    {
        "preset_id": "custom",
        "name": "自定义 OpenAI-compatible",
        "protocol": "openai_compatible",
        "base_url": "https://",
        "supports_model_discovery": True,
        "supports_json_object": True,
        "description": "填写任意兼容端点；不支持模型发现时可手动填写模型 ID。",
    },
)


LLM_AGENT_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "agent_id": "ProductManagerAgent",
        "name": "产品经理",
        "profile": "中文理解、长上下文、结构化输出",
        "schemas": ["product_chat", "product_requirements"],
    },
    {
        "agent_id": "DeliveryDirectorAgent",
        "name": "交付总监",
        "profile": "复杂推理、风险分析、长上下文",
        "schemas": ["requirements_workshop"],
    },
    {
        "agent_id": "ProjectManagerAgent",
        "name": "项目经理",
        "profile": "任务规划、预算排期、结构化输出",
        "schemas": ["project_plan"],
    },
    {
        "agent_id": "ArchitectAgent",
        "name": "首席架构师",
        "profile": "架构推理、技术选型、代码理解",
        "schemas": ["architecture_blueprint"],
    },
    {
        "agent_id": "DevelopmentTeamAgent",
        "name": "开发团队",
        "profile": "代码生成、长输出、定向修复",
        "schemas": ["generated_project", "project_revision"],
    },
    {
        "agent_id": "DevOpsAgent",
        "name": "DevOps",
        "profile": "部署配置、运行手册、故障处理",
        "schemas": ["deployment_guide"],
    },
    {
        "agent_id": "QualityAssuranceAgent",
        "name": "测试负责人",
        "profile": "测试设计、代码审查、缺陷定位",
        "schemas": ["qa_report"],
    },
    {
        "agent_id": "ReleaseReviewerAgent",
        "name": "发布审查官",
        "profile": "独立安全审查、边界核验、结构化判断",
        "schemas": ["studio_review"],
    },
)


SCHEMA_AGENT_ROUTES = {
    schema: item["agent_id"]
    for item in LLM_AGENT_PROFILES
    for schema in item["schemas"]
}
SCHEMA_AGENT_ROUTES.update(
    {
        "change_plan": "PlannerAgent",
        "code_edits": "CoderAgent",
        "review_decision": "ReviewerAgent",
        "reflection": "ReflectionAgent",
    }
)


@dataclass(frozen=True)
class _CredentialRecord:
    credential_id: str
    preset_id: str
    provider_name: str
    base_url: str
    api_key: str
    secret_hint: str
    source: str
    status: str
    supports_json_object: bool
    models: tuple[ProviderModelView, ...]
    note: str = ""

    def view(self) -> ProviderCredentialView:
        return ProviderCredentialView(
            credential_id=self.credential_id,
            preset_id=self.preset_id,
            provider_name=self.provider_name,
            base_url=self.base_url,
            secret_hint=self.secret_hint,
            source=self.source,
            status=self.status,
            supports_json_object=self.supports_json_object,
            models=list(self.models),
            note=self.note,
        )


class ModelProviderRegistry:
    """Thread-safe process-local BYOK vault and route resolver.

    User secrets intentionally never touch disk. Deployments that require
    durable credentials should provide them through an external secret manager
    and register them at process startup.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = RLock()
        self._credentials: OrderedDict[str, _CredentialRecord] = OrderedDict()
        self._credentials["builtin:mock"] = _CredentialRecord(
            credential_id="builtin:mock",
            preset_id="mock",
            provider_name="Mock",
            base_url="builtin://mock",
            api_key="",
            secret_hint="无需密钥",
            source="builtin",
            status="ready",
            supports_json_object=True,
            models=(
                ProviderModelView(id="mock-deterministic", owned_by="RepoPilot"),
            ),
            note="确定性测试替身，不产生外部 API 调用或费用。",
        )
        if settings.deepseek_api_key:
            self._credentials["env:deepseek"] = _CredentialRecord(
                credential_id="env:deepseek",
                preset_id="deepseek",
                provider_name="DeepSeek（服务端）",
                base_url=settings.deepseek_base_url,
                api_key=settings.deepseek_api_key,
                secret_hint=self._secret_hint(settings.deepseek_api_key),
                source="environment",
                status="ready",
                supports_json_object=True,
                models=(
                    ProviderModelView(
                        id=settings.deepseek_model,
                        owned_by="DeepSeek",
                    ),
                ),
                note="由服务端环境变量托管。",
            )

    @staticmethod
    def catalog() -> list[dict[str, Any]]:
        return [dict(item) for item in BUILTIN_PROVIDER_CATALOG]

    def list_credentials(self) -> list[ProviderCredentialView]:
        with self._lock:
            return [record.view() for record in self._credentials.values()]

    def get_view(self, credential_id: str) -> ProviderCredentialView:
        with self._lock:
            record = self._credentials.get(credential_id)
            if record is None:
                raise KeyError(f"Unknown model credential: {credential_id}")
            return record.view()

    def add_credential(
        self, request: ProviderCredentialCreate
    ) -> ProviderCredentialView:
        api_key = request.api_key.get_secret_value().strip()
        if not api_key:
            raise ValueError("API Key 不能为空")
        preset = next(
            (
                item
                for item in BUILTIN_PROVIDER_CATALOG
                if item["preset_id"] == request.preset_id
            ),
            None,
        )
        if preset is None:
            raise ValueError(f"未知供应商预设：{request.preset_id}")
        base_url = self.validate_base_url(request.base_url)
        models: list[ProviderModelView] = []
        discovery_error = ""
        try:
            models = self._discover_models(base_url, api_key)
        except Exception as exc:
            discovery_error = self._safe_provider_error(exc)
        if request.manual_model:
            if request.manual_model not in {item.id for item in models}:
                try:
                    self._probe_model(
                        base_url,
                        api_key,
                        request.manual_model,
                    )
                except Exception as exc:
                    raise ValueError(
                        "手动模型 ID 不在发现列表中，且最小调用验证失败："
                        f"{self._safe_provider_error(exc)}"
                    ) from exc
                models.insert(
                    0,
                    ProviderModelView(
                        id=request.manual_model,
                        owned_by=request.display_name,
                    ),
                )
        if not models:
            raise ValueError(
                "无法获取模型列表。若供应商不支持 /models，请填写模型 ID 后重试。"
                + (f" 连接结果：{discovery_error}" if discovery_error else "")
            )
        credential_id = f"cred:{uuid.uuid4().hex}"
        status = "manual" if discovery_error else "ready"
        note = (
            f"模型列表接口不可用，已采用手动模型 ID。{discovery_error}"
            if discovery_error
            else f"连接成功，发现 {len(models)} 个模型。"
        )
        record = _CredentialRecord(
            credential_id=credential_id,
            preset_id=request.preset_id,
            provider_name=request.display_name.strip(),
            base_url=base_url,
            api_key=api_key,
            secret_hint=self._secret_hint(api_key),
            source="session",
            status=status,
            supports_json_object=request.supports_json_object,
            models=tuple(models[:200]),
            note=note,
        )
        with self._lock:
            session_count = sum(
                item.source == "session" for item in self._credentials.values()
            )
            if session_count >= 32:
                raise RuntimeError("本次服务进程最多保存 32 个用户模型凭证")
            self._credentials[credential_id] = record
        return record.view()

    def refresh_models(self, credential_id: str) -> ProviderCredentialView:
        with self._lock:
            record = self._credentials.get(credential_id)
        if record is None:
            raise KeyError(f"Unknown model credential: {credential_id}")
        if record.source == "builtin":
            return record.view()
        models = self._discover_models(record.base_url, record.api_key)
        updated = _CredentialRecord(
            **{
                **record.__dict__,
                "status": "ready",
                "models": tuple(models[:200]),
                "note": f"连接成功，发现 {len(models)} 个模型。",
            }
        )
        with self._lock:
            self._credentials[credential_id] = updated
        return updated.view()

    def delete_credential(self, credential_id: str) -> None:
        with self._lock:
            record = self._credentials.get(credential_id)
            if record is None:
                raise KeyError(f"Unknown model credential: {credential_id}")
            if record.source != "session":
                raise PermissionError("内置或环境变量凭证不能从前端删除")
            self._credentials.pop(credential_id)

    def validate_routing(
        self, routing: ModelRoutingConfig
    ) -> ModelRoutingValidation:
        allowed_agents = {
            item["agent_id"] for item in LLM_AGENT_PROFILES
        } | {"PlannerAgent", "CoderAgent", "ReviewerAgent", "ReflectionAgent"}
        unknown_agents = set(routing.agent_routes) - allowed_agents
        if unknown_agents:
            raise ValueError(
                f"路由包含未知或非 LLM Agent：{sorted(unknown_agents)}"
            )
        warnings: list[str] = []
        self._validate_route(routing.default_route, "default", warnings)
        for agent_id, route in routing.agent_routes.items():
            self._validate_route(route, agent_id, warnings)
        reviewer = routing.agent_routes.get("ReleaseReviewerAgent")
        developer = routing.agent_routes.get("DevelopmentTeamAgent")
        if (
            reviewer
            and developer
            and reviewer.credential_id == developer.credential_id
            and reviewer.model == developer.model
        ):
            warnings.append(
                "开发团队与发布审查官使用同一模型；建议改用不同供应商以降低同源错误。"
            )
        return ModelRoutingValidation(
            valid=True,
            routing=routing,
            warnings=warnings,
        )

    def resolve(
        self, routing: dict[str, Any] | ModelRoutingConfig, schema_name: str
    ) -> dict[str, Any]:
        contract = (
            routing
            if isinstance(routing, ModelRoutingConfig)
            else ModelRoutingConfig.model_validate(routing)
        )
        agent_id = SCHEMA_AGENT_ROUTES.get(schema_name, schema_name)
        route = contract.agent_routes.get(agent_id, contract.default_route)
        with self._lock:
            record = self._credentials.get(route.credential_id)
        if record is None:
            raise RuntimeError(
                f"模型凭证 {route.credential_id} 已不存在；请重新配置后启动新项目"
            )
        self._validate_route(route, agent_id, [])
        return {
            "credential_id": record.credential_id,
            "provider_id": record.preset_id,
            "provider_name": record.provider_name,
            "base_url": record.base_url,
            "api_key": record.api_key,
            "model": route.model,
            "supports_json_object": record.supports_json_object,
            "agent_id": agent_id,
            "cache_key": (
                f"routed:{record.credential_id}:{route.model}:"
                f"{int(record.supports_json_object)}"
            ),
        }

    def redact(self, text: str) -> str:
        redacted = text
        with self._lock:
            secrets = [
                record.api_key
                for record in self._credentials.values()
                if record.api_key
            ]
        for secret in secrets:
            redacted = redacted.replace(secret, "[REDACTED]")
        return redacted

    def _validate_route(
        self,
        route: AgentModelRoute,
        label: str,
        warnings: list[str],
    ) -> None:
        with self._lock:
            record = self._credentials.get(route.credential_id)
        if record is None:
            raise ValueError(f"{label} 使用了不存在的凭证 {route.credential_id}")
        model_ids = {item.id for item in record.models}
        if model_ids and route.model not in model_ids:
            raise ValueError(
                f"{label} 的模型 {route.model} 不在凭证可用模型列表中"
            )
        if not record.supports_json_object:
            warnings.append(
                f"{label} 的模型将使用提示词 JSON 模式，结构化输出稳定性可能较低。"
            )

    def validate_base_url(self, value: str) -> str:
        raw = value.strip().rstrip("/")
        parsed = urlsplit(raw)
        if parsed.scheme not in {"https", "http"}:
            raise ValueError("Base URL 必须使用 https:// 或 http://")
        if not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Base URL 主机无效，且不能包含用户名或密码")
        if parsed.query or parsed.fragment:
            raise ValueError("Base URL 不能包含查询参数或片段")
        port = parsed.port
        host = parsed.hostname.casefold()
        local_names = {"localhost", "localhost.localdomain"}
        is_named_local = host in local_names or host.endswith(".local")
        addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
        try:
            addresses = {
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(
                    parsed.hostname,
                    port or (443 if parsed.scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            }
        except socket.gaierror as exc:
            raise ValueError("Base URL 域名无法解析") from exc
        private = is_named_local or any(
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
            for address in addresses
        )
        if private and not self.settings.allow_private_model_endpoints:
            raise ValueError(
                "私网/本机模型端点默认禁用；管理员可设置 "
                "REPOPILOT_ALLOW_PRIVATE_MODEL_ENDPOINTS=1"
            )
        if parsed.scheme != "https" and not private:
            raise ValueError("公网模型端点必须使用 HTTPS")
        netloc = parsed.hostname
        if ":" in netloc and not netloc.startswith("["):
            netloc = f"[{netloc}]"
        if port:
            netloc = f"{netloc}:{port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))

    @staticmethod
    def _discover_models(base_url: str, api_key: str) -> list[ProviderModelView]:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30.0,
            max_retries=0,
        )
        try:
            page = client.models.list()
            models = [
                ProviderModelView(
                    id=str(item.id),
                    owned_by=str(getattr(item, "owned_by", "") or ""),
                )
                for item in page.data[:200]
                if str(getattr(item, "id", "") or "").strip()
            ]
            return sorted(models, key=lambda item: item.id.casefold())
        finally:
            client.close()

    @staticmethod
    def _probe_model(base_url: str, api_key: str, model: str) -> None:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30.0,
            max_retries=0,
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Reply with OK to verify this model endpoint.",
                    }
                ],
                max_tokens=4,
            )
            if not response.choices:
                raise ValueError("Provider returned no choices")
        finally:
            client.close()

    @staticmethod
    def _secret_hint(secret: str) -> str:
        if len(secret) <= 8:
            return "••••••••"
        return f"{secret[:3]}••••{secret[-4:]}"

    @staticmethod
    def _safe_provider_error(exc: Exception) -> str:
        status = getattr(exc, "status_code", None)
        if status:
            return f"HTTP {status} / {type(exc).__name__}"
        return type(exc).__name__
