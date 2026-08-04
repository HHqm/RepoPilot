from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
import weakref
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from .agents import AgentSuite
from .config import Settings
from .models import (
    ProductUnderstanding,
    ProjectAsset,
    ProjectChatRequest,
    ProjectLaunchRequest,
    ProjectMessage,
    ProjectSessionCreate,
    ProjectSessionView,
)
from .prompts import PRODUCT_CONVERSATION_SYSTEM
from .rag.parsers import ALLOWED_SUFFIXES


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ProjectCollaborationManager:
    """Persistent project rooms for client discovery and document-grounded chat."""

    def __init__(self, settings: Settings, suite: AgentSuite):
        self.settings = settings
        self.suite = suite
        self.root = (settings.runtime_dir / "collaborations").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_lock = RLock()
        self._session_locks: weakref.WeakValueDictionary[str, RLock] = (
            weakref.WeakValueDictionary()
        )

    def create(self, request: ProjectSessionCreate) -> ProjectSessionView:
        session_id = uuid.uuid4().hex
        routing = None
        if request.model_routing is not None:
            if self.suite.model_registry is None:
                raise RuntimeError("模型供应商注册表不可用")
            routing = self.suite.model_registry.validate_routing(
                request.model_routing
            ).routing.model_dump()
        mode = "routed" if routing else (
            request.llm_mode or self.settings.llm_mode
        )
        if mode == "routed" and not routing:
            raise ValueError("llm_mode=routed requires model_routing")
        session = {
            "session_id": session_id,
            "status": "discovering",
            "project_name": request.project_name,
            "target_users": request.target_users,
            "budget_cny": request.budget_cny,
            "delivery_deadline": request.delivery_deadline,
            "policy_constraints": request.policy_constraints,
            "business_constraints": request.business_constraints,
            "preferred_stack": request.preferred_stack,
            "deployment_target": request.deployment_target,
            "deployment_conditions": request.deployment_conditions,
            "llm_mode": mode,
            "model_routing": routing,
            "messages": [
                self._message(
                    "product_manager",
                    "产品经理·言澈",
                    (
                        f"你好，我是「{request.project_name}」的产品经理。"
                        "请先告诉我业务为什么要做、谁会使用、最重要的流程和首期不做什么。"
                        "你也可以上传 PDF、Word、Excel、PPT、Markdown、文本或代码资料，"
                        "我会基于文件内容继续澄清。"
                    ),
                )
            ],
            "assets": [],
            "understanding": {
                "summary": "尚未收到客户的业务需求说明。",
                "confirmed_goals": [],
                "constraints": [
                    *request.policy_constraints,
                    *request.business_constraints,
                ],
                "open_questions": [
                    "要解决的核心业务问题是什么？",
                    "目标用户与首期核心流程是什么？",
                    "首期明确不做什么？",
                ],
                "assumptions": [],
                "ready_for_workshop": False,
            },
            "run_id": None,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        with self._session_lock(session_id):
            with self._cache_lock:
                self._sessions[session_id] = session
                self._trim_session_cache_locked()
            self._save_locked(session)
        return self._view(session)

    def get(self, session_id: str) -> ProjectSessionView:
        with self._session_lock(session_id):
            return self._view(self._load_locked(session_id))

    def chat(
        self,
        session_id: str,
        request: ProjectChatRequest,
    ) -> ProjectSessionView:
        with self._session_lock(session_id):
            session = self._load_locked(session_id)
            if session.get("run_id") or session.get("status") == "launching":
                raise RuntimeError("需求评审已启动，发现阶段不能再追加客户消息")
            session["messages"].append(
                self._message("client", "客户", request.content)
            )
            evidence = self._search_uploaded_locked(session, request.content)
            result, _ = self.suite._complete_json(
                {
                    "llm_mode": session["llm_mode"],
                    "model_routing": session.get("model_routing"),
                },
                system_prompt=PRODUCT_CONVERSATION_SYSTEM,
                schema_name="product_chat",
                payload={
                    "project_name": session["project_name"],
                    "conversation_memory": self._conversation_memory(session),
                    "assets": session["assets"],
                    "policy_constraints": session["policy_constraints"],
                    "business_constraints": session["business_constraints"],
                    "evidence": evidence,
                },
            )
            understanding = ProductUnderstanding.model_validate(
                result.get("understanding") or {}
            )
            understanding, gate_note = self._normalize_discovery_gate(
                session,
                understanding,
            )
            session["understanding"] = understanding.model_dump()
            if gate_note:
                result["reply"] = (
                    f"{str(result.get('reply') or '').strip()}\n\n{gate_note}"
                ).strip()
            session["messages"].append(
                self._message(
                    "product_manager",
                    "产品经理·言澈",
                    str(result.get("reply") or "我已经更新需求理解。"),
                    [
                        str(item.get("evidence_id"))
                        for item in evidence
                        if item.get("evidence_id")
                    ],
                )
            )
            session["status"] = (
                "ready_for_workshop"
                if understanding.ready_for_workshop
                else "discovering"
            )
            session["updated_at"] = _utc_now()
            self._save_locked(session)
            return self._view(session)

    @staticmethod
    def _normalize_discovery_gate(
        session: dict[str, Any],
        understanding: ProductUnderstanding,
    ) -> tuple[ProductUnderstanding, str]:
        """Bound discovery and repair contradictory model readiness signals."""

        client_messages = [
            str(item.get("content") or "")
            for item in session["messages"]
            if item.get("role") == "client"
        ]
        combined_client_text = "\n".join(client_messages)
        target_users = str(session.get("target_users") or "").strip()
        deadline = str(session.get("delivery_deadline") or "").strip()
        core_dimensions_explicit = all(
            (
                len(combined_client_text) >= 160,
                len(understanding.summary.strip()) >= 80,
                bool(understanding.confirmed_goals),
                bool(understanding.constraints),
                bool(target_users) and "待" not in target_users,
                bool(deadline) and "待" not in deadline,
                session.get("deployment_target") != "undecided",
            )
        )
        questions = list(understanding.open_questions)
        assumptions = list(understanding.assumptions)
        model_ready = understanding.ready_for_workshop
        note = ""

        if questions:
            ready = False
            if len(client_messages) >= 3 and core_dimensions_explicit:
                assumptions.extend(
                    f"需求评审会待确认：{question}" for question in questions
                )
                questions = []
                ready = True
                note = (
                    "需求发现已达到三轮且核心业务边界完整；剩余实现偏好已转为"
                    "评审会议题，不再无限追加追问。现在可以召集跨职能需求评审会。"
                )
        else:
            ready = bool(model_ready or core_dimensions_explicit)
            if ready and not model_ready:
                note = (
                    "核心目标、用户、范围、约束和交付条件已经完整，准备度门禁"
                    "已完成一致性校验，可以召集跨职能需求评审会。"
                )

        return (
            understanding.model_copy(
                update={
                    "open_questions": questions,
                    "assumptions": list(dict.fromkeys(assumptions)),
                    "ready_for_workshop": ready,
                }
            ),
            note,
        )

    def add_document(
        self,
        session_id: str,
        *,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> ProjectSessionView:
        with self._session_lock(session_id):
            session = self._load_locked(session_id)
            if session.get("run_id") or session.get("status") == "launching":
                raise RuntimeError("需求评审已启动，当前项目资料已冻结")
            original = Path(filename.replace("\\", "/")).name.strip()
            if (
                not original
                or original in {".", ".."}
                or any(ord(character) < 32 for character in original)
            ):
                raise ValueError("文件名不合法")
            suffix = Path(original).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                supported = ", ".join(sorted(ALLOWED_SUFFIXES))
                raise ValueError(f"不支持 {suffix or '无扩展名'}；支持：{supported}")
            if not content:
                raise ValueError("上传文件不能为空")
            if len(content) > self.settings.rag_max_document_bytes:
                raise ValueError(
                    f"文件超过 {self.settings.rag_max_document_bytes} 字节限制"
                )

            knowledge_root = self._knowledge_root(session_id)
            client_root = knowledge_root / "client"
            client_root.mkdir(parents=True, exist_ok=True)
            safe_stem = re.sub(
                r"[^a-zA-Z0-9._-]+",
                "-",
                Path(original).stem,
            ).strip("-")[:80] or "document"
            stored_name = f"{uuid.uuid4().hex[:10]}-{safe_stem}{suffix}"
            target = (client_root / stored_name).resolve()
            target.relative_to(client_root.resolve())
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{stored_name}.",
                suffix=".upload.tmp",
                dir=client_root,
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(content)
                os.replace(temporary_name, target)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)

            relative = target.relative_to(knowledge_root).as_posix()
            try:
                index = self.suite.rag.index_repository(knowledge_root)
                indexed = index.files.get(relative)
                if indexed is None:
                    reason = index.skipped_files.get(relative, "解析器没有生成内容")
                    raise ValueError(f"文件无法进入知识库：{reason}")
            except Exception:
                target.unlink(missing_ok=True)
                raise

            asset = ProjectAsset(
                asset_id=uuid.uuid4().hex,
                original_name=original,
                stored_name=relative,
                media_type=media_type or "application/octet-stream",
                size=len(content),
                parser=indexed.parser,
                chunk_count=indexed.chunk_count,
                redaction_count=indexed.redaction_count,
                uploaded_at=_utc_now(),
            ).model_dump()
            session["assets"].append(asset)
            session["messages"].append(
                self._message(
                    "system",
                    "项目资料库",
                    (
                        f"已解析并索引「{original}」：{indexed.chunk_count} 个知识块，"
                        f"解析器 {indexed.parser}，脱敏命中 {indexed.redaction_count} 处。"
                    ),
                )
            )
            session["updated_at"] = _utc_now()
            self._save_locked(session)
            return self._view(session)

    def reserve_launch(
        self, session_id: str
    ) -> tuple[ProjectLaunchRequest, str]:
        """Atomically freeze discovery so concurrent launch requests cannot race."""

        with self._session_lock(session_id):
            session = self._load_locked(session_id)
            if session.get("run_id"):
                raise RuntimeError("该项目已经进入需求评审会")
            if session.get("status") == "launching":
                raise RuntimeError("该项目正在启动需求评审会，请勿重复提交")
            client_messages = [
                item["content"]
                for item in session["messages"]
                if item["role"] == "client"
            ]
            if not client_messages:
                raise RuntimeError("请先与产品经理说明业务需求")
            if not session["understanding"].get("ready_for_workshop"):
                raise RuntimeError("产品经理仍有关键问题，请完成需求澄清后再开评审会")
            understanding = ProductUnderstanding.model_validate(
                session["understanding"]
            )
            memory = {
                "open_questions": understanding.open_questions,
                "assumptions": understanding.assumptions,
                "recent_client_messages": client_messages[-3:],
            }
            summary = json.dumps(memory, ensure_ascii=False, indent=2)
            client_brief = (
                f"{understanding.summary}\n\n"
                + "\n".join(
                    f"- 已确认目标：{item}"
                    for item in understanding.confirmed_goals
                )
                + "\n"
                + "\n".join(
                    f"- 已确认约束：{item}"
                    for item in understanding.constraints
                )
            )
            launch_token = uuid.uuid4().hex
            session["status"] = "launching"
            session["launch_token"] = launch_token
            session["updated_at"] = _utc_now()
            self._save_locked(session)
            return ProjectLaunchRequest(
                project_name=session["project_name"],
                client_brief=client_brief,
                target_users=session["target_users"],
                budget_cny=session["budget_cny"],
                delivery_deadline=session["delivery_deadline"],
                policy_constraints=session["policy_constraints"],
                business_constraints=session["business_constraints"],
                preferred_stack=session["preferred_stack"],
                deployment_target=session["deployment_target"],
                deployment_conditions=session["deployment_conditions"],
                llm_mode=session["llm_mode"],
                model_routing=session.get("model_routing"),
                collaboration_id=session_id,
                conversation_summary=summary,
                uploaded_documents=session["assets"],
            ), launch_token

    def mark_launched(
        self, session_id: str, launch_token: str, run_id: str
    ) -> ProjectSessionView:
        with self._session_lock(session_id):
            session = self._load_locked(session_id)
            if session.get("launch_token") != launch_token:
                raise RuntimeError("项目启动令牌已失效，拒绝关联未知运行")
            session["run_id"] = run_id
            session["status"] = "workshop_running"
            session.pop("launch_token", None)
            session["updated_at"] = _utc_now()
            self._save_locked(session)
            return self._view(session)

    def abort_launch(self, session_id: str, launch_token: str) -> None:
        with self._session_lock(session_id):
            session = self._load_locked(session_id)
            if session.get("launch_token") != launch_token or session.get("run_id"):
                return
            session["status"] = (
                "ready_for_workshop"
                if session["understanding"].get("ready_for_workshop")
                else "discovering"
            )
            session.pop("launch_token", None)
            session["updated_at"] = _utc_now()
            self._save_locked(session)

    def knowledge_root(self, session_id: str) -> Path:
        """Return the canonical server-owned knowledge root for a real session."""

        with self._session_lock(session_id):
            self._load_locked(session_id)
            return self._knowledge_root(session_id)

    def _search_uploaded_locked(
        self,
        session: dict[str, Any],
        query: str,
    ) -> list[dict[str, Any]]:
        if not session["assets"]:
            return []
        root = self._knowledge_root(session["session_id"])
        index = self.suite.rag.index_repository(root)
        return [
            item.model_dump()
            for item in self.suite.rag.search(
                index,
                query,
                top_k=min(6, self.settings.rag_top_k),
            )
        ]

    def _knowledge_root(self, session_id: str) -> Path:
        return (self._session_root(session_id) / "knowledge").resolve()

    def _session_lock(self, session_id: str) -> RLock:
        # Validate before storing an attacker-controlled key in the lock cache.
        self._session_root(session_id)
        with self._cache_lock:
            return self._session_locks.setdefault(session_id, RLock())

    def _session_root(self, session_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", session_id):
            raise KeyError(f"Unknown project session: {session_id}")
        path = (self.root / session_id).resolve()
        path.relative_to(self.root)
        return path

    def _load_locked(self, session_id: str) -> dict[str, Any]:
        with self._cache_lock:
            cached = self._sessions.get(session_id)
            if cached is not None:
                self._sessions.move_to_end(session_id)
        if cached is not None:
            return cached
        path = self._session_root(session_id) / "session.json"
        if not path.is_file():
            raise KeyError(f"Unknown project session: {session_id}")
        session = json.loads(path.read_text(encoding="utf-8"))
        with self._cache_lock:
            cached = self._sessions.setdefault(session_id, session)
            self._sessions.move_to_end(session_id)
            self._trim_session_cache_locked()
            return cached

    def _save_locked(self, session: dict[str, Any]) -> None:
        root = self._session_root(session["session_id"])
        root.mkdir(parents=True, exist_ok=True)
        target = root / "session.json"
        temporary = root / "session.json.tmp"
        temporary.write_text(
            json.dumps(session, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, target)

    def _trim_session_cache_locked(self) -> None:
        while len(self._sessions) > self.settings.max_cached_sessions:
            self._sessions.popitem(last=False)

    @staticmethod
    def _conversation_memory(session: dict[str, Any]) -> dict[str, Any]:
        """Select a bounded recent window while retaining structured decisions."""

        selected: list[dict[str, Any]] = []
        remaining_chars = 24_000
        for item in reversed(session["messages"]):
            content = str(item.get("content", ""))
            if not content:
                continue
            allowed = min(6_000, remaining_chars)
            if allowed <= 0:
                break
            if len(content) > allowed:
                content = content[:allowed] + "\n…[较早内容已压缩]"
            selected.append(
                {
                    "message_id": item.get("message_id"),
                    "role": item.get("role"),
                    "author": item.get("author"),
                    "content": content,
                    "evidence_ids": item.get("evidence_ids", []),
                }
            )
            remaining_chars -= len(content)
            if len(selected) >= 8 or remaining_chars <= 0:
                break
        selected.reverse()
        return {
            "rolling_understanding": session["understanding"],
            "recent_messages": selected,
            "selection_policy": "structured understanding + latest 8 messages / 24k chars",
        }

    @staticmethod
    def _message(
        role: str,
        author: str,
        content: str,
        evidence_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return ProjectMessage(
            message_id=uuid.uuid4().hex,
            role=role,
            author=author,
            content=content,
            created_at=_utc_now(),
            evidence_ids=evidence_ids or [],
        ).model_dump()

    @staticmethod
    def _view(session: dict[str, Any]) -> ProjectSessionView:
        return ProjectSessionView.model_validate(session)
