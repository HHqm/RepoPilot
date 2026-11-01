from __future__ import annotations

import bisect
import copy
import json
import re
import sqlite3
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import BoundedSemaphore, Condition, Lock, RLock
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .agents import AgentSuite
from .collaboration import ProjectCollaborationManager
from .config import Settings
from .model_routing import ModelProviderRegistry
from .models import (
    ApprovalDecision,
    IssueRequest,
    ProjectLaunchRequest,
    ProjectRequest,
    ProjectSessionView,
    RunCreated,
)
from .state import STATE_SCHEMA_VERSION
from .studio_agents import SoftwareStudioAgentSuite
from .studio_workflow import build_studio_workflow
from .workflow import build_workflow


TERMINAL_STATUSES = {
    "completed",
    "cancelled",
    "review_rejected",
    "release_rejected",
    "delivery_failed",
    "failed_rolled_back",
    "failed",
}
HUMAN_CHECKPOINT_STATUSES = {
    "awaiting_approval",
    "awaiting_requirements_approval",
    "awaiting_release_approval",
}
WAITABLE_STATUSES = TERMINAL_STATUSES | HUMAN_CHECKPOINT_STATUSES


@dataclass
class RunRuntime:
    run_id: str
    status: str
    state: dict[str, Any]
    interrupt: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    last_sequence: int = 0
    closed: bool = False
    condition: Condition = field(default_factory=Condition)
    operation_lock: Lock = field(default_factory=Lock)
    future: Future | None = None


class RepoPilotService:
    """Thread-safe local runtime with incremental events and SQLite checkpoints.

    Each graph execution owns its SQLite connection. WAL mode and busy_timeout
    protect concurrent readers/writers without sharing one connection across
    FastAPI worker threads. The append-only event log is the exception: it
    uses one long-lived, lock-guarded connection because every agent trace is
    a tiny write and reconnecting per event dominated publish latency.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.settings.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.database = self.settings.runtime_dir / "checkpoints.sqlite3"
        self.model_registry = ModelProviderRegistry(self.settings)
        self.suite = AgentSuite(self.settings, self.model_registry)
        self.collaboration = ProjectCollaborationManager(
            self.settings,
            self.suite,
        )
        self.studio_suite = SoftwareStudioAgentSuite(self.suite)
        self._runtimes: dict[str, RunRuntime] = {}
        self._runtimes_lock = RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=self.settings.max_workers,
            thread_name_prefix="repopilot-run",
        )
        self._run_slots = BoundedSemaphore(self.settings.max_pending_runs)
        # Compiled graphs are immutable templates; each session copies one and
        # binds its own checkpointer, which is ~80x cheaper than recompiling.
        # Templates are built lazily so agent overrides made before the first
        # run are still honoured.
        self._graph_templates: dict[str, Any] = {}
        self._graph_templates_lock = Lock()
        self._closed = False
        self._setup_database()
        self._event_db = self._connect()
        self._event_db.execute("PRAGMA synchronous=NORMAL")
        self._event_db_lock = Lock()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        with self._event_db_lock:
            self._event_db.close()
        self.suite.close()

    def __enter__(self) -> RepoPilotService:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def start(self, request: IssueRequest) -> RunCreated:
        """Accept a run immediately and execute the graph in the background."""

        self._ensure_open()
        repo_path = Path(request.repo_path).resolve()
        if self.settings.allowed_workspace_roots:
            allowed = False
            for workspace_root in self.settings.allowed_workspace_roots:
                try:
                    repo_path.relative_to(workspace_root.resolve())
                    allowed = True
                    break
                except ValueError:
                    continue
            if not allowed:
                raise PermissionError(
                    "Repository is outside REPOPILOT_WORKSPACE_ROOTS"
                )
        run_id = uuid.uuid4().hex
        mode, routing = self._resolve_llm_mode(request)
        initial: dict[str, Any] = {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "run_id": run_id,
            "repo_path": str(repo_path),
            "issue": request.issue,
            "constraints": request.constraints,
            "test_command": request.test_command,
            "llm_mode": mode,
            "model_routing": routing,
            "status": "queued",
            "approved": False,
            "traces": [],
        }
        return self._launch_runtime(
            initial,
            "Run accepted and queued for background execution.",
        )

    def start_project(
        self, request: ProjectRequest | ProjectLaunchRequest
    ) -> RunCreated:
        """Start a client-brief-to-deployable-project studio workflow."""

        self._ensure_open()
        run_id = uuid.uuid4().hex
        mode, routing = self._resolve_llm_mode(request)
        collaboration_id = (
            request.collaboration_id
            if isinstance(request, ProjectLaunchRequest)
            else ""
        )
        conversation_summary = (
            request.conversation_summary
            if isinstance(request, ProjectLaunchRequest)
            else ""
        )
        uploaded_documents = (
            request.uploaded_documents
            if isinstance(request, ProjectLaunchRequest)
            else []
        )
        project_knowledge_path = (
            self.collaboration.knowledge_root(collaboration_id)
            if collaboration_id
            else (
                self.settings.runtime_dir
                / "collaborations"
                / run_id
                / "knowledge"
            ).resolve()
        )
        safe_name = (
            re.sub(
                r"[^a-z0-9]+",
                "-",
                request.project_name.casefold(),
            ).strip("-")[:60]
            or "software-project"
        )
        project_root = (
            self.settings.runtime_dir
            / "projects"
            / f"{safe_name}-{run_id[:8]}"
        ).resolve()
        initial: dict[str, Any] = {
            "state_schema_version": STATE_SCHEMA_VERSION,
            "run_id": run_id,
            "workflow_kind": "project_factory",
            "repo_path": str(project_root),
            "project_output_path": str(project_root),
            "project_name": request.project_name,
            "collaboration_id": collaboration_id,
            "conversation_summary": conversation_summary,
            "client_messages": [],
            "uploaded_documents": uploaded_documents,
            "project_knowledge_path": str(project_knowledge_path),
            "client_brief": request.client_brief,
            "target_users": request.target_users,
            "budget_cny": request.budget_cny,
            "delivery_deadline": request.delivery_deadline,
            "policy_constraints": request.policy_constraints,
            "business_constraints": request.business_constraints,
            "preferred_stack": request.preferred_stack,
            "deployment_target": request.deployment_target,
            "deployment_conditions": request.deployment_conditions,
            "requirements_revision": 0,
            "development_revision": 0,
            "llm_mode": mode,
            "model_routing": routing,
            "status": "queued",
            "traces": [],
        }
        return self._launch_runtime(
            initial,
            "Client brief accepted; the software company workflow is queued.",
        )

    def _launch_runtime(
        self, initial: dict[str, Any], summary: str
    ) -> RunCreated:
        run_id = initial["run_id"]
        runtime = RunRuntime(run_id=run_id, status="queued", state=initial)
        with self._runtimes_lock:
            self._runtimes[run_id] = runtime
        self._publish(
            runtime,
            "run_status",
            {"status": "queued", "summary": summary},
        )
        try:
            runtime.future = self._submit_graph(runtime, initial, False)
        except Exception:
            with self._runtimes_lock:
                self._runtimes.pop(run_id, None)
            raise
        return self._runtime_response(runtime)

    def _resolve_llm_mode(self, request) -> tuple[str, dict[str, Any] | None]:
        routing = self._validated_routing(request.model_routing)
        mode = "routed" if routing else (
            request.llm_mode or self.settings.llm_mode
        )
        if mode == "routed" and not routing:
            raise ValueError("llm_mode=routed requires model_routing")
        return mode, routing

    def launch_collaboration(
        self,
        session_id: str,
    ) -> tuple[RunCreated, ProjectSessionView]:
        """Freeze a discovery room and start its cross-functional workshop."""

        request, launch_token = self.collaboration.reserve_launch(session_id)
        try:
            run = self.start_project(request)
            session = self.collaboration.mark_launched(
                session_id,
                launch_token,
                run.run_id,
            )
            return run, session
        except Exception:
            self.collaboration.abort_launch(session_id, launch_token)
            raise

    def _validated_routing(self, routing) -> dict[str, Any] | None:
        if routing is None:
            return None
        validation = self.model_registry.validate_routing(routing)
        return validation.routing.model_dump()

    def decide(self, run_id: str, decision: ApprovalDecision) -> RunCreated:
        """Queue a human decision and resume without blocking the HTTP request."""

        self._ensure_open()
        checkpoint_response, next_nodes = self._checkpoint_response(run_id)
        if not checkpoint_response.state:
            raise KeyError(f"Unknown run_id: {run_id}")
        checkpoint_nodes = {
            "approval",
            "requirements_approval",
            "release_approval",
        }
        active_checkpoint = next(
            (node for node in next_nodes if node in checkpoint_nodes),
            None,
        )
        if active_checkpoint is None:
            raise RuntimeError(
                "Run is not awaiting approval; current status is "
                f"{checkpoint_response.status}"
            )
        expected_digest = str(
            (checkpoint_response.interrupt or {}).get("artifact_digest") or ""
        )
        if expected_digest and decision.artifact_digest != expected_digest:
            raise RuntimeError(
                "Approval artifact digest is missing or stale; refresh the "
                "checkpoint before deciding"
            )
        runtime = self._ensure_runtime(checkpoint_response)
        with runtime.condition:
            if runtime.status == "resuming":
                raise RuntimeError("A decision is already resuming this run")
            runtime.status = "resuming"
            runtime.state["status"] = "resuming"
            runtime.interrupt = None
        self._publish(
            runtime,
            "human_decision",
            {
                "approved": decision.approved,
                "comment": decision.comment,
                "actor_id": decision.actor_id,
                "artifact_digest": decision.artifact_digest,
                "checkpoint": active_checkpoint,
            },
        )
        self._publish(
            runtime,
            "run_status",
            {
                "status": "resuming",
                "summary": "Human decision accepted; workflow is resuming.",
            },
        )
        try:
            runtime.future = self._submit_graph(
                runtime,
                Command(resume=decision.model_dump()),
                True,
            )
        except Exception:
            with runtime.condition:
                runtime.status = checkpoint_response.status
                runtime.state = copy.deepcopy(checkpoint_response.state)
                runtime.interrupt = copy.deepcopy(
                    checkpoint_response.interrupt
                )
            raise
        return self._runtime_response(runtime)

    def get(self, run_id: str) -> RunCreated:
        runtime = self._get_runtime(run_id)
        if runtime is not None:
            return self._runtime_response(runtime)
        response, _ = self._checkpoint_response(run_id)
        if response.state:
            return response
        artifact = self._artifact_response(run_id)
        if artifact is not None:
            return artifact
        raise KeyError(f"Unknown run_id: {run_id}")

    def wait(
        self,
        run_id: str,
        statuses: set[str] | None = None,
        timeout_seconds: float = 30,
    ) -> RunCreated:
        """Wait for CLI/tests while keeping the API itself non-blocking."""

        targets = statuses or WAITABLE_STATUSES
        deadline = time.monotonic() + timeout_seconds
        while True:
            response = self.get(run_id)
            ready_for_approval = not (
                response.status in HUMAN_CHECKPOINT_STATUSES
                and response.interrupt is None
            )
            runtime = self._get_runtime(run_id)
            terminal_ready = not (
                response.status in TERMINAL_STATUSES
                and runtime is not None
                and not runtime.closed
            )
            if response.status in targets and ready_for_approval and terminal_ready:
                return response
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Run {run_id} did not reach {sorted(targets)} in time"
                )
            if runtime is None:
                time.sleep(min(0.05, remaining))
                continue
            with runtime.condition:
                runtime.condition.wait(timeout=min(0.25, remaining))

    def iter_events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        heartbeat_seconds: float = 10,
    ) -> Iterator[dict[str, Any] | None]:
        """Yield replayable events and then block for genuinely new events.

        ``None`` represents an SSE heartbeat. The stream stays open while a run
        waits for human approval and continues when ``decide`` resumes it.
        """

        response = self.get(run_id)
        runtime = self._ensure_runtime(response)
        cursor = max(0, after_sequence)
        while True:
            with runtime.condition:
                pending = self._events_after(runtime, cursor)
                if not pending and not runtime.closed:
                    runtime.condition.wait(timeout=heartbeat_seconds)
                    pending = self._events_after(runtime, cursor)
                closed = runtime.closed
            if pending:
                for event in pending:
                    cursor = event["sequence"]
                    yield event
                continue
            if closed:
                return
            yield None

    @staticmethod
    def _events_after(
        runtime: RunRuntime, cursor: int
    ) -> list[dict[str, Any]]:
        # Events are appended in strictly increasing sequence order, so a
        # reconnecting SSE client only pays for the tail it has not seen.
        start = bisect.bisect_right(
            runtime.events,
            cursor,
            key=lambda event: event["sequence"],
        )
        return [copy.deepcopy(event) for event in runtime.events[start:]]

    def _run_graph(
        self,
        runtime: RunRuntime,
        graph_input: dict[str, Any] | Command,
        is_resume: bool,
    ) -> None:
        try:
            with runtime.operation_lock:
                try:
                    if not is_resume:
                        self._publish(
                            runtime,
                            "run_status",
                            {
                                "status": "running",
                                "summary": "Background graph execution started.",
                            },
                        )
                        with runtime.condition:
                            runtime.status = "running"
                            runtime.state["status"] = "running"
                    workflow_kind = str(
                        runtime.state.get("workflow_kind", "maintenance")
                    )
                    with self._graph_session(workflow_kind) as graph:
                        for chunk in graph.stream(
                            graph_input,
                            config=self._config(runtime.run_id),
                            stream_mode="updates",
                        ):
                            self._consume_chunk(runtime, chunk)
                        snapshot = graph.get_state(self._config(runtime.run_id))
                        response = self._response_from_snapshot(
                            runtime.run_id, snapshot
                        )
                    if not snapshot.next and response.status not in TERMINAL_STATUSES:
                        # A finished graph must end in a terminal status, or SSE
                        # streams and waiters would block on it forever.
                        raise RuntimeError(
                            "Workflow ended with non-terminal status "
                            f"{response.status!r}"
                        )
                    self._replace_runtime_state(runtime, response)
                    self._publish(
                        runtime,
                        "run_status",
                        {
                            "status": response.status,
                            "summary": (
                                "Checkpoint reached; human decision required."
                                if response.status in HUMAN_CHECKPOINT_STATUSES
                                else "Checkpoint committed."
                            ),
                            "checkpointed": True,
                        },
                    )
                    self._persist_artifact(response)
                    if response.status in TERMINAL_STATUSES:
                        self._close_runtime(runtime)
                except Exception as exc:
                    self._handle_framework_failure(runtime, exc)
        finally:
            self._run_slots.release()

    def _submit_graph(
        self,
        runtime: RunRuntime,
        graph_input: dict[str, Any] | Command,
        is_resume: bool,
    ) -> Future:
        if not self._run_slots.acquire(blocking=False):
            raise RuntimeError(
                "Run queue is full; retry after active work reaches a checkpoint"
            )
        try:
            return self._executor.submit(
                self._run_graph,
                runtime,
                graph_input,
                is_resume,
            )
        except Exception:
            self._run_slots.release()
            raise

    def _consume_chunk(self, runtime: RunRuntime, chunk: Any) -> None:
        if not isinstance(chunk, dict):
            return
        for node_name, update in chunk.items():
            if node_name == "__interrupt__" or not isinstance(update, dict):
                continue
            traces = list(update.get("traces") or [])
            with runtime.condition:
                for key, value in update.items():
                    if key == "traces":
                        runtime.state.setdefault("traces", []).extend(
                            copy.deepcopy(value)
                        )
                    else:
                        runtime.state[key] = copy.deepcopy(value)
                if "status" in update:
                    runtime.status = str(update["status"])
                runtime.condition.notify_all()
            for trace in traces:
                self._publish(
                    runtime,
                    "agent_trace",
                    {"node": node_name, **copy.deepcopy(trace)},
                )
            if update.get("error"):
                self._publish(
                    runtime,
                    "run_error",
                    copy.deepcopy(update["error"]),
                )
            if "status" in update:
                summary = traces[-1].get("summary") if traces else None
                self._publish(
                    runtime,
                    "run_status",
                    {
                        "status": update["status"],
                        "node": node_name,
                        "summary": summary,
                    },
                )

    def _replace_runtime_state(
        self, runtime: RunRuntime, response: RunCreated
    ) -> None:
        with runtime.condition:
            runtime.status = response.status
            runtime.state = copy.deepcopy(response.state)
            runtime.interrupt = copy.deepcopy(response.interrupt)
            runtime.closed = response.status in TERMINAL_STATUSES
            runtime.condition.notify_all()

    def _handle_framework_failure(
        self, runtime: RunRuntime, exc: Exception
    ) -> None:
        message = str(exc).strip() or exc.__class__.__name__
        message = self.suite.redact_secrets(message)
        rollback: dict[str, Any] | None = None
        state = copy.deepcopy(runtime.state)
        changed_files = state.get("apply_result", {}).get("changed_files", [])
        already_rolled_back = (
            state.get("verification", {}).get("rollback")
            or state.get("quality_gate", {}).get("rollback")
        )
        verification_succeeded = bool(
            state.get("verification", {}).get("success")
            or state.get("quality_gate", {}).get("success")
        )
        if changed_files and not already_rolled_back and not verification_succeeded:
            try:
                if state.get("workflow_kind") == "project_factory":
                    rollback = self.studio_suite.writer.rollback(
                        state["repo_path"],
                        changed_files,
                    )
                else:
                    rollback = self.suite.applier.rollback(
                        state["repo_path"],
                        runtime.run_id,
                        changed_files,
                        expected_applied_sha256=state.get(
                            "apply_result", {}
                        ).get("applied_sha256"),
                    )
            except Exception as rollback_error:
                rollback = {
                    "success": False,
                    "error": self.suite.redact_secrets(
                        str(rollback_error)
                    )[:1000],
                }
        error = {
            "agent": "WorkflowRuntime",
            "type": exc.__class__.__name__,
            "message": message[:1000],
        }
        if rollback is not None:
            error["rollback"] = rollback
        trace = {
            "agent": "FailureReporterAgent",
            "status": "failed",
            "summary": f"Runtime failed safely: {error['message']}",
            "latency_ms": 0,
            "metadata": {
                "failed_agent": "WorkflowRuntime",
                "rollback": rollback,
            },
        }
        with runtime.condition:
            runtime.status = "failed"
            runtime.state["status"] = "failed"
            runtime.state["error"] = error
            runtime.state.setdefault("traces", []).append(trace)
            runtime.state["report"] = (
                f"# RepoPilot Run {runtime.run_id} — Failed safely\n\n"
                f"- Error type: `{error['type']}`\n"
                f"- Message: {error['message']}\n"
            )
            runtime.interrupt = None
        self._publish(runtime, "agent_trace", trace)
        self._publish(runtime, "run_error", error)
        self._publish(
            runtime,
            "run_status",
            {
                "status": "failed",
                "summary": "Workflow runtime ended through the failure boundary.",
            },
        )
        response = self._runtime_response(runtime)
        self._persist_artifact(response)
        self._close_runtime(runtime)

    def _publish(
        self,
        runtime: RunRuntime,
        event_type: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        with runtime.condition:
            occurred_at = datetime.now(UTC).isoformat()
            next_memory_sequence = runtime.last_sequence + 1
            try:
                sequence = self._append_durable_event(
                    runtime.run_id,
                    event_type,
                    data,
                    occurred_at,
                    minimum_sequence=next_memory_sequence,
                )
            except sqlite3.Error:
                sequence = next_memory_sequence
            event = {
                "sequence": sequence,
                "type": event_type,
                "occurred_at": occurred_at,
                "data": copy.deepcopy(data),
            }
            runtime.events.append(event)
            runtime.last_sequence = sequence
            runtime.condition.notify_all()
            return copy.deepcopy(event)

    def _close_runtime(self, runtime: RunRuntime) -> None:
        with runtime.condition:
            runtime.closed = True
            runtime.condition.notify_all()
        with self._runtimes_lock:
            if self._runtimes.get(runtime.run_id) is runtime:
                self._runtimes.pop(runtime.run_id, None)

    def _runtime_response(self, runtime: RunRuntime) -> RunCreated:
        with runtime.condition:
            return RunCreated(
                run_id=runtime.run_id,
                status=runtime.status,
                state=copy.deepcopy(runtime.state),
                interrupt=copy.deepcopy(runtime.interrupt),
            )

    def _ensure_runtime(self, response: RunCreated) -> RunRuntime:
        existing = self._get_runtime(response.run_id)
        if existing is not None:
            return existing
        runtime = RunRuntime(
            run_id=response.run_id,
            status=response.status,
            state=copy.deepcopy(response.state),
            interrupt=copy.deepcopy(response.interrupt),
            closed=response.status in TERMINAL_STATUSES,
        )
        runtime.events = self._load_durable_events(response.run_id)
        if not runtime.events:
            for trace in response.state.get("traces", []):
                runtime.events.append(
                    {
                        "sequence": len(runtime.events) + 1,
                        "type": "agent_trace",
                        "occurred_at": datetime.now(UTC).isoformat(),
                        "data": copy.deepcopy(trace),
                    }
                )
            runtime.events.append(
                {
                    "sequence": len(runtime.events) + 1,
                    "type": "run_status",
                    "occurred_at": datetime.now(UTC).isoformat(),
                    "data": {
                        "status": response.status,
                        "summary": "Recovered from a legacy checkpoint.",
                        "checkpointed": True,
                    },
                }
            )
        runtime.last_sequence = runtime.events[-1]["sequence"]
        with self._runtimes_lock:
            return self._runtimes.setdefault(response.run_id, runtime)

    def _get_runtime(self, run_id: str) -> RunRuntime | None:
        with self._runtimes_lock:
            return self._runtimes.get(run_id)

    def _checkpoint_response(
        self, run_id: str
    ) -> tuple[RunCreated, tuple[str, ...]]:
        with self._graph_session("maintenance") as graph:
            snapshot = graph.get_state(self._config(run_id))
            values = dict(snapshot.values or {})
            if not values:
                return (
                    RunCreated(
                        run_id=run_id,
                        status="unknown",
                        state={},
                        interrupt=None,
                    ),
                    (),
                )
        workflow_kind = str(values.get("workflow_kind", "maintenance"))
        with self._graph_session(workflow_kind) as graph:
            snapshot = graph.get_state(self._config(run_id))
            response = self._response_from_snapshot(run_id, snapshot)
            return response, tuple(snapshot.next or ())

    @staticmethod
    def _response_from_snapshot(run_id: str, snapshot) -> RunCreated:
        values = dict(snapshot.values or {})
        interrupt_payload = None
        interrupts = getattr(snapshot, "interrupts", ()) or ()
        if interrupts:
            interrupt_payload = interrupts[0].value
        return RunCreated(
            run_id=run_id,
            status=values.get("status", "unknown"),
            state=values,
            interrupt=interrupt_payload,
        )

    def _artifact_response(self, run_id: str) -> RunCreated | None:
        state_file = self.settings.runtime_dir / "runs" / run_id / "state.json"
        if not state_file.is_file():
            return None
        try:
            return RunCreated.model_validate_json(
                state_file.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None

    def _persist_artifact(self, response: RunCreated) -> None:
        output_dir = self.settings.runtime_dir / "runs" / response.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        state_file = output_dir / "state.json"
        temporary = output_dir / "state.json.tmp"
        temporary.write_text(
            json.dumps(response.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(state_file)
        report = response.state.get("report")
        if report:
            report_file = output_dir / "report.md"
            report_temporary = output_dir / "report.md.tmp"
            report_temporary.write_text(report, encoding="utf-8")
            report_temporary.replace(report_file)

    def _setup_database(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            SqliteSaver(connection).setup()
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_events (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_run_events_run_sequence
                ON run_events (run_id, sequence)
                """
            )
            connection.commit()
        finally:
            connection.close()

    def _append_durable_event(
        self,
        run_id: str,
        event_type: str,
        data: dict[str, Any],
        occurred_at: str,
        *,
        minimum_sequence: int = 1,
    ) -> int:
        with self._event_db_lock:
            connection = self._event_db
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) + 1
                    FROM run_events
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
                sequence = max(int(row[0]), minimum_sequence)
                connection.execute(
                    """
                    INSERT INTO run_events
                        (run_id, sequence, event_type, occurred_at, data_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        sequence,
                        event_type,
                        occurred_at,
                        json.dumps(data, ensure_ascii=False),
                    ),
                )
            except BaseException:
                connection.rollback()
                raise
            connection.commit()
            return sequence

    def _load_durable_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._event_db_lock:
            rows = self._event_db.execute(
                """
                SELECT sequence, event_type, occurred_at, data_json
                FROM run_events
                WHERE run_id = ?
                ORDER BY sequence
                """,
                (run_id,),
            ).fetchall()
        result = []
        for sequence, event_type, occurred_at, data_json in rows:
            try:
                data = json.loads(data_json)
            except (TypeError, ValueError):
                data = {"error": "event payload could not be decoded"}
            result.append(
                {
                    "sequence": int(sequence),
                    "type": str(event_type),
                    "occurred_at": str(occurred_at),
                    "data": data,
                }
            )
        return result

    @contextmanager
    def _graph_session(self, workflow_kind: str = "maintenance"):
        template = self._graph_template(workflow_kind)
        connection = self._connect()
        try:
            yield template.copy({"checkpointer": SqliteSaver(connection)})
        finally:
            connection.close()

    def _graph_template(self, workflow_kind: str):
        kind = (
            "project_factory"
            if workflow_kind == "project_factory"
            else "maintenance"
        )
        with self._graph_templates_lock:
            template = self._graph_templates.get(kind)
            if template is None:
                if kind == "project_factory":
                    template = build_studio_workflow(self.studio_suite, None)
                else:
                    template = build_workflow(self.suite, None)
                self._graph_templates[kind] = template
            return template

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database,
            timeout=30,
            check_same_thread=False,
        )
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RepoPilotService is closed")

    @staticmethod
    def _config(run_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": run_id}}
