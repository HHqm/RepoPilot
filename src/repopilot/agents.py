from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from typing import Any

from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt

from .config import Settings
from .context import ContextCompressor
from .llm import (
    LLMConfigurationError,
    StructuredLLM,
    build_llm,
    build_routed_llm,
)
from .model_routing import ModelProviderRegistry
from .models import (
    ArchitectureBlueprint,
    ChangePlan,
    ChangeProposal,
    CodeEditSet,
    DeploymentGuide,
    GeneratedProject,
    ProductChatResponse,
    ProductRequirementsDocument,
    ProjectExecutionPlan,
    ProjectRevision,
    QualityAssuranceReport,
    Reflection,
    RequirementsWorkshop,
    ReviewDecision,
)
from .prompts import (
    CODER_SYSTEM,
    PLANNER_SYSTEM,
    REFLECTION_SYSTEM,
    REVIEWER_SYSTEM,
)
from .rag import HybridRAG, IndexedRepository
from .rag.multi_query import fuse_rankings, plan_queries
from .repo_tools import PatchApplier, TestRunner, artifact_digest, unified_diff
from .run_metrics import render_run_metrics, summarize_run_metrics
from .security import resolve_safe_path, scan_content
from .state import RepoPilotState


class TargetedEditMismatch(ValueError):
    def __init__(self, path: str, occurrences: int):
        self.path = path
        self.occurrences = occurrences
        super().__init__(
            "Targeted edit search must match exactly once in "
            f"{path}; found {occurrences}"
        )


class AgentSuite:
    _SCHEMA_MODELS = {
        "product_chat": ProductChatResponse,
        "change_plan": ChangePlan,
        "code_edits": CodeEditSet,
        "review_decision": ReviewDecision,
        "reflection": Reflection,
        "product_requirements": ProductRequirementsDocument,
        "requirements_workshop": RequirementsWorkshop,
        "project_plan": ProjectExecutionPlan,
        "architecture_blueprint": ArchitectureBlueprint,
        "generated_project": GeneratedProject,
        "project_revision": ProjectRevision,
        "deployment_guide": DeploymentGuide,
        "qa_report": QualityAssuranceReport,
        "studio_review": ReviewDecision,
    }

    def __init__(
        self,
        settings: Settings,
        model_registry: ModelProviderRegistry | None = None,
    ):
        self.settings = settings
        self.model_registry = model_registry
        self.rag = HybridRAG.from_settings(settings)
        self.applier = PatchApplier(settings.runtime_dir)
        self.test_runner = TestRunner(
            settings.test_timeout_seconds,
            allow_host_execution=settings.allow_host_test_execution,
        )
        self.context_compressor = ContextCompressor.from_settings(settings)
        self._indices: OrderedDict[str, IndexedRepository] = OrderedDict()
        self._llms: dict[str, StructuredLLM] = {}
        self._cache_lock = RLock()

    def _llm(
        self, state: RepoPilotState, schema_name: str = ""
    ) -> StructuredLLM:
        mode = state.get("llm_mode", self.settings.llm_mode)
        routing = state.get("model_routing")
        if mode == "routed" or routing:
            if not routing:
                raise LLMConfigurationError(
                    "llm_mode=routed requires a model_routing contract"
                )
            if self.model_registry is None:
                raise LLMConfigurationError(
                    "Model provider registry is unavailable"
                )
            route = self.model_registry.resolve(routing, schema_name)
            cache_key = str(route["cache_key"])
            with self._cache_lock:
                if cache_key not in self._llms:
                    self._llms[cache_key] = build_routed_llm(
                        route,
                        self.settings,
                    )
                return self._llms[cache_key]
        with self._cache_lock:
            if mode not in self._llms:
                self._llms[mode] = build_llm(mode, self.settings)
            return self._llms[mode]

    def _complete_json(
        self,
        state: RepoPilotState,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        schema_name: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        compressed, context_stats = self.context_compressor.compress(
            schema_name,
            payload,
        )
        llm = self._llm(state, schema_name)
        result = llm.complete_json(
            system_prompt=system_prompt,
            payload=compressed,
            schema_name=schema_name,
        )
        # Read usage right after each call: the adapter keeps only the latest.
        api_usage = getattr(llm, "last_usage", None)
        if api_usage:
            context_stats["api_usage"] = dict(api_usage)
        repair_attempts = 0
        schema_model = self._SCHEMA_MODELS.get(schema_name)
        if schema_model is not None:
            try:
                result = schema_model.model_validate(result).model_dump()
            except Exception as exc:
                repair_attempts = 1
                validation_error = str(exc)[:4000]
                repair_payload = {
                    "original_task_input": compressed,
                    "invalid_response": result,
                    "validation_error": validation_error,
                    "repair_rule": (
                        "Preserve valid facts. Fix only schema/type/required-field "
                        "errors. Return one JSON object and no markdown."
                    ),
                }
                repaired, repair_context = self.context_compressor.compress(
                    "schema_repair",
                    repair_payload,
                )
                result = llm.complete_json(
                    system_prompt=(
                        system_prompt
                        + "\n\nYour previous response failed deterministic schema "
                        "validation. Repair it once using the supplied validation "
                        "error. Do not invent facts to fill optional fields."
                    ),
                    payload=repaired,
                    schema_name=schema_name,
                )
                repair_usage = getattr(llm, "last_usage", None)
                if repair_usage:
                    repair_context["api_usage"] = dict(repair_usage)
                result = schema_model.model_validate(result).model_dump()
                context_stats["repair_context"] = repair_context
        route_metadata = getattr(llm, "route_metadata", None)
        if route_metadata:
            context_stats["model_route"] = dict(route_metadata)
        context_stats["repair_attempts"] = repair_attempts
        return result, context_stats

    def guarded(
        self,
        agent_name: str,
        node: Callable[[RepoPilotState], dict[str, Any]],
    ) -> Callable[[RepoPilotState], dict[str, Any]]:
        """Convert node exceptions into structured workflow state.

        LangGraph's GraphInterrupt is deliberately re-raised because it is the
        control signal used by the human approval checkpoint.
        """

        def execute(state: RepoPilotState) -> dict[str, Any]:
            started = time.perf_counter()
            try:
                return node(state)
            except GraphInterrupt:
                raise
            except Exception as exc:
                message = str(exc).strip() or exc.__class__.__name__
                message = self.redact_secrets(message)
                message = message[:1000]
                return {
                    "status": "failed",
                    "error": {
                        "agent": agent_name,
                        "type": exc.__class__.__name__,
                        "message": message,
                    },
                    "traces": [
                        self._trace(
                            agent_name,
                            started,
                            f"Failed safely: {message}",
                            status="failed",
                        )
                    ],
                }

        return execute

    def close(self) -> None:
        with self._cache_lock:
            self._indices.clear()
            for llm in self._llms.values():
                close = getattr(llm, "close", None)
                if callable(close):
                    close()
            self._llms.clear()
        self.rag.close()

    def invalidate_credential(self, credential_id: str) -> None:
        prefix = f"routed:{credential_id}:"
        with self._cache_lock:
            keys = [key for key in self._llms if key.startswith(prefix)]
            for key in keys:
                llm = self._llms.pop(key)
                close = getattr(llm, "close", None)
                if callable(close):
                    close()

    def redact_secrets(self, text: str) -> str:
        redacted = self.settings.redact_secrets(text)
        if self.model_registry is not None:
            redacted = self.model_registry.redact(redacted)
        return redacted

    @staticmethod
    def _trace(
        agent: str,
        started: float,
        summary: str,
        *,
        status: str = "success",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "agent": agent,
            "status": status,
            "summary": summary,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "metadata": metadata or {},
        }

    def intake_agent(self, state: RepoPilotState) -> dict[str, Any]:
        started = time.perf_counter()
        root = Path(state["repo_path"]).resolve()
        if not root.is_dir():
            raise ValueError(f"Repository does not exist: {root}")
        self.test_runner.validate_command(state["test_command"])
        return {
            "repo_path": str(root),
            "status": "mapping_repository",
            "traces": [
                self._trace(
                    "IntakeAgent",
                    started,
                    "Validated repository, issue, and allowlisted test command.",
                )
            ],
        }

    def repo_mapper_agent(self, state: RepoPilotState) -> dict[str, Any]:
        started = time.perf_counter()
        index = self.rag.index_repository(state["repo_path"])
        with self._cache_lock:
            self._indices[state["run_id"]] = index
            self._indices.move_to_end(state["run_id"])
            while len(self._indices) > self.settings.max_cached_indices:
                self._indices.popitem(last=False)
        repo_map = index.repo_map()
        return {
            "repo_map": repo_map,
            "status": "retrieving_context",
            "traces": [
                self._trace(
                    "RepoMapperAgent",
                    started,
                    (
                        f"Mapped {len(repo_map)} files and {index.chunk_count} "
                        "persistent hybrid-search chunks."
                    ),
                    metadata={
                        "files": len(repo_map),
                        "chunks": index.chunk_count,
                        "indexed_chunks": index.indexed_chunks,
                        "reused_files": index.reused_files,
                        "deleted_files": index.deleted_files,
                        "skipped_files": index.skipped_files,
                        "collection": index.collection_name,
                    },
                )
            ],
        }

    def retrieval_agent(self, state: RepoPilotState) -> dict[str, Any]:
        started = time.perf_counter()
        index = self._get_index(state)
        history = state.get("attempt_history") or []
        queries = plan_queries(
            state["issue"],
            state.get("constraints", []),
            last_failure=history[-1] if history else None,
        )
        rankings = []
        query_reports = []
        for planned in queries:
            ranking, stats = self.rag.search_with_stats(
                index,
                planned.text,
                top_k=self.settings.rag_top_k,
            )
            rankings.append(ranking)
            query_reports.append(
                {
                    **planned.to_dict(),
                    "top_paths": list(
                        dict.fromkeys(item.path for item in ranking)
                    )[:5],
                    "stats": stats.to_dict(),
                }
            )
        evidence = fuse_rankings(rankings, top_k=self.settings.rag_top_k)
        retrieval = {
            "queries": query_reports,
            "fusion": "reciprocal_rank_fusion" if len(queries) > 1 else "none",
            "evidence_count": len(evidence),
        }
        purposes = "+".join(item.purpose for item in queries)
        return {
            "evidence": [item.model_dump() for item in evidence],
            "retrieval": retrieval,
            "status": "planning_change",
            "traces": [
                self._trace(
                    "RetrievalAgent",
                    started,
                    (
                        f"Retrieved {len(evidence)} grounded chunks from "
                        f"{len(queries)} planned quer"
                        f"{'y' if len(queries) == 1 else 'ies'} ({purposes})."
                    ),
                    metadata={
                        "top_paths": list(
                            dict.fromkeys(item.path for item in evidence)
                        ),
                        "retrieval": retrieval,
                        "pipeline": "qdrant_dense_sparse_rrf_rerank",
                        "embedding_model": self.rag.store.embedding.model_id,
                        "sparse_model": self.rag.store.embedding.sparse_model_id,
                        "reranker_model": self.rag.store.reranker.model_id,
                    },
                )
            ],
        }

    def planner_agent(self, state: RepoPilotState) -> dict[str, Any]:
        started = time.perf_counter()
        data, context_stats = self._complete_json(
            state,
            system_prompt=PLANNER_SYSTEM,
            schema_name="change_plan",
            payload={
                "issue": state["issue"],
                "constraints": state.get("constraints", []),
                "repo_map": state["repo_map"],
                "evidence": state["evidence"],
                **self._attempt_context(state),
            },
        )
        plan = ChangePlan.model_validate(data)
        known_files = {item["path"] for item in state["repo_map"]}
        known_evidence = {item["evidence_id"] for item in state["evidence"]}
        for step in plan.steps:
            unknown_files = set(step.target_files) - known_files
            if unknown_files:
                raise ValueError(
                    f"Planner referenced unknown files: {sorted(unknown_files)}"
                )
            unknown_evidence = set(step.evidence_ids) - known_evidence
            if unknown_evidence:
                raise ValueError(
                    f"Planner referenced unknown evidence: {sorted(unknown_evidence)}"
                )
        attempt = int(state.get("repair_attempt", 0))
        return {
            "plan": plan.model_dump(),
            "status": "generating_patch",
            "traces": [
                self._trace(
                    "PlannerAgent",
                    started,
                    f"Created a {len(plan.steps)}-step grounded change plan"
                    + (
                        f" for repair attempt {attempt}."
                        if attempt
                        else "."
                    ),
                    metadata={"context": context_stats, "repair_attempt": attempt},
                )
            ],
        }

    @staticmethod
    def _attempt_context(state: RepoPilotState) -> dict[str, Any]:
        """Failed attempts are only sent when they exist, keeping the first
        attempt's payload identical to a run without self-correction."""

        history = state.get("attempt_history") or []
        return {"previous_attempts": history} if history else {}

    def coder_agent(self, state: RepoPilotState) -> dict[str, Any]:
        started = time.perf_counter()
        target_files = list(
            dict.fromkeys(
                path
                for step in state["plan"]["steps"]
                for path in step["target_files"]
            )
        )
        if not target_files:
            raise ValueError("Plan contains no target files")
        files: list[dict[str, Any]] = []
        originals: dict[str, str] = {}
        source_context_chars = min(
            self.settings.max_file_bytes,
            self.settings.context_max_file_item_tokens * 5,
        )
        for relative in target_files:
            path = resolve_safe_path(state["repo_path"], relative)
            content = path.read_text(encoding="utf-8", errors="replace")
            originals[relative] = content
            excerpts: list[str] = []
            remaining = source_context_chars
            for evidence in state.get("evidence", []):
                if evidence.get("path") != relative or remaining <= 0:
                    continue
                location = self._evidence_location(evidence)
                raw_excerpt = str(evidence.get("text", ""))
                excerpt = (
                    f"[{evidence.get('evidence_id', 'evidence')} "
                    f"{location}]\n{raw_excerpt}"
                )[:remaining]
                if excerpt:
                    excerpts.append(excerpt)
                    remaining -= len(excerpt)
            if not excerpts:
                excerpts.append(content[:source_context_chars])
            files.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "excerpts": excerpts,
                }
            )
        feedback: dict[str, Any] = dict(self._attempt_context(state))
        if state.get("status") == "revising_patch" and state.get("review_feedback"):
            feedback["review_feedback"] = state["review_feedback"]
        draft_data, context_stats = self._complete_json(
            state,
            system_prompt=CODER_SYSTEM,
            schema_name="code_edits",
            payload={
                "issue": state["issue"],
                "constraints": state.get("constraints", []),
                "plan": state["plan"],
                "evidence": state["evidence"],
                "files": files,
                **feedback,
            },
        )
        drafts = CodeEditSet.model_validate(draft_data)
        allowed = set(target_files)
        try:
            edited, reasons = self._apply_targeted_edits(
                drafts,
                originals,
                allowed,
            )
        except TargetedEditMismatch as mismatch:
            # Bounded read-only tool recovery: expose the current target file
            # once, then ask the model to repair its exact-match edit. Writing
            # remains unavailable until deterministic review and human approval.
            repair_files = [
                {
                    "path": mismatch.path,
                    "size": len(originals[mismatch.path].encode("utf-8")),
                    "excerpts": [originals[mismatch.path]],
                }
            ]
            repaired_data, repair_stats = self._complete_json(
                state,
                system_prompt=(
                    CODER_SYSTEM
                    + "\n\nThe previous exact-match edit failed. A bounded "
                    "read_file tool result is supplied. Repair the edit once; "
                    "the search text must occur exactly once."
                ),
                schema_name="code_edits",
                payload={
                    "issue": state["issue"],
                    "constraints": state.get("constraints", []),
                    "plan": state["plan"],
                    "evidence": state["evidence"],
                    "files": repair_files,
                    "tool_error": str(mismatch),
                    **feedback,
                },
            )
            repaired = CodeEditSet.model_validate(repaired_data)
            edited, reasons = self._apply_targeted_edits(
                repaired,
                originals,
                allowed,
            )
            context_stats["bounded_tool_repair"] = {
                "tool": "read_file",
                "path": mismatch.path,
                "attempts": 1,
                "context": repair_stats,
            }

        index = self._get_index(state)
        proposals: list[ChangeProposal] = []
        for path, new_content in edited.items():
            original = originals[path]
            if original == new_content:
                continue
            proposals.append(
                ChangeProposal(
                    path=path,
                    reasoning=" ".join(reasons.get(path, [])),
                    original_sha256=index.files[path].sha256,
                    new_content=new_content,
                    diff=unified_diff(path, original, new_content),
                )
            )
        if not proposals:
            raise ValueError("Coder proposed no effective targeted edits")
        revising = "review_feedback" in feedback
        return {
            "proposals": [item.model_dump() for item in proposals],
            "status": "reviewing_patch",
            "traces": [
                self._trace(
                    "CoderAgent",
                    started,
                    f"Proposed {len(proposals)} file replacement(s)"
                    + (" addressing review findings." if revising else "."),
                    metadata={
                        "paths": [item.path for item in proposals],
                        "addresses_review": revising,
                        "context": context_stats,
                    },
                )
            ],
        }

    def _apply_targeted_edits(
        self,
        drafts: CodeEditSet,
        originals: dict[str, str],
        allowed: set[str],
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
        edited = dict(originals)
        reasons: dict[str, list[str]] = {}
        for draft in drafts.edits:
            if draft.path not in allowed:
                raise PermissionError(
                    f"Coder attempted an unplanned file: {draft.path}"
                )
            if len(draft.replace.encode("utf-8")) > self.settings.max_file_bytes:
                raise ValueError(
                    f"Coder replacement is too large for {draft.path}"
                )
            current = edited[draft.path]
            occurrences = current.count(draft.search)
            if occurrences != 1:
                raise TargetedEditMismatch(draft.path, occurrences)
            edited[draft.path] = current.replace(
                draft.search, draft.replace, 1
            )
            reasons.setdefault(draft.path, []).append(draft.reasoning)
        return edited, reasons

    def reviewer_agent(self, state: RepoPilotState) -> dict[str, Any]:
        started = time.perf_counter()
        deterministic_findings: list[dict[str, str]] = []
        for proposal in state["proposals"]:
            deterministic_findings.extend(
                scan_content(proposal["path"], proposal["new_content"])
            )
        all_diffs = "\n".join(item["diff"] for item in state["proposals"])
        review_data, context_stats = self._complete_json(
            state,
            system_prompt=REVIEWER_SYSTEM,
            schema_name="review_decision",
            payload={
                "issue": state["issue"],
                "constraints": state.get("constraints", []),
                "plan": state["plan"],
                "evidence": state["evidence"],
                "diffs": all_diffs,
            },
        )
        if deterministic_findings:
            review_data["approved"] = False
            review_data.setdefault("findings", []).extend(deterministic_findings)
            review_data["summary"] = (
                "Rejected by deterministic safety checks; "
                f"{len(deterministic_findings)} blocking finding(s) detected."
            )
        review = ReviewDecision.model_validate(review_data)
        revision = int(state.get("review_revision", 0))
        update: dict[str, Any] = {"review": review.model_dump()}
        if review.approved:
            update["status"] = "awaiting_approval"
        elif revision < self.settings.max_review_revisions:
            # Critic loop: hand the findings back to the coder instead of
            # ending the run. Deterministic safety findings are included, and
            # the revised patch must pass this same gate again.
            update.update(
                status="revising_patch",
                review_revision=revision + 1,
                review_feedback={
                    "summary": review.summary,
                    "findings": [item.model_dump() for item in review.findings],
                    "rejected_diffs": all_diffs,
                },
            )
        else:
            update["status"] = "review_rejected"
        update["traces"] = [
            self._trace(
                "ReviewerAgent",
                started,
                review.summary,
                metadata={
                    "approved": review.approved,
                    "findings": len(review.findings),
                    "review_revision": revision,
                    "revision_requested": update["status"] == "revising_patch",
                    "context": context_stats,
                },
            )
        ]
        return update

    def approval_agent(self, state: RepoPilotState) -> dict[str, Any]:
        started = time.perf_counter()
        approval_material = {
            "review": state["review"],
            "diffs": [
                {"path": item["path"], "diff": item["diff"]}
                for item in state["proposals"]
            ],
        }
        decision = interrupt(
            {
                "run_id": state["run_id"],
                "question": "Approve the reviewed file changes?",
                **approval_material,
                "artifact_digest": artifact_digest(approval_material),
            }
        )
        if isinstance(decision, bool):
            approved, comment = decision, ""
        else:
            approved = bool(decision.get("approved", False))
            comment = str(decision.get("comment", ""))
        return {
            "approved": approved,
            "approval_comment": comment,
            "status": "applying_patch" if approved else "cancelled",
            "traces": [
                self._trace(
                    "ApprovalAgent",
                    started,
                    "Human approved the patch."
                    if approved
                    else "Human rejected the patch.",
                    metadata={"approved": approved},
                )
            ],
        }

    def executor_agent(self, state: RepoPilotState) -> dict[str, Any]:
        started = time.perf_counter()
        if not state.get("approved"):
            raise PermissionError("Patch execution requires explicit approval")
        result = self.applier.apply(
            state["repo_path"], state["run_id"], state["proposals"]
        )
        return {
            "apply_result": result,
            "status": "running_tests",
            "traces": [
                self._trace(
                    "ExecutorAgent",
                    started,
                    f"Atomically changed {len(result['changed_files'])} file(s).",
                    metadata={"changed_files": result["changed_files"]},
                )
            ],
        }

    def test_agent(self, state: RepoPilotState) -> dict[str, Any]:
        started = time.perf_counter()
        result = self.test_runner.run(state["repo_path"], state["test_command"])
        return {
            "test_result": result,
            "status": "verifying_result",
            "traces": [
                self._trace(
                    "TestAgent",
                    started,
                    "Allowlisted tests passed."
                    if result["success"]
                    else "Allowlisted tests failed.",
                    status="success" if result["success"] else "failed",
                    metadata={"returncode": result["returncode"]},
                )
            ],
        }

    def verifier_agent(self, state: RepoPilotState) -> dict[str, Any]:
        started = time.perf_counter()
        success = bool(state["test_result"]["success"])
        verification: dict[str, Any] = {
            "success": success,
            "verdict": "accepted" if success else "failed_and_rolled_back",
            "rollback": None,
        }
        if not success:
            verification["rollback"] = self.applier.rollback(
                state["repo_path"],
                state["run_id"],
                state["apply_result"]["changed_files"],
                expected_applied_sha256=state["apply_result"].get(
                    "applied_sha256"
                ),
            )
        return {
            "verification": verification,
            "status": "completed" if success else "reflecting_on_failure",
            "traces": [
                self._trace(
                    "VerifierAgent",
                    started,
                    "Verified patch with passing tests."
                    if success
                    else "Verification failed; restored all changed files.",
                    status="success" if success else "failed",
                )
            ],
        }

    def reflection_agent(self, state: RepoPilotState) -> dict[str, Any]:
        started = time.perf_counter()
        data, context_stats = self._complete_json(
            state,
            system_prompt=REFLECTION_SYSTEM,
            schema_name="reflection",
            payload={
                "issue": state["issue"],
                "plan": state["plan"],
                "diffs": [item["diff"] for item in state["proposals"]],
                "test_result": state["test_result"],
                "verification": state["verification"],
            },
        )
        reflection = Reflection.model_validate(data)
        attempt = int(state.get("repair_attempt", 0))
        history = list(state.get("attempt_history") or [])
        history.append(
            {
                "attempt": attempt,
                "plan_summary": state["plan"].get("summary", ""),
                "changed_files": [item["path"] for item in state["proposals"]],
                "diff": "\n".join(item["diff"] for item in state["proposals"])[
                    :4_000
                ],
                "test_failure": self._test_failure_excerpt(state["test_result"]),
                "root_cause": reflection.root_cause,
                "next_action": reflection.next_action,
            }
        )
        retry = (
            reflection.retry_recommended
            and attempt < self.settings.max_repair_attempts
        )
        update: dict[str, Any] = {
            "reflection": reflection.model_dump(),
            "attempt_history": history,
        }
        if retry:
            # The verifier already restored every file. Clear the attempt's
            # artifacts so the next patch is re-reviewed and re-approved by a
            # human from scratch; only the lessons in attempt_history carry on.
            update.update(
                status="replanning",
                repair_attempt=attempt + 1,
                proposals=[],
                review={},
                review_revision=0,
                review_feedback={},
                approved=False,
                approval_comment="",
                apply_result={},
                test_result={},
                verification={},
            )
            summary = (
                f"{reflection.root_cause} Starting repair attempt "
                f"{attempt + 1}/{self.settings.max_repair_attempts}."
            )
        else:
            update["status"] = "failed_rolled_back"
            summary = reflection.root_cause
        update["traces"] = [
            self._trace(
                "ReflectionAgent",
                started,
                summary,
                status="failed",
                metadata={
                    "context": context_stats,
                    "retry": retry,
                    "repair_attempt": attempt,
                    "retry_recommended": reflection.retry_recommended,
                },
            )
        ]
        return update

    @staticmethod
    def _test_failure_excerpt(test_result: dict[str, Any]) -> str:
        output = "\n".join(
            part
            for part in (
                str(test_result.get("stdout") or ""),
                str(test_result.get("stderr") or ""),
            )
            if part
        )
        return output[-2_000:]

    def reporter_agent(self, state: RepoPilotState) -> dict[str, Any]:
        started = time.perf_counter()
        status = state.get("status", "unknown")
        if status == "review_rejected":
            final_status = "review_rejected"
        elif status == "cancelled":
            final_status = "cancelled"
        elif state.get("verification", {}).get("success"):
            final_status = "completed"
        elif state.get("verification"):
            final_status = "failed_rolled_back"
        else:
            final_status = status
        evidence_lines = "\n".join(
            f"- `{item['evidence_id']}` `{self._evidence_location(item)}` "
            f"fusion={item['fusion_score']} rerank={item['rerank_score']}"
            for item in state.get("evidence", [])
        )
        proposal_lines = "\n".join(
            f"- `{item['path']}` — {item['reasoning']}"
            for item in state.get("proposals", [])
        )
        trace_lines = "\n".join(
            f"- **{item['agent']}** `{item['status']}` "
            f"{item['latency_ms']} ms — {item['summary']}"
            for item in state.get("traces", [])
        )
        test_result = state.get("test_result")
        test_text = (
            f"returncode={test_result['returncode']}, "
            f"success={test_result['success']}"
            if test_result
            else "not executed"
        )
        attempt_lines = "\n".join(
            f"- Attempt {item['attempt'] + 1}: changed "
            f"{', '.join(f'`{path}`' for path in item['changed_files']) or 'nothing'}"
            f" — tests failed; root cause: {item['root_cause']}"
            for item in state.get("attempt_history", [])
        )
        query_lines = "\n".join(
            f"- `{item['purpose']}`: {item['query'][:120]}"
            for item in (state.get("retrieval") or {}).get("queries", [])
        )
        metrics = summarize_run_metrics(state.get("traces", []))
        report = f"""# RepoPilot Run {state['run_id']}

## Outcome

- Status: `{final_status}`
- Issue: {state['issue']}
- LLM mode: `{state.get('llm_mode')}`
- Test: {test_text}
- Repair attempts used: {state.get('repair_attempt', 0)}/{self.settings.max_repair_attempts}
- Review revisions used in final attempt: {state.get('review_revision', 0)}/{self.settings.max_review_revisions}

## Self-correction history

{attempt_lines or '- First attempt; no failed attempts.'}

## Retrieval plan

{query_lines or '- none'}

## Retrieved evidence

{evidence_lines or '- none'}

## Proposed changes

{proposal_lines or '- none'}

## Review

```json
{json.dumps(state.get('review', {}), ensure_ascii=False, indent=2)}
```

## Cost and latency

{render_run_metrics(metrics)}

## Agent trace

{trace_lines or '- none'}
"""
        result = {
            "report": report,
            "run_metrics": metrics,
            "status": final_status,
            "traces": [
                self._trace(
                    "ReporterAgent",
                    started,
                    f"Generated final report with status {final_status}.",
                )
            ],
        }
        self._release_index(state["run_id"])
        return result

    def failure_reporter_agent(self, state: RepoPilotState) -> dict[str, Any]:
        started = time.perf_counter()
        error = dict(state.get("error") or {})
        rollback: dict[str, Any] | None = None
        changed_files = state.get("apply_result", {}).get("changed_files", [])
        already_rolled_back = state.get("verification", {}).get("rollback")
        if changed_files and not already_rolled_back:
            try:
                rollback = self.applier.rollback(
                    state["repo_path"],
                    state["run_id"],
                    changed_files,
                    expected_applied_sha256=state.get(
                        "apply_result", {}
                    ).get("applied_sha256"),
                )
            except Exception as rollback_error:
                rollback = {
                    "success": False,
                    "error": str(rollback_error)[:1000],
                }
        if rollback is not None:
            error["rollback"] = rollback
        agent = error.get("agent", "unknown")
        error_type = error.get("type", "RuntimeError")
        message = error.get("message", "Unknown workflow failure")
        metrics = summarize_run_metrics(state.get("traces", []))
        report = f"""# RepoPilot Run {state['run_id']} — Failed safely

## Outcome

- Status: `failed`
- Failed agent: `{agent}`
- Error type: `{error_type}`
- Message: {message}
- Repository rollback: `{json.dumps(rollback, ensure_ascii=False) if rollback else 'not required'}`

## Issue

{state.get('issue', '(missing)')}

## Safety note

The exception was converted into structured state and the workflow ended through
the failure reporter instead of exposing an unhandled server exception.

## Cost and latency

{render_run_metrics(metrics)}
"""
        self._release_index(state["run_id"])
        return {
            "status": "failed",
            "error": error,
            "run_metrics": metrics,
            "report": report,
            "traces": [
                self._trace(
                    "FailureReporterAgent",
                    started,
                    f"Recorded {error_type} from {agent} and ended safely.",
                    status="failed",
                    metadata={"failed_agent": agent, "rollback": rollback},
                )
            ],
        }

    def _get_index(self, state: RepoPilotState) -> IndexedRepository:
        run_id = state["run_id"]
        with self._cache_lock:
            index = self._indices.get(run_id)
            if index is not None:
                self._indices.move_to_end(run_id)
                return index
        index = self.rag.index_repository(state["repo_path"])
        with self._cache_lock:
            self._indices[run_id] = index
            self._indices.move_to_end(run_id)
            while len(self._indices) > self.settings.max_cached_indices:
                self._indices.popitem(last=False)
        return index

    def _release_index(self, run_id: str) -> None:
        with self._cache_lock:
            self._indices.pop(run_id, None)

    @staticmethod
    def _evidence_location(item: dict[str, Any]) -> str:
        if item.get("page_start") is not None:
            start = item["page_start"]
            end = item.get("page_end") or start
            return f"{item['path']}#page={start}-{end}"
        if item.get("start_line") is not None:
            start = item["start_line"]
            end = item.get("end_line") or start
            return f"{item['path']}:{start}-{end}"
        return str(item["path"])
